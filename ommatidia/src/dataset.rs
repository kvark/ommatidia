//! The `.omd` training set container.
//!
//! A dataset is a fixed-size header followed by tightly packed records, one
//! per sample. Each record is the low resolution planes followed by the high
//! resolution ones, every plane stored channel-major so a record is already
//! in NCHW order and can be handed to meganeura without a shuffle.
//!
//! Values are `f16` throughout, and are the physical quantities the renderer
//! produced rather than anything the network wants. `f16` spends its bits on
//! an exponent, so it holds high dynamic range radiance at roughly 0.1%
//! relative precision all the way up; pre-conditioning the values for the
//! network would throw that away. The conversion to network inputs lives in
//! [`crate::transform`], applied on load.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use half::f16;

/// Magic at the start of every `.omd` file.
pub const MAGIC: [u8; 8] = *b"OMMATIDA";
/// Format revision. Bumped on any layout change.
pub const VERSION: u32 = 1;
/// Byte size of the header, including the reserved tail.
pub const HEADER_SIZE: usize = 64;

/// A group of channels a sample can carry.
///
/// The set present in a file is recorded in the header, so a dataset with
/// more planes than a given model consumes stays readable.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[repr(u32)]
pub enum Plane {
    /// Linear radiance, unmodified.
    ///
    /// Clamped only at 65504, where `f16` runs out; nothing short of a sun
    /// disc reaches that after exposure.
    Color = 0,
    /// View-space distance from the camera.
    Depth = 1,
    /// World-space shading normal, in `[-1, 1]`.
    Normal = 2,
    /// Base colour with the specularly reflected part taken out, in `[0, 1]`.
    DiffuseAlbedo = 3,
    /// Specular reflectance at normal incidence, in `[0, 1]`.
    SpecularF0 = 4,
    /// Surface roughness, in `[0, 1]`.
    Roughness = 5,
    /// Screen-space motion since the previous frame, in pixels.
    ///
    /// Unused by the static model; reserved so that adding temporal context
    /// does not invalidate datasets generated before it.
    Motion = 6,
}

/// Every plane, in the order they are stored in a record.
pub const ALL_PLANES: [Plane; 7] = [
    Plane::Color,
    Plane::Depth,
    Plane::Normal,
    Plane::DiffuseAlbedo,
    Plane::SpecularF0,
    Plane::Roughness,
    Plane::Motion,
];

impl Plane {
    /// Number of channels this plane contributes.
    pub const fn channels(self) -> usize {
        match self {
            Self::Color | Self::Normal | Self::DiffuseAlbedo | Self::SpecularF0 => 3,
            Self::Depth | Self::Roughness => 1,
            Self::Motion => 2,
        }
    }

    const fn bit(self) -> u32 {
        1 << (self as u32)
    }
}

/// The set of planes a record carries, as an ordered bit set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlaneSet(u32);

impl PlaneSet {
    /// The empty set.
    pub const fn new() -> Self {
        Self(0)
    }

    /// Add a plane. Idempotent.
    pub const fn with(self, plane: Plane) -> Self {
        Self(self.0 | plane.bit())
    }

    /// Is this plane present?
    pub const fn contains(self, plane: Plane) -> bool {
        self.0 & plane.bit() != 0
    }

    /// Total channel count across the whole set.
    pub fn channels(self) -> usize {
        self.iter().map(Plane::channels).sum()
    }

    /// Planes in storage order.
    pub fn iter(self) -> impl Iterator<Item = Plane> {
        ALL_PLANES.into_iter().filter(move |&p| self.contains(p))
    }

    /// Offset, in channels, of `plane` within a record's low or high
    /// resolution block. `None` if the plane is absent.
    pub fn channel_offset(self, plane: Plane) -> Option<usize> {
        if !self.contains(plane) {
            return None;
        }
        Some(
            self.iter()
                .take_while(|&p| p != plane)
                .map(Plane::channels)
                .sum(),
        )
    }

    /// Raw bit representation, as stored in the header.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Rebuild from the header representation, rejecting unknown bits so a
    /// file from a newer writer fails loudly instead of being misread.
    pub fn from_bits(bits: u32) -> Result<Self, Error> {
        let known = ALL_PLANES.iter().fold(0, |acc, p| acc | p.bit());
        if bits & !known != 0 {
            return Err(Error::UnknownPlane(bits & !known));
        }
        Ok(Self(bits))
    }
}

impl FromIterator<Plane> for PlaneSet {
    fn from_iter<I: IntoIterator<Item = Plane>>(iter: I) -> Self {
        iter.into_iter().fold(Self::new(), Self::with)
    }
}

/// What a dataset holds, independent of how many samples are in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Layout {
    /// High resolution is the low resolution extent times this.
    pub scale: u32,
    pub lr_width: u32,
    pub lr_height: u32,
    /// Planes stored at low resolution: the network's conditioning.
    pub lr_planes: PlaneSet,
    /// Planes stored at high resolution: the reference the network fits.
    pub hr_planes: PlaneSet,
}

impl Layout {
    pub fn hr_width(&self) -> u32 {
        self.lr_width * self.scale
    }

    pub fn hr_height(&self) -> u32 {
        self.lr_height * self.scale
    }

    /// Texels in one low resolution plane channel.
    pub fn lr_texels(&self) -> usize {
        self.lr_width as usize * self.lr_height as usize
    }

    /// Texels in one high resolution plane channel.
    pub fn hr_texels(&self) -> usize {
        self.hr_width() as usize * self.hr_height() as usize
    }

    /// `f16` values in the low resolution block of a record.
    pub fn lr_len(&self) -> usize {
        self.lr_planes.channels() * self.lr_texels()
    }

    /// `f16` values in the high resolution block of a record.
    pub fn hr_len(&self) -> usize {
        self.hr_planes.channels() * self.hr_texels()
    }

    /// `f16` values in a whole record.
    pub fn record_len(&self) -> usize {
        self.lr_len() + self.hr_len()
    }

    fn record_bytes(&self) -> u64 {
        self.record_len() as u64 * 2
    }
}

/// One sample: the conditioning planes and the reference planes.
///
/// Both are flat channel-major blocks sized by [`Layout::lr_len`] and
/// [`Layout::hr_len`].
#[derive(Clone, Debug)]
pub struct Sample {
    pub lr: Vec<f16>,
    pub hr: Vec<f16>,
}

impl Sample {
    /// Borrow one channel of one low resolution plane.
    ///
    /// `channel` is relative to the plane, so `Plane::Normal` channel 1 is Y.
    pub fn lr_channel(&self, layout: &Layout, plane: Plane, channel: usize) -> Option<&[f16]> {
        let base = layout.lr_planes.channel_offset(plane)? + channel;
        let texels = layout.lr_texels();
        self.lr.get(base * texels..(base + 1) * texels)
    }

    /// Borrow one channel of one high resolution plane.
    pub fn hr_channel(&self, layout: &Layout, plane: Plane, channel: usize) -> Option<&[f16]> {
        let base = layout.hr_planes.channel_offset(plane)? + channel;
        let texels = layout.hr_texels();
        self.hr.get(base * texels..(base + 1) * texels)
    }
}

/// Largest magnitude `f16` can hold. Values are clamped here on write.
pub const F16_MAX: f32 = 65504.0;

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    /// The file does not start with [`MAGIC`].
    BadMagic,
    /// Written by a different format revision.
    Version(u32),
    /// A plane bit this build does not know about.
    UnknownPlane(u32),
    /// Header describes a layout with a zero extent or scale.
    EmptyLayout,
    /// The file is shorter than its header claims.
    Truncated {
        expected: u64,
        actual: u64,
    },
    /// A sample handed to the writer is not the size the layout implies.
    SampleSize {
        expected: usize,
        actual: usize,
    },
    /// Requested a sample past the end.
    OutOfRange {
        index: usize,
        count: usize,
    },
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Io(ref e) => write!(f, "{e}"),
            Self::BadMagic => write!(f, "not an ommatidia dataset"),
            Self::Version(v) => write!(f, "unsupported format version {v}, expected {VERSION}"),
            Self::UnknownPlane(bits) => write!(f, "unknown plane bits {bits:#x}"),
            Self::EmptyLayout => write!(f, "layout has a zero extent or scale"),
            Self::Truncated { expected, actual } => {
                write!(f, "truncated: expected {expected} bytes, found {actual}")
            }
            Self::SampleSize { expected, actual } => {
                write!(f, "sample has {actual} values, layout implies {expected}")
            }
            Self::OutOfRange { index, count } => {
                write!(f, "sample {index} is out of range, the set holds {count}")
            }
        }
    }
}

impl std::error::Error for Error {}

fn write_u32(out: &mut [u8], offset: usize, value: u32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn read_u32(input: &[u8], offset: usize) -> u32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&input[offset..offset + 4]);
    u32::from_le_bytes(bytes)
}

fn encode_header(layout: &Layout, count: u32) -> [u8; HEADER_SIZE] {
    let mut header = [0u8; HEADER_SIZE];
    header[..8].copy_from_slice(&MAGIC);
    write_u32(&mut header, 8, VERSION);
    write_u32(&mut header, 12, layout.scale);
    write_u32(&mut header, 16, layout.lr_width);
    write_u32(&mut header, 20, layout.lr_height);
    write_u32(&mut header, 24, layout.lr_planes.bits());
    write_u32(&mut header, 28, layout.hr_planes.bits());
    write_u32(&mut header, 32, count);
    // 36..64 reserved, left zero.
    header
}

fn decode_header(header: &[u8; HEADER_SIZE]) -> Result<(Layout, u32), Error> {
    if header[..8] != MAGIC {
        return Err(Error::BadMagic);
    }
    let version = read_u32(header, 8);
    if version != VERSION {
        return Err(Error::Version(version));
    }
    let layout = Layout {
        scale: read_u32(header, 12),
        lr_width: read_u32(header, 16),
        lr_height: read_u32(header, 20),
        lr_planes: PlaneSet::from_bits(read_u32(header, 24))?,
        hr_planes: PlaneSet::from_bits(read_u32(header, 28))?,
    };
    if layout.scale == 0 || layout.lr_width == 0 || layout.lr_height == 0 {
        return Err(Error::EmptyLayout);
    }
    Ok((layout, read_u32(header, 32)))
}

/// Streams samples out to a `.omd` file.
///
/// The sample count is not known when the header goes down, so it is written
/// as zero and patched by [`finish`](Self::finish). A writer dropped without
/// finishing leaves a file that reads as empty rather than one that claims
/// samples it does not have.
pub struct Writer {
    file: BufWriter<File>,
    layout: Layout,
    count: u32,
}

impl Writer {
    /// Create a dataset at `path`, truncating anything already there.
    pub fn create(path: impl AsRef<Path>, layout: Layout) -> Result<Self, Error> {
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(&encode_header(&layout, 0))?;
        Ok(Self {
            file,
            layout,
            count: 0,
        })
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Append one sample.
    pub fn write(&mut self, sample: &Sample) -> Result<(), Error> {
        let expected_lr = self.layout.lr_len();
        if sample.lr.len() != expected_lr {
            return Err(Error::SampleSize {
                expected: expected_lr,
                actual: sample.lr.len(),
            });
        }
        let expected_hr = self.layout.hr_len();
        if sample.hr.len() != expected_hr {
            return Err(Error::SampleSize {
                expected: expected_hr,
                actual: sample.hr.len(),
            });
        }
        self.file.write_all(bytemuck::cast_slice(&sample.lr))?;
        self.file.write_all(bytemuck::cast_slice(&sample.hr))?;
        self.count += 1;
        Ok(())
    }

    /// Flush and patch the sample count into the header.
    pub fn finish(self) -> Result<u32, Error> {
        let count = self.count;
        let mut file = self.file.into_inner().map_err(|e| e.into_error())?;
        file.seek(SeekFrom::Start(32))?;
        file.write_all(&count.to_le_bytes())?;
        file.flush()?;
        Ok(count)
    }
}

/// Random-access reader over a `.omd` file.
///
/// Records are fixed size, so a sample is one seek and one read. The trainer
/// shuffles indices rather than the file, and nothing is held in memory beyond
/// the samples actually asked for.
pub struct Reader {
    file: BufReader<File>,
    layout: Layout,
    count: usize,
}

impl Reader {
    /// Open a dataset, validating the header against the file length.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = File::open(path)?;
        let byte_len = file.metadata()?.len();
        let mut file = BufReader::new(file);

        let mut header = [0u8; HEADER_SIZE];
        file.read_exact(&mut header)?;
        let (layout, count) = decode_header(&header)?;

        let expected = HEADER_SIZE as u64 + layout.record_bytes() * count as u64;
        if byte_len < expected {
            return Err(Error::Truncated {
                expected,
                actual: byte_len,
            });
        }

        Ok(Self {
            file,
            layout,
            count: count as usize,
        })
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Number of samples in the set.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Read one sample by index.
    pub fn sample(&mut self, index: usize) -> Result<Sample, Error> {
        if index >= self.count {
            return Err(Error::OutOfRange {
                index,
                count: self.count,
            });
        }
        let offset = HEADER_SIZE as u64 + self.layout.record_bytes() * index as u64;
        self.file.seek(SeekFrom::Start(offset))?;

        let mut lr = vec![f16::ZERO; self.layout.lr_len()];
        self.file.read_exact(bytemuck::cast_slice_mut(&mut lr))?;
        let mut hr = vec![f16::ZERO; self.layout.hr_len()];
        self.file.read_exact(bytemuck::cast_slice_mut(&mut hr))?;

        Ok(Sample { lr, hr })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> Layout {
        Layout {
            scale: 2,
            lr_width: 4,
            lr_height: 3,
            lr_planes: PlaneSet::new()
                .with(Plane::Color)
                .with(Plane::Depth)
                .with(Plane::Normal),
            hr_planes: PlaneSet::new().with(Plane::Color),
        }
    }

    #[test]
    fn plane_offsets_follow_storage_order() {
        let set = layout().lr_planes;
        assert_eq!(set.channels(), 7);
        assert_eq!(set.channel_offset(Plane::Color), Some(0));
        assert_eq!(set.channel_offset(Plane::Depth), Some(3));
        assert_eq!(set.channel_offset(Plane::Normal), Some(4));
        assert_eq!(set.channel_offset(Plane::Roughness), None);
    }

    #[test]
    fn plane_set_rejects_unknown_bits() {
        assert!(PlaneSet::from_bits(1 << 30).is_err());
        assert!(PlaneSet::from_bits(Plane::Color.bit()).is_ok());
    }

    #[test]
    fn layout_sizes_add_up() {
        let l = layout();
        assert_eq!(l.hr_width(), 8);
        assert_eq!(l.lr_len(), 7 * 12);
        assert_eq!(l.hr_len(), 3 * 48);
        assert_eq!(l.record_len(), l.lr_len() + l.hr_len());
    }

    #[test]
    fn roundtrip_through_a_file() {
        let l = layout();
        let dir = std::env::temp_dir().join("ommatidia-dataset-roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("set.omd");

        let make = |seed: f32| Sample {
            lr: (0..l.lr_len())
                .map(|i| f16::from_f32(seed + i as f32 * 0.01))
                .collect(),
            hr: (0..l.hr_len())
                .map(|i| f16::from_f32(seed - i as f32 * 0.01))
                .collect(),
        };

        let mut writer = Writer::create(&path, l).unwrap();
        writer.write(&make(0.0)).unwrap();
        writer.write(&make(1.0)).unwrap();
        assert_eq!(writer.finish().unwrap(), 2);

        let mut reader = Reader::open(&path).unwrap();
        assert_eq!(*reader.layout(), l);
        assert_eq!(reader.len(), 2);
        // Out of order, to exercise the seek rather than sequential reads.
        assert_eq!(reader.sample(1).unwrap().lr, make(1.0).lr);
        assert_eq!(reader.sample(0).unwrap().hr, make(0.0).hr);
        assert!(reader.sample(2).is_err());

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn channel_views_land_on_the_right_plane() {
        let l = layout();
        let texels = l.lr_texels();
        let mut lr = vec![f16::ZERO; l.lr_len()];
        // Mark every channel with its own index so a misaligned view shows up.
        for c in 0..l.lr_planes.channels() {
            for t in 0..texels {
                lr[c * texels + t] = f16::from_f32(c as f32);
            }
        }
        let sample = Sample {
            lr,
            hr: vec![f16::ZERO; l.hr_len()],
        };

        let depth = sample.lr_channel(&l, Plane::Depth, 0).unwrap();
        assert!(depth.iter().all(|&v| v == f16::from_f32(3.0)));
        let normal_z = sample.lr_channel(&l, Plane::Normal, 2).unwrap();
        assert!(normal_z.iter().all(|&v| v == f16::from_f32(6.0)));
        assert!(sample.lr_channel(&l, Plane::Motion, 0).is_none());
    }

    #[test]
    fn bad_magic_is_rejected() {
        let dir = std::env::temp_dir().join("ommatidia-dataset-magic");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("junk.omd");
        std::fs::write(&path, [0u8; HEADER_SIZE]).unwrap();
        assert!(matches!(Reader::open(&path), Err(Error::BadMagic)));
        std::fs::remove_file(&path).unwrap();
    }
}
