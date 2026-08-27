//! External glTF objects and interiors for expanding the training corpus.
//!
//! Procedural rooms still supply lighting, canopy, and camera variety. Catalog
//! entries add authored materials and geometry that a sphere-and-box generator
//! cannot invent. Each entry names a **family** and a **split** so the same
//! object or interior cannot appear in both training and the untouched audit.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use blade_graphics as gpu;
use ommatidia::rng::Rng;
use serde::{Deserialize, Serialize};

/// Where an asset was published. License terms differ; see `docs/catalog.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Source {
    Abo,
    Hssd,
    Dtc,
    /// Checked-in test mesh, not a published corpus.
    Fixture,
}

/// How the generator places the asset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    /// Sit on the procedural ground among the usual spheres, boxes, and lights.
    Object,
    /// Replace the procedural room. The camera samples the interior AABB.
    Interior,
}

/// Which side of the family hold-out this asset belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Split {
    Train,
    Holdout,
}

impl Split {
    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "train" => Ok(Self::Train),
            "holdout" => Ok(Self::Holdout),
            other => Err(format!(
                "--catalog-split wants train or holdout, got {other:?}"
            )),
        }
    }
}

/// One downloadable glTF, identified so a later audit can refuse overlap.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub id: String,
    pub source: Source,
    pub license: String,
    pub kind: Kind,
    pub family: String,
    pub split: Split,
    /// Path relative to the catalog file, or absolute.
    pub path: PathBuf,
    /// Optional Habitat scene-instance file. When present, its object GLBs are
    /// loaded and placed along with the architectural stage.
    #[serde(default)]
    pub scene: Option<PathBuf>,
    /// Optional root containing renderer-compatible Habitat object GLBs.
    /// HSSD's uncompressed model repository is usable by Blade; the compact
    /// `hssd-hab` objects require KTX2/BasisU texture support.
    #[serde(default)]
    pub object_root: Option<PathBuf>,
}

impl Entry {
    pub fn resolved_path(&self, catalog_dir: &Path) -> PathBuf {
        if self.path.is_absolute() {
            self.path.clone()
        } else {
            catalog_dir.join(&self.path)
        }
    }

    pub fn resolved_scene(&self, catalog_dir: &Path) -> Option<PathBuf> {
        self.scene.as_ref().map(|path| {
            if path.is_absolute() {
                path.clone()
            } else {
                catalog_dir.join(path)
            }
        })
    }

    pub fn resolved_object_root(&self, catalog_dir: &Path) -> Option<PathBuf> {
        self.object_root.as_ref().map(|path| {
            if path.is_absolute() {
                path.clone()
            } else {
                catalog_dir.join(path)
            }
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CatalogFile {
    pub entries: Vec<Entry>,
}

/// Loaded catalog plus the directory used to resolve relative paths.
#[derive(Clone, Debug)]
pub struct Catalog {
    pub dir: PathBuf,
    pub entries: Vec<Entry>,
}

impl Catalog {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|error| format!("cannot read catalog {}: {error}", path.display()))?;
        let file: CatalogFile = serde_json::from_str(&text)
            .map_err(|error| format!("catalog {}: {error}", path.display()))?;
        let dir = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        if file.entries.is_empty() {
            return Err(format!("{} contains no entries", path.display()));
        }
        let mut seen = HashMap::new();
        let mut family_splits = HashMap::new();
        for entry in &file.entries {
            if entry.id.trim().is_empty() {
                return Err("catalog entry is missing an id".into());
            }
            if entry.family.trim().is_empty() {
                return Err(format!("catalog entry {} is missing a family", entry.id));
            }
            if let Some(previous) = seen.insert(entry.id.clone(), entry.split) {
                if previous != entry.split {
                    return Err(format!(
                        "catalog id {} is listed in both {:?} and {:?}",
                        entry.id, previous, entry.split
                    ));
                }
                return Err(format!("catalog id {} is listed twice", entry.id));
            }
            if let Some(previous) = family_splits.insert(entry.family.clone(), entry.split)
                && previous != entry.split
            {
                return Err(format!(
                    "catalog family {} is listed in both {:?} and {:?}",
                    entry.family, previous, entry.split
                ));
            }
        }
        Ok(Self {
            dir,
            entries: file.entries,
        })
    }

    pub fn filtered(
        &self,
        split: Option<Split>,
        kind: Option<Kind>,
    ) -> Result<Vec<&Entry>, String> {
        let entries: Vec<&Entry> = self
            .entries
            .iter()
            .filter(|entry| split.is_none_or(|wanted| entry.split == wanted))
            .filter(|entry| kind.is_none_or(|wanted| entry.kind == wanted))
            .collect();
        if entries.is_empty() {
            return Err(match (split, kind) {
                (Some(split), Some(kind)) => {
                    format!("catalog has no {kind:?} entries in the {split:?} split")
                }
                (Some(split), None) => format!("catalog has no entries in the {split:?} split"),
                (None, Some(kind)) => format!("catalog has no {kind:?} entries"),
                (None, None) => "catalog has no entries".into(),
            });
        }
        Ok(entries)
    }
}

/// Shuffle-and-cycle through a pool so consecutive scenes do not reuse the
/// same object until every other one has been used.
pub struct Pool {
    entries: Vec<Entry>,
    next: usize,
}

impl Pool {
    pub fn new(entries: Vec<Entry>, rng: &mut Rng) -> Self {
        let mut order = entries;
        shuffle(&mut order, rng);
        Self {
            entries: order,
            next: 0,
        }
    }

    pub fn take(&mut self, count: usize, rng: &mut Rng) -> Vec<Entry> {
        let count = count.min(self.entries.len()).max(1);
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            if self.next >= self.entries.len() {
                shuffle(&mut self.entries, rng);
                self.next = 0;
            }
            out.push(self.entries[self.next].clone());
            self.next += 1;
        }
        out
    }
}

fn shuffle<T>(items: &mut [T], rng: &mut Rng) {
    for i in (1..items.len()).rev() {
        let j = rng.below((i + 1) as u32) as usize;
        items.swap(i, j);
    }
}

/// Uniform scale, yaw about Y, then translation. Longest AABB axis becomes
/// `target_extent`; the box sits on y = 0 with its XZ centre at `position_xz`.
pub fn object_transform(
    min: [f32; 3],
    max: [f32; 3],
    target_extent: f32,
    position_xz: [f32; 2],
    yaw: f32,
) -> gpu::Transform {
    let size = [
        (max[0] - min[0]).abs(),
        (max[1] - min[1]).abs(),
        (max[2] - min[2]).abs(),
    ];
    let longest = size[0].max(size[1]).max(size[2]).max(1.0e-6);
    let scale = target_extent / longest;
    let center = [
        0.5 * (min[0] + max[0]),
        0.5 * (min[1] + max[1]),
        0.5 * (min[2] + max[2]),
    ];
    let (sin, cos) = yaw.sin_cos();
    // Translate so the scaled centre lands on (px, 0 + scaled_half_height, pz).
    // After S, min.y is scale*min.y; we add ty so that equals 0.
    let ty = -scale * min[1];
    let dx = position_xz[0] - scale * (cos * center[0] + sin * center[2]);
    let dz = position_xz[1] - scale * (-sin * center[0] + cos * center[2]);
    gpu::Transform {
        x: [cos * scale, 0.0, sin * scale, dx].into(),
        y: [0.0, scale, 0.0, ty].into(),
        z: [-sin * scale, 0.0, cos * scale, dz].into(),
    }
}

/// Scale an interior so its longest axis is `target_extent`, sitting on y = 0
/// and centred on the origin. Identity if the box is already a usable size.
pub fn interior_transform(min: [f32; 3], max: [f32; 3], target_extent: f32) -> gpu::Transform {
    object_transform(min, max, target_extent, [0.0, 0.0], 0.0)
}

/// AABB after an affine placement. Used so interior cameras sample the room
/// the renderer actually sees, not the unscaled source mesh.
pub fn transformed_aabb(
    min: [f32; 3],
    max: [f32; 3],
    transform: &gpu::Transform,
) -> ([f32; 3], [f32; 3]) {
    let mut out_min = [f32::INFINITY; 3];
    let mut out_max = [f32::NEG_INFINITY; 3];
    for x in [min[0], max[0]] {
        for y in [min[1], max[1]] {
            for z in [min[2], max[2]] {
                let point = transform_point(transform, [x, y, z]);
                for axis in 0..3 {
                    out_min[axis] = out_min[axis].min(point[axis]);
                    out_max[axis] = out_max[axis].max(point[axis]);
                }
            }
        }
    }
    (out_min, out_max)
}

fn transform_point(transform: &gpu::Transform, point: [f32; 3]) -> [f32; 3] {
    [
        transform.x.x * point[0]
            + transform.x.y * point[1]
            + transform.x.z * point[2]
            + transform.x.w,
        transform.y.x * point[0]
            + transform.y.y * point[1]
            + transform.y.z * point[2]
            + transform.y.w,
        transform.z.x * point[0]
            + transform.z.y * point[1]
            + transform.z.z * point[2]
            + transform.z.w,
    ]
}

/// What one generated scene borrowed from the catalog.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneRecord {
    pub index: usize,
    pub ids: Vec<String>,
    pub families: Vec<String>,
    pub sources: Vec<Source>,
    pub kind: Kind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sidecar {
    pub catalog: PathBuf,
    pub split: Option<Split>,
    pub scenes: Vec<SceneRecord>,
}

impl Sidecar {
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        let text = serde_json::to_string_pretty(self)
            .map_err(|error| format!("catalog sidecar: {error}"))?;
        fs::write(path, text).map_err(|error| format!("cannot write {}: {error}", path.display()))
    }
}

/// Convert a `.omd` path into the sidecar sitting beside it.
pub fn sidecar_path(dataset: &Path) -> PathBuf {
    let stem = dataset.file_stem().unwrap_or_default();
    dataset.with_file_name(format!("{}.catalog.json", stem.to_string_lossy()))
}

/// Loaded handles, keyed by catalog id. The generator looks these up per scene.
pub struct Loaded {
    pub catalog: Catalog,
    pub handles: HashMap<String, blade_asset::Handle<blade_render::Model>>,
    bounds: HashMap<String, ([f32; 3], [f32; 3])>,
    furnishings: HashMap<String, Vec<Furnishing>>,
}

#[derive(Clone)]
pub struct Furnishing {
    pub id: String,
    pub handle: blade_asset::Handle<blade_render::Model>,
    pub transform: gpu::Transform,
}

#[derive(Deserialize)]
struct HabitatScene {
    object_instances: Vec<HabitatObject>,
}

#[derive(Deserialize)]
struct HabitatObject {
    template_name: String,
    translation: [f32; 3],
    /// Habitat serializes quaternions as `[w, x, y, z]`.
    rotation: [f32; 4],
    non_uniform_scale: [f32; 3],
}

impl Loaded {
    pub fn bake(catalog: Catalog, hub: &blade_render::AssetHub) -> Result<Self, String> {
        let mut handles = HashMap::new();
        let mut bounds = HashMap::new();
        let mut furnishings = HashMap::new();
        let mut tasks = Vec::new();
        for entry in &catalog.entries {
            let path = entry.resolved_path(&catalog.dir);
            if !path.exists() {
                return Err(format!(
                    "catalog {} points at missing {}",
                    entry.id,
                    path.display()
                ));
            }
            bounds.insert(entry.id.clone(), gltf_aabb(&path)?);
            let (handle, task) = hub.models.load(
                &path,
                blade_render::model::Meta {
                    generate_tangents: true,
                    ..Default::default()
                },
            );
            tasks.push(task.clone());
            handles.insert(entry.id.clone(), handle);
            if let Some(scene_path) = entry.resolved_scene(&catalog.dir) {
                let text = fs::read_to_string(&scene_path).map_err(|error| {
                    format!(
                        "cannot read Habitat scene {}: {error}",
                        scene_path.display()
                    )
                })?;
                let scene: HabitatScene = serde_json::from_str(&text)
                    .map_err(|error| format!("Habitat scene {}: {error}", scene_path.display()))?;
                let scene_root = scene_path
                    .parent()
                    .and_then(Path::parent)
                    .ok_or_else(|| format!("{} has no HSSD root", scene_path.display()))?;
                let object_root = entry
                    .resolved_object_root(&catalog.dir)
                    .unwrap_or_else(|| scene_root.join("objects"));
                let mut placed = Vec::with_capacity(scene.object_instances.len());
                let mut missing = 0usize;
                for object in scene.object_instances {
                    let bucket = object.template_name.chars().next().ok_or_else(|| {
                        format!("{} has an empty object id", scene_path.display())
                    })?;
                    let object_path = object_root
                        .join(bucket.to_string())
                        .join(format!("{}.glb", object.template_name));
                    if !object_path.exists() {
                        missing += 1;
                        continue;
                    }
                    let key = format!("hssd-object/{}", object.template_name);
                    let handle = if let Some(handle) = handles.get(&key) {
                        *handle
                    } else {
                        let (handle, task) = hub.models.load(
                            &object_path,
                            blade_render::model::Meta {
                                generate_tangents: true,
                                ..Default::default()
                            },
                        );
                        tasks.push(task.clone());
                        handles.insert(key.clone(), handle);
                        handle
                    };
                    placed.push(Furnishing {
                        id: key,
                        handle,
                        transform: habitat_transform(
                            object.translation,
                            object.rotation,
                            object.non_uniform_scale,
                        ),
                    });
                }
                if missing != 0 {
                    eprintln!(
                        "warning: Habitat scene {} omitted {missing} instances whose public model GLBs are unavailable",
                        scene_path.display()
                    );
                }
                if placed.is_empty() {
                    return Err(format!(
                        "Habitat scene {} has no loadable furnishings under {}",
                        scene_path.display(),
                        object_root.display()
                    ));
                }
                furnishings.insert(entry.id.clone(), placed);
            }
        }
        for task in tasks {
            task.join();
        }
        Ok(Self {
            catalog,
            handles,
            bounds,
            furnishings,
        })
    }

    pub fn handle(&self, id: &str) -> blade_asset::Handle<blade_render::Model> {
        self.handles[id]
    }

    pub fn aabb(&self, id: &str) -> ([f32; 3], [f32; 3]) {
        self.bounds[id]
    }

    pub fn furnishings(&self, id: &str) -> &[Furnishing] {
        self.furnishings.get(id).map_or(&[], Vec::as_slice)
    }
}

fn habitat_transform(translation: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> gpu::Transform {
    let [w, x, y, z] = rotation;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    gpu::Transform {
        x: [
            (1.0 - 2.0 * (yy + zz)) * scale[0],
            (2.0 * (xy - wz)) * scale[1],
            (2.0 * (xz + wy)) * scale[2],
            translation[0],
        ]
        .into(),
        y: [
            (2.0 * (xy + wz)) * scale[0],
            (1.0 - 2.0 * (xx + zz)) * scale[1],
            (2.0 * (yz - wx)) * scale[2],
            translation[1],
        ]
        .into(),
        z: [
            (2.0 * (xz - wy)) * scale[0],
            (2.0 * (yz + wx)) * scale[1],
            (1.0 - 2.0 * (xx + yy)) * scale[2],
            translation[2],
        ]
        .into(),
    }
}

/// Apply `child` in an authored scene, then the scene-wide `parent` placement.
pub fn compose_transform(parent: &gpu::Transform, child: &gpu::Transform) -> gpu::Transform {
    let p = [parent.x, parent.y, parent.z];
    let c = [child.x, child.y, child.z];
    let at = |row: usize, column: usize| match column {
        0 => p[row].x * c[0].x + p[row].y * c[1].x + p[row].z * c[2].x,
        1 => p[row].x * c[0].y + p[row].y * c[1].y + p[row].z * c[2].y,
        2 => p[row].x * c[0].z + p[row].y * c[1].z + p[row].z * c[2].z,
        3 => p[row].x * c[0].w + p[row].y * c[1].w + p[row].z * c[2].w + p[row].w,
        _ => unreachable!(),
    };
    gpu::Transform {
        x: [at(0, 0), at(0, 1), at(0, 2), at(0, 3)].into(),
        y: [at(1, 0), at(1, 1), at(1, 2), at(1, 3)].into(),
        z: [at(2, 0), at(2, 1), at(2, 2), at(2, 3)].into(),
    }
}

type Matrix = [[f32; 4]; 4];

fn identity() -> Matrix {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Column-major matrix product, matching glTF's transform convention.
fn matrix_product(left: Matrix, right: Matrix) -> Matrix {
    let mut out = [[0.0; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            out[column][row] = (0..4)
                .map(|inner| left[inner][row] * right[column][inner])
                .sum();
        }
    }
    out
}

fn matrix_point(matrix: Matrix, point: [f32; 3]) -> [f32; 3] {
    [
        matrix[0][0] * point[0] + matrix[1][0] * point[1] + matrix[2][0] * point[2] + matrix[3][0],
        matrix[0][1] * point[0] + matrix[1][1] * point[1] + matrix[2][1] * point[2] + matrix[3][1],
        matrix[0][2] * point[0] + matrix[1][2] * point[1] + matrix[2][2] * point[2] + matrix[3][2],
    ]
}

fn extend_node_aabb(node: gltf::Node<'_>, parent: Matrix, min: &mut [f32; 3], max: &mut [f32; 3]) {
    let world = matrix_product(parent, node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            let bounds = primitive.bounding_box();
            for x in [bounds.min[0], bounds.max[0]] {
                for y in [bounds.min[1], bounds.max[1]] {
                    for z in [bounds.min[2], bounds.max[2]] {
                        let point = matrix_point(world, [x, y, z]);
                        for axis in 0..3 {
                            min[axis] = min[axis].min(point[axis]);
                            max[axis] = max[axis].max(point[axis]);
                        }
                    }
                }
            }
        }
    }
    for child in node.children() {
        extend_node_aabb(child, world, min, max);
    }
}

fn gltf_aabb(path: &Path) -> Result<([f32; 3], [f32; 3]), String> {
    let gltf = gltf::Gltf::open(path)
        .map_err(|error| format!("cannot inspect glTF {}: {error}", path.display()))?;
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for scene in gltf.scenes() {
        for node in scene.nodes() {
            extend_node_aabb(node, identity(), &mut min, &mut max);
        }
    }
    if min.iter().chain(&max).any(|value| !value.is_finite()) {
        return Err(format!(
            "glTF {} contains no bounded geometry",
            path.display()
        ));
    }
    Ok((min, max))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, split: Split, kind: Kind) -> Entry {
        Entry {
            id: id.into(),
            source: Source::Fixture,
            license: "MIT".into(),
            kind,
            family: id.into(),
            split,
            path: PathBuf::from(format!("{id}.glb")),
            scene: None,
            object_root: None,
        }
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let dir = std::env::temp_dir().join("ommatidia-catalog-dup");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("catalog.json");
        fs::write(
            &path,
            r#"{"entries":[
                {"id":"a","source":"fixture","license":"MIT","kind":"object","family":"a","split":"train","path":"a.glb"},
                {"id":"a","source":"fixture","license":"MIT","kind":"object","family":"a","split":"holdout","path":"a.glb"}
            ]}"#,
        )
        .unwrap();
        let error = Catalog::load(&path).unwrap_err();
        assert!(error.contains("both"), "{error}");
    }

    #[test]
    fn one_family_cannot_cross_splits_under_different_ids() {
        let dir = std::env::temp_dir().join("ommatidia-catalog-family-split");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("catalog.json");
        fs::write(
            &path,
            r#"{"entries":[
                {"id":"chair-a","source":"fixture","license":"MIT","kind":"object","family":"chair","split":"train","path":"a.glb"},
                {"id":"chair-b","source":"fixture","license":"MIT","kind":"object","family":"chair","split":"holdout","path":"b.glb"}
            ]}"#,
        )
        .unwrap();
        let error = Catalog::load(&path).unwrap_err();
        assert!(error.contains("family chair"), "{error}");
        assert!(error.contains("both"), "{error}");
    }

    #[test]
    fn split_filter_never_leaks_a_family() {
        let catalog = Catalog {
            dir: PathBuf::from("."),
            entries: vec![
                entry("chair-a", Split::Train, Kind::Object),
                entry("chair-b", Split::Holdout, Kind::Object),
                entry("room-a", Split::Train, Kind::Interior),
            ],
        };
        let train: Vec<_> = catalog
            .filtered(Some(Split::Train), None)
            .unwrap()
            .into_iter()
            .map(|entry| entry.id.as_str())
            .collect();
        let holdout: Vec<_> = catalog
            .filtered(Some(Split::Holdout), None)
            .unwrap()
            .into_iter()
            .map(|entry| entry.id.as_str())
            .collect();
        assert_eq!(train, ["chair-a", "room-a"]);
        assert_eq!(holdout, ["chair-b"]);
        for id in &train {
            assert!(!holdout.contains(id));
        }
    }

    #[test]
    fn object_transform_sits_on_the_ground_at_the_requested_size() {
        let min = [10.0, 4.0, -2.0];
        let max = [14.0, 8.0, 2.0];
        let transform = object_transform(min, max, 2.0, [3.0, 5.0], 0.0);
        let placed_min = transform_point(&transform, min);
        let placed_max = transform_point(&transform, max);
        assert!((placed_min[1]).abs() < 1.0e-5, "{placed_min:?}");
        assert!((placed_max[1] - 2.0).abs() < 1.0e-5, "{placed_max:?}");
        let cx = 0.5 * (placed_min[0] + placed_max[0]);
        let cz = 0.5 * (placed_min[2] + placed_max[2]);
        assert!((cx - 3.0).abs() < 1.0e-5, "{cx}");
        assert!((cz - 5.0).abs() < 1.0e-5, "{cz}");
        let dx = placed_max[0] - placed_min[0];
        let dy = placed_max[1] - placed_min[1];
        let dz = placed_max[2] - placed_min[2];
        assert!((dx.max(dy).max(dz) - 2.0).abs() < 1.0e-5);
    }

    #[test]
    fn habitat_object_placement_is_composed_inside_the_stage() {
        let stage = object_transform([0.0; 3], [2.0; 3], 4.0, [3.0, 5.0], 0.0);
        let object = habitat_transform([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0], [2.0, 3.0, 4.0]);
        let composed = compose_transform(&stage, &object);
        let point = [0.25, 0.5, 0.75];
        let expected = transform_point(&stage, transform_point(&object, point));
        let actual = transform_point(&composed, point);
        for axis in 0..3 {
            assert!((actual[axis] - expected[axis]).abs() < 1e-5);
        }
    }

    #[test]
    fn pool_cycles_without_immediate_repeats_when_the_set_is_large() {
        let catalog = Catalog {
            dir: PathBuf::from("."),
            entries: (0..8)
                .map(|i| entry(&format!("o{i}"), Split::Train, Kind::Object))
                .collect(),
        };
        let entries = catalog
            .filtered(Some(Split::Train), Some(Kind::Object))
            .unwrap()
            .into_iter()
            .cloned()
            .collect();
        let mut rng = Rng::new(7);
        let mut pool = Pool::new(entries, &mut rng);
        let first: Vec<_> = pool
            .take(8, &mut rng)
            .into_iter()
            .map(|e| e.id.clone())
            .collect();
        let mut unique = first.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 8);
    }

    #[test]
    fn sidecar_path_sits_beside_the_dataset() {
        assert_eq!(
            sidecar_path(Path::new("data/rich-train.omd")),
            PathBuf::from("data/rich-train.catalog.json")
        );
    }

    #[test]
    fn fixture_catalog_resolves_checked_in_meshes() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/catalog.json");
        let catalog = Catalog::load(&path).unwrap();
        assert_eq!(catalog.entries.len(), 2);
        for entry in &catalog.entries {
            let resolved = entry.resolved_path(&catalog.dir);
            assert!(resolved.exists(), "{}", resolved.display());
        }
        let train = catalog.filtered(Some(Split::Train), None).unwrap();
        let holdout = catalog.filtered(Some(Split::Holdout), None).unwrap();
        assert_eq!(train[0].id, "fixture/plane");
        assert_eq!(holdout[0].id, "fixture/sphere");
        assert!(
            train
                .iter()
                .all(|entry| !holdout.iter().any(|other| other.id == entry.id))
        );
        let plane = gltf_aabb(&train[0].resolved_path(&catalog.dir)).unwrap();
        assert!(
            plane
                .0
                .iter()
                .chain(&plane.1)
                .all(|value| value.is_finite())
        );
    }
}
