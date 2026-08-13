//! The reconstruction network, as a meganeura graph.
//!
//! A timestep-conditioned U-Net that runs entirely at input resolution and
//! emits `3 * scale^2` channels, which the runtime scatters into the high
//! resolution target. See `docs/design.md` for why the network never touches
//! output resolution.
//!
//! The same backbone serves both objectives. Under [`Objective::Diffusion`] it
//! takes a noised residual alongside the conditioning and predicts the noise;
//! under [`Objective::Direct`] it takes only the conditioning and predicts the
//! residual itself. Only the input channel count and the loss target differ.

use meganeura::{Graph, NodeId};
use serde::{Deserialize, Serialize};

use crate::dataset::{Plane, PlaneSet};

/// What the network is trained to predict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Objective {
    /// e-prediction. Input carries a noised residual and a noise level, output
    /// is the noise. Sampling is iterative.
    Diffusion,
    /// Direct regression of the residual. One forward pass, no noise input.
    ///
    /// The fast path, and the baseline any distilled sampler has to beat.
    Direct,
}

/// Spatial backbone used between the sub-pixel input and output contracts.
///
/// The hybrid keeps the U-shaped multi-scale encoder/decoder, but replaces
/// its two lowest-resolution convolutional blocks with shifted local
/// self-attention plus convolutional feed-forward blocks.  "Transformer" and
/// "U-Net" are not mutually exclusive choices for dense restoration: global
/// ViT attention is quadratic in frame size, while the hierarchy and skip
/// paths remain valuable for exact reconstruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backbone {
    /// The measured v0.1 convolutional U-Net.
    #[default]
    Conv,
    /// Linear-in-pixel local attention at the narrowest U-Net resolution.
    HybridWindowAttention {
        /// Square attention window in bottleneck pixels.
        window: u32,
        /// Channels per attention head. Must be a power of two.
        head_dim: u32,
    },
}

/// How a parameter should be filled before training starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitKind {
    /// Kaiming normal, scaled by `sqrt(2 / fan_in)`, for weights behind SiLU.
    Kaiming {
        fan_in: usize,
    },
    Zeros,
    Ones,
}

/// A parameter the graph declared, with enough information to initialise it.
#[derive(Clone, Debug)]
pub struct ParamInit {
    pub name: String,
    pub len: usize,
    pub kind: InitKind,
}

/// Shape of the network.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Output resolution is input resolution times this.
    pub scale: u32,
    /// Square input tile the network is compiled for.
    ///
    /// Compilation bakes in the extent, so training tiles and the runtime
    /// window have to agree. Must be divisible by `2^(levels - 1)`.
    pub tile: u32,
    pub batch: u32,
    /// Which dataset planes feed the conditioning, in storage order.
    pub cond_planes: PlaneSet,
    /// Channel width at the first level.
    pub base_channels: u32,
    /// Width multiplier per level; its length is the number of levels.
    pub level_multipliers: Vec<u32>,
    /// Residual blocks per level.
    pub blocks_per_level: usize,
    pub num_groups: u32,
    pub gn_eps: f32,
    /// Reconstruction backbone. Defaults to [`Backbone::Conv`] when absent so
    /// v0.1 checkpoint sidecars remain loadable.
    #[serde(default)]
    pub backbone: Backbone,
    /// Width of the sinusoidal timestep embedding the host computes.
    pub time_input_dim: u32,
    /// Width the timestep MLP projects to.
    pub time_embed_dim: u32,
    /// Factor the sub-pixel residual is multiplied by to reach unit variance.
    ///
    /// Measured from the training set with [`crate::batch::estimate_gain`] and
    /// carried in the checkpoint, because inference has to divide by exactly
    /// the value training multiplied by. See [`crate::batch::write_residual`]
    /// for why a diffusion model cannot be trained without it.
    pub residual_gain: f32,
    pub objective: Objective,
}

impl Default for ModelConfig {
    /// The measured deployment baseline: a one-pass, 649k-parameter U-Net
    /// that retained the larger model's reconstruction quality.
    fn default() -> Self {
        Self {
            scale: 2,
            tile: 64,
            batch: 8,
            cond_planes: PlaneSet::new()
                .with(Plane::Color)
                .with(Plane::Depth)
                .with(Plane::Normal)
                .with(Plane::DiffuseAlbedo)
                .with(Plane::SpecularF0)
                .with(Plane::Roughness),
            base_channels: 24,
            level_multipliers: vec![1, 2, 4],
            blocks_per_level: 1,
            num_groups: 8,
            gn_eps: 1e-5,
            backbone: Backbone::Conv,
            time_input_dim: 64,
            time_embed_dim: 256,
            // Overwritten from the data; 1.0 leaves the residual as it is.
            residual_gain: 1.0,
            objective: Objective::Direct,
        }
    }
}

impl ModelConfig {
    /// Number of levels in the U-Net.
    pub fn levels(&self) -> usize {
        self.level_multipliers.len()
    }

    /// Conditioning channels, from the plane set.
    pub fn cond_channels(&self) -> u32 {
        self.cond_planes.channels() as u32
    }

    /// Output channels: RGB for every sub-pixel of the scale factor.
    pub fn target_channels(&self) -> u32 {
        3 * self.scale * self.scale
    }

    /// Channels the first convolution consumes.
    pub fn in_channels(&self) -> u32 {
        match self.objective {
            Objective::Diffusion => self.target_channels() + self.cond_channels(),
            Objective::Direct => self.cond_channels(),
        }
    }

    /// Channel width at `level`.
    pub fn channels_at(&self, level: usize) -> u32 {
        self.base_channels * self.level_multipliers[level]
    }

    /// Estimate the convolution and attention arithmetic for one frame, in
    /// GFLOP.
    ///
    /// Counts convolution multiply-accumulates and the two attention matrix
    /// products. Normalisation, softmax, activation, and window movement are
    /// bandwidth or scalar work and are not counted even though they matter to
    /// measured frame time. This is therefore a floor on cost, useful for
    /// ruling a configuration out rather than predicting its runtime.
    ///
    /// `output_pixels` lets a configuration compiled for one tile be costed at
    /// the extent it would actually run at.
    pub fn flops(&self, output_pixels: usize) -> f64 {
        // The network runs at input resolution; the tile it was compiled for
        // is irrelevant to the cost per output pixel.
        let input_pixels = output_pixels as f64 / (self.scale * self.scale) as f64;
        let mut total = 0.0;
        // Two multiply-accumulates per tap, k*k taps.
        let conv = |pixels: f64, cin: u32, cout: u32, k: u32| {
            2.0 * pixels * cin as f64 * cout as f64 * (k * k) as f64
        };

        let mut pixels = input_pixels;
        let mut channels = self.base_channels;
        total += conv(pixels, self.in_channels(), self.base_channels, 3); // stem

        // Encoder, then the downsample that follows every level but the last.
        let levels = self.levels();
        let mut skips = Vec::new();
        for level in 0..levels {
            let width = self.channels_at(level);
            for _ in 0..self.blocks_per_level {
                total += conv(pixels, channels, width, 3) + conv(pixels, width, width, 3);
                if channels != width {
                    total += conv(pixels, channels, width, 1); // residual projection
                }
                channels = width;
            }
            if level + 1 < levels {
                skips.push((pixels, channels));
                pixels /= 4.0; // half in each axis
                total += conv(pixels, width, width, 3); // strided downsample
            }
        }

        // Middle. The hybrid is deliberately close to the convolutional
        // baseline's arithmetic: four 1x1 attention projections, a 3x3
        // positional mixer, and a 2x channel FFN replace each two-convolution
        // residual block. Window attention itself is linear in pixel count.
        match self.backbone {
            Backbone::Conv => {
                for _ in 0..2 {
                    total += conv(pixels, channels, channels, 3) * 2.0;
                }
            }
            Backbone::HybridWindowAttention { window, .. } => {
                for _ in 0..2 {
                    total += conv(pixels, channels, channels, 1) * 4.0;
                    total += conv(pixels, channels, channels, 3);
                    total += conv(pixels, channels, 2 * channels, 1);
                    total += conv(pixels, 2 * channels, channels, 1);
                    // QK^T and softmax-weighted V: two matrix products, each
                    // one multiply-add per channel per query/key pair.
                    total += 4.0 * pixels * (window * window * channels) as f64;
                }
            }
        }

        // Decoder: upsample, concatenate the skip, then narrow back down.
        for level in (0..levels.saturating_sub(1)).rev() {
            let (skip_pixels, skip_channels) = skips.pop().expect("a skip per level");
            pixels = skip_pixels;
            channels += skip_channels;
            let width = self.channels_at(level);
            for _ in 0..self.blocks_per_level {
                total += conv(pixels, channels, width, 3) + conv(pixels, width, width, 3);
                if channels != width {
                    total += conv(pixels, channels, width, 1);
                }
                channels = width;
            }
        }

        total += conv(pixels, channels, self.target_channels(), 3); // head
        total / 1e9
    }

    /// Elements in one conditioning tensor of a batch.
    pub fn cond_len(&self) -> usize {
        self.cond_len_for_extent([self.tile, self.tile])
    }

    /// Elements in one conditioning tensor at a runtime extent.
    pub fn cond_len_for_extent(&self, extent: [u32; 2]) -> usize {
        (self.batch * self.cond_channels() * extent[0] * extent[1]) as usize
    }

    /// Elements in one target or output tensor of a batch.
    pub fn target_len(&self) -> usize {
        self.target_len_for_extent([self.tile, self.tile])
    }

    /// Elements in one target or output tensor at a runtime extent.
    pub fn target_len_for_extent(&self, extent: [u32; 2]) -> usize {
        (self.batch * self.target_channels() * extent[0] * extent[1]) as usize
    }

    /// Elements in one batch of timestep embeddings.
    pub fn time_len(&self) -> usize {
        (self.batch * self.time_input_dim) as usize
    }

    /// Reject configurations the graph cannot express, with a reason.
    ///
    /// Called by [`build`]; exposed so a caller can check a configuration it
    /// assembled without paying for a compile.
    pub fn validate(&self) -> Result<(), String> {
        if self.scale < 2 {
            return Err(format!("scale {} must be at least 2", self.scale));
        }
        if self.levels() == 0 {
            return Err("the network needs at least one level".into());
        }
        for level in 0..self.levels() {
            let channels = self.channels_at(level);
            if !channels.is_multiple_of(self.num_groups) {
                return Err(format!(
                    "level {level} has {channels} channels, not divisible by \
                     num_groups {}",
                    self.num_groups
                ));
            }
        }
        self.validate_extent([self.tile, self.tile])?;
        if let Backbone::HybridWindowAttention { window, head_dim } = self.backbone {
            if window < 2 {
                return Err(format!("attention window {window} must be at least 2"));
            }
            if !head_dim.is_power_of_two() || head_dim < 2 {
                return Err(format!(
                    "attention head_dim {head_dim} must be a power of two at least 2"
                ));
            }
            let channels = self.channels_at(self.levels() - 1);
            if !channels.is_multiple_of(head_dim) {
                return Err(format!(
                    "bottleneck width {channels} is not divisible by attention head_dim {head_dim}"
                ));
            }
        }
        if !self.time_input_dim.is_multiple_of(2) {
            return Err(format!(
                "time_input_dim {} must be even for a sinusoidal embedding",
                self.time_input_dim
            ));
        }
        if self.cond_channels() == 0 {
            return Err("the conditioning plane set is empty".into());
        }
        if !self.residual_gain.is_finite() || self.residual_gain <= 0.0 {
            return Err(format!(
                "residual_gain {} must be finite and positive",
                self.residual_gain
            ));
        }
        Ok(())
    }

    /// Reject a runtime extent that the checkpoint's U-Net cannot express.
    pub fn validate_extent(&self, extent: [u32; 2]) -> Result<(), String> {
        if self.levels() == 0 {
            return Err("the network needs at least one level".into());
        }
        let shrink = 1u32 << (self.levels() - 1);
        for (axis, value) in [("width", extent[0]), ("height", extent[1])] {
            if !value.is_multiple_of(shrink) {
                return Err(format!(
                    "{axis} {value} is not divisible by {shrink}, which {} levels of downsampling need",
                    self.levels()
                ));
            }
            if value / shrink < 2 {
                return Err(format!(
                    "{axis} {value} collapses below 2 after {} levels",
                    self.levels()
                ));
            }
        }
        if let Backbone::HybridWindowAttention { window, head_dim } = self.backbone {
            if window == 0 || head_dim == 0 {
                return Err("attention window and head_dim must be non-zero".into());
            }
            let shrink = 1u32 << (self.levels() - 1);
            let width = extent[0] / shrink;
            let height = extent[1] / shrink;
            let channels = self.channels_at(self.levels() - 1);
            // The second block shifts by half a window and therefore has the
            // larger padded grid. Windows are represented as attention heads;
            // Meganeura packs the head counts into 16 bits.
            let shift = window / 2;
            let windows = (width + shift).div_ceil(window) * (height + shift).div_ceil(window);
            let heads = self.batch * windows * (channels / head_dim);
            if heads > u16::MAX as u32 {
                return Err(format!(
                    "shifted bottleneck needs {heads} attention heads, above the portable 65535 limit"
                ));
            }
        }
        Ok(())
    }
}

/// A built network: the graph, its output, and how to initialise it.
pub struct Model {
    pub graph: Graph,
    pub config: ModelConfig,
    /// Spatial extent baked into this graph, in input pixels.
    pub input_extent: [u32; 2],
    /// Predicted noise under [`Objective::Diffusion`], predicted residual
    /// under [`Objective::Direct`]. Shape `[batch, target_channels, tile,
    /// tile]`, flattened.
    pub output: NodeId,
    /// The loss, present only when the graph was built for training.
    pub loss: Option<NodeId>,
    pub params: Vec<ParamInit>,
}

impl Model {
    /// Fill every declared parameter in a session.
    ///
    /// Deterministic given `seed`, so a run can be replayed.
    pub fn initialize(&self, session: &mut meganeura::Session, seed: u64) {
        let mut rng = crate::rng::Rng::new(seed);
        for param in &self.params {
            let data: Vec<f32> = match param.kind {
                InitKind::Kaiming { fan_in } => {
                    let scale = (2.0 / fan_in.max(1) as f32).sqrt();
                    (0..param.len).map(|_| rng.normal() * scale).collect()
                }
                InitKind::Zeros => vec![0.0; param.len],
                InitKind::Ones => vec![1.0; param.len],
            };
            session.set_parameter(&param.name, &data);
        }
    }
}

/// Tracks the tensor shape as the builder walks the U-Net.
#[derive(Clone, Copy)]
struct Shape {
    channels: u32,
    h: u32,
    w: u32,
}

impl Shape {
    fn spatial(&self) -> u32 {
        self.h * self.w
    }
}

#[derive(Clone, Copy)]
struct WindowAttentionSpec {
    window: u32,
    head_dim: u32,
    shift: u32,
}

/// Collects parameter declarations as the graph is built.
struct Builder<'a> {
    g: &'a mut Graph,
    config: &'a ModelConfig,
    params: Vec<ParamInit>,
}

impl<'a> Builder<'a> {
    fn param(&mut self, name: &str, len: usize, kind: InitKind) -> NodeId {
        self.params.push(ParamInit {
            name: name.to_string(),
            len,
            kind,
        });
        self.g.parameter(name, &[len])
    }

    /// A convolution's weight, laid out as meganeura wants it: `[out, in, kh,
    /// kw]` flattened.
    fn conv_weight(&mut self, name: &str, out_c: u32, in_c: u32, k: u32) -> NodeId {
        let fan_in = (in_c * k * k) as usize;
        self.param(
            name,
            (out_c as usize) * fan_in,
            InitKind::Kaiming { fan_in },
        )
    }

    fn conv(&mut self, x: NodeId, name: &str, s: Shape, out_c: u32, k: u32, stride: u32) -> NodeId {
        let weight = self.conv_weight(name, out_c, s.channels, k);
        // "same" padding for odd kernels; 1x1 needs none.
        let padding = k / 2;
        self.g.conv2d(
            x,
            weight,
            self.config.batch,
            s.channels,
            s.h,
            s.w,
            out_c,
            k,
            k,
            stride,
            padding,
        )
    }

    fn group_norm(&mut self, x: NodeId, name: &str, s: Shape) -> NodeId {
        let weight = self.param(
            &format!("{name}.weight"),
            s.channels as usize,
            InitKind::Ones,
        );
        let bias = self.param(
            &format!("{name}.bias"),
            s.channels as usize,
            InitKind::Zeros,
        );
        self.g.group_norm(
            x,
            weight,
            bias,
            self.config.batch,
            s.channels,
            s.spatial(),
            self.config.num_groups,
            self.config.gn_eps,
        )
    }

    /// `[rows, in] @ [in, out] + bias`.
    fn linear(&mut self, x: NodeId, name: &str, in_dim: u32, out_dim: u32) -> NodeId {
        let weight = self.param(
            &format!("{name}.weight"),
            (in_dim * out_dim) as usize,
            InitKind::Kaiming {
                fan_in: in_dim as usize,
            },
        );
        let weight = self.g.reshape(weight, &[in_dim as usize, out_dim as usize]);
        let bias = self.param(&format!("{name}.bias"), out_dim as usize, InitKind::Zeros);
        let out = self.g.matmul(x, weight);
        self.g.bias_add(out, bias)
    }

    /// Broadcast a per-channel vector over the spatial plane of an NCHW
    /// tensor.
    ///
    /// `[B, C] -> [B*C, 1] @ [1, HW] -> [B, C, H, W]`. The flat result of that
    /// matmul is already in NCHW order, so no transpose is needed.
    fn broadcast_spatial(&mut self, per_channel: NodeId, channels: u32, spatial: u32) -> NodeId {
        let rows = (self.config.batch * channels) as usize;
        let column = self.g.reshape(per_channel, &[rows, 1]);
        let ones = self
            .g
            .constant(vec![1.0; spatial as usize], &[1, spatial as usize]);
        let plane = self.g.matmul(column, ones);
        self.g.reshape(plane, &[rows * spatial as usize])
    }

    /// GroupNorm, SiLU, conv, add the projected timestep, GroupNorm, SiLU,
    /// conv, plus a residual that is projected when the width changes.
    fn resblock(
        &mut self,
        x: NodeId,
        time: Option<NodeId>,
        name: &str,
        s: Shape,
        out_c: u32,
    ) -> NodeId {
        let h = self.group_norm(x, &format!("{name}.norm1"), s);
        let h = self.g.silu(h);
        let mut h = self.conv(h, &format!("{name}.conv1.weight"), s, out_c, 3, 1);

        if let Some(time) = time {
            let projected = self.linear(
                time,
                &format!("{name}.time_proj"),
                self.config.time_embed_dim,
                out_c,
            );
            let plane = self.broadcast_spatial(projected, out_c, s.spatial());
            h = self.g.add(h, plane);
        }

        let wide = Shape {
            channels: out_c,
            h: s.h,
            w: s.w,
        };
        let h = self.group_norm(h, &format!("{name}.norm2"), wide);
        let h = self.g.silu(h);
        let h = self.conv(h, &format!("{name}.conv2.weight"), wide, out_c, 3, 1);

        if s.channels == out_c {
            self.g.add(x, h)
        } else {
            let skip = self.conv(x, &format!("{name}.skip.weight"), s, out_c, 1, 1);
            self.g.add(skip, h)
        }
    }

    /// Pre-normalized local attention followed by a convolutional FFN.
    ///
    /// Packing windows as independent heads composes Meganeura's existing
    /// differentiable attention kernel; no image-specialized attention shader
    /// is needed. Alternating an unshifted and half-window-shifted block lets
    /// information cross fixed window boundaries.
    fn hybrid_window_block(
        &mut self,
        x: NodeId,
        time: Option<NodeId>,
        name: &str,
        s: Shape,
        attention: WindowAttentionSpec,
    ) -> NodeId {
        let WindowAttentionSpec {
            window,
            head_dim,
            shift,
        } = attention;
        debug_assert!(s.channels.is_multiple_of(head_dim));
        let normalized = self.group_norm(x, &format!("{name}.attn.norm"), s);
        let q = self.conv(
            normalized,
            &format!("{name}.attn.q.weight"),
            s,
            s.channels,
            1,
            1,
        );
        let k = self.conv(
            normalized,
            &format!("{name}.attn.k.weight"),
            s,
            s.channels,
            1,
            1,
        );
        let v = self.conv(
            normalized,
            &format!("{name}.attn.v.weight"),
            s,
            s.channels,
            1,
            1,
        );
        let pack = |g: &mut Graph, value| {
            g.window_partition_2d(
                value,
                self.config.batch,
                s.channels,
                s.h,
                s.w,
                window,
                shift,
            )
        };
        let q = pack(self.g, q);
        let k = pack(self.g, k);
        let v = pack(self.g, v);
        let windows_y = (s.h + shift).div_ceil(window);
        let windows_x = (s.w + shift).div_ceil(window);
        let heads = self.config.batch * windows_y * windows_x * (s.channels / head_dim);
        let attended = self
            .g
            .multi_head_attn(q, k, v, heads, heads, head_dim, false);
        let attended = self.g.window_merge_2d(
            attended,
            self.config.batch,
            s.channels,
            s.h,
            s.w,
            window,
            shift,
        );
        let attended = self.conv(
            attended,
            &format!("{name}.attn.out.weight"),
            s,
            s.channels,
            1,
            1,
        );
        let residual = self.g.add(x, attended);

        let normalized = self.group_norm(residual, &format!("{name}.ffn.norm"), s);
        let mut mixed = self.conv(
            normalized,
            &format!("{name}.ffn.spatial.weight"),
            s,
            s.channels,
            3,
            1,
        );
        if let Some(time) = time {
            let projected = self.linear(
                time,
                &format!("{name}.ffn.time_proj"),
                self.config.time_embed_dim,
                s.channels,
            );
            let plane = self.broadcast_spatial(projected, s.channels, s.spatial());
            mixed = self.g.add(mixed, plane);
        }
        let mixed = self.g.silu(mixed);
        let expanded = self.conv(
            mixed,
            &format!("{name}.ffn.expand.weight"),
            s,
            2 * s.channels,
            1,
            1,
        );
        let expanded = self.g.silu(expanded);
        let expanded_shape = Shape {
            channels: 2 * s.channels,
            ..s
        };
        let projected = self.conv(
            expanded,
            &format!("{name}.ffn.project.weight"),
            expanded_shape,
            s.channels,
            1,
            1,
        );
        self.g.add(residual, projected)
    }
}

/// Build the network.
///
/// With `training`, the graph gains a `target` input and an MSE loss, and
/// [`Model::loss`] is set. Without it the graph ends at the prediction, ready
/// for an inference session.
///
/// Graph inputs:
/// - `cond`: `[batch, cond_channels, tile, tile]`, the conditioning planes.
/// - `x_t`: `[batch, target_channels, tile, tile]`, the noised residual.
///   Diffusion only.
/// - `t_emb`: `[batch, time_input_dim]`, the sinusoidal timestep embedding.
///   Diffusion only.
/// - `target`: `[batch, target_channels, tile, tile]`, training only. The
///   noise under diffusion, the residual under direct regression.
pub fn build(config: &ModelConfig, training: bool) -> Result<Model, String> {
    build_for_extent(config, training, [config.tile, config.tile])
}

/// Build the network for a rectangular runtime extent.
///
/// Convolution weights are independent of the spatial dimensions, so a
/// checkpoint trained on square crops can be instantiated for a full Blade
/// frame. The extent still has to survive every U-Net downsampling level.
pub fn build_for_extent(
    config: &ModelConfig,
    training: bool,
    extent: [u32; 2],
) -> Result<Model, String> {
    config.validate()?;
    config.validate_extent(extent)?;

    let mut graph = Graph::new();
    let mut builder = Builder {
        g: &mut graph,
        config,
        params: Vec::new(),
    };

    let batch = config.batch;
    let [width, height] = extent;
    let spatial = width * height;
    let cond = builder
        .g
        .input("cond", &[config.cond_len_for_extent(extent)]);

    // Under diffusion the network sees the noised residual next to the
    // conditioning, and the noise level tells it how much of what it sees is
    // signal. Direct regression has neither.
    let (input, time) = match config.objective {
        Objective::Diffusion => {
            let x_t = builder
                .g
                .input("x_t", &[config.target_len_for_extent(extent)]);
            let joined = builder.g.concat(
                x_t,
                cond,
                batch,
                config.target_channels(),
                config.cond_channels(),
                spatial,
            );

            let t_emb = builder
                .g
                .input("t_emb", &[batch as usize, config.time_input_dim as usize]);
            let embedded = builder.linear(
                t_emb,
                "time.in",
                config.time_input_dim,
                config.time_embed_dim,
            );
            let embedded = builder.g.silu(embedded);
            let embedded = builder.linear(
                embedded,
                "time.out",
                config.time_embed_dim,
                config.time_embed_dim,
            );
            (joined, Some(embedded))
        }
        Objective::Direct => (cond, None),
    };

    // Stem.
    let stem_shape = Shape {
        channels: config.in_channels(),
        h: height,
        w: width,
    };
    let mut h = builder.conv(input, "stem.weight", stem_shape, config.base_channels, 3, 1);
    let mut shape = Shape {
        channels: config.base_channels,
        h: height,
        w: width,
    };

    // Encoder. One skip per level, taken before the downsample.
    let levels = config.levels();
    let mut skips: Vec<(NodeId, Shape)> = Vec::new();
    for level in 0..levels {
        let width = config.channels_at(level);
        for block in 0..config.blocks_per_level {
            h = builder.resblock(h, time, &format!("down.{level}.{block}"), shape, width);
            shape.channels = width;
        }
        if level + 1 < levels {
            skips.push((h, shape));
            // Strided convolution rather than pooling: the downsample gets to
            // learn what to keep, and it is the same cost as a 3x3 stride-1.
            h = builder.conv(h, &format!("down.{level}.pool.weight"), shape, width, 3, 2);
            shape = Shape {
                channels: width,
                h: shape.h / 2,
                w: shape.w / 2,
            };
        }
    }

    // Middle. Local attention lives only at the narrowest resolution: this
    // buys a 32x32-input-pixel receptive window at the default three levels,
    // while avoiding quadratic full-frame attention and preserving the
    // high-resolution convolutional skip path.
    let middle = shape.channels;
    match config.backbone {
        Backbone::Conv => {
            h = builder.resblock(h, time, "middle.0", shape, middle);
            h = builder.resblock(h, time, "middle.1", shape, middle);
        }
        Backbone::HybridWindowAttention { window, head_dim } => {
            h = builder.hybrid_window_block(
                h,
                time,
                "middle.0",
                shape,
                WindowAttentionSpec {
                    window,
                    head_dim,
                    shift: 0,
                },
            );
            h = builder.hybrid_window_block(
                h,
                time,
                "middle.1",
                shape,
                WindowAttentionSpec {
                    window,
                    head_dim,
                    shift: window / 2,
                },
            );
        }
    }

    // Decoder. Upsample, concatenate the matching skip, then narrow back down.
    for level in (0..levels - 1).rev() {
        h = builder
            .g
            .upsample_2x(h, batch, shape.channels, shape.h, shape.w);
        let upsampled = Shape {
            channels: shape.channels,
            h: shape.h * 2,
            w: shape.w * 2,
        };

        let (skip, skip_shape) = skips.pop().expect("a skip per encoder level");
        debug_assert_eq!(skip_shape.h, upsampled.h);
        h = builder.g.concat(
            h,
            skip,
            batch,
            upsampled.channels,
            skip_shape.channels,
            upsampled.spatial(),
        );
        shape = Shape {
            channels: upsampled.channels + skip_shape.channels,
            h: upsampled.h,
            w: upsampled.w,
        };

        let width = config.channels_at(level);
        for block in 0..config.blocks_per_level {
            h = builder.resblock(h, time, &format!("up.{level}.{block}"), shape, width);
            shape.channels = width;
        }
    }

    // Head. The output convolution starts at zero so the network's first
    // prediction is exactly zero — under diffusion that is a far better
    // starting point than noise, and under direct regression it means the
    // untrained network passes the input through unchanged.
    let h = builder.group_norm(h, "head.norm", shape);
    let h = builder.g.silu(h);
    let head_weight = builder.param(
        "head.conv.weight",
        (config.target_channels() * shape.channels * 9) as usize,
        InitKind::Zeros,
    );
    let output = builder.g.conv2d(
        h,
        head_weight,
        batch,
        shape.channels,
        shape.h,
        shape.w,
        config.target_channels(),
        3,
        3,
        1,
        1,
    );

    let params = builder.params;
    let loss = if training {
        let target = graph.input("target", &[config.target_len_for_extent(extent)]);
        let loss = graph.mse_loss(output, target);
        graph.set_outputs(vec![loss]);
        Some(loss)
    } else {
        graph.set_outputs(vec![output]);
        None
    };

    Ok(Model {
        graph,
        config: config.clone(),
        input_extent: extent,
        output,
        loss,
        params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> ModelConfig {
        ModelConfig {
            batch: 1,
            tile: 16,
            base_channels: 16,
            level_multipliers: vec![1, 2],
            blocks_per_level: 1,
            num_groups: 8,
            time_input_dim: 16,
            time_embed_dim: 32,
            cond_planes: PlaneSet::new().with(Plane::Color),
            // Most graph tests predate the deployment default and exercise
            // the superset path; direct-specific tests override this below.
            objective: Objective::Diffusion,
            ..ModelConfig::default()
        }
    }

    #[test]
    fn default_is_the_measured_deployment_baseline() {
        let c = ModelConfig::default();
        assert_eq!(c.objective, Objective::Direct);
        assert_eq!(c.base_channels, 24);
        assert_eq!(c.blocks_per_level, 1);
        assert_eq!(c.batch, 8);
    }

    #[test]
    fn channel_counts_follow_the_scale() {
        let mut c = small();
        assert_eq!(c.target_channels(), 12); // scale 2 -> 3 * 4
        assert_eq!(c.cond_channels(), 3);
        assert_eq!(c.in_channels(), 15);
        c.objective = Objective::Direct;
        assert_eq!(c.in_channels(), 3, "direct sees only the conditioning");
        c.scale = 4;
        assert_eq!(c.target_channels(), 48);
    }

    #[test]
    fn validation_catches_bad_geometry() {
        let mut c = small();
        assert!(c.validate().is_ok());

        c.level_multipliers = vec![1, 2, 4, 8, 16];
        assert!(
            c.validate().unwrap_err().contains("collapses"),
            "a 16px tile cannot survive 5 levels"
        );

        c = small();
        c.tile = 12; // 4 levels shrink by 8, and 12 is not a multiple of 8
        c.level_multipliers = vec![1, 2, 4, 8];
        assert!(c.validate().unwrap_err().contains("divisible by 8"));

        c = small();
        c.num_groups = 7; // does not divide 16
        assert!(c.validate().unwrap_err().contains("num_groups"));

        c = small();
        c.scale = 1;
        assert!(c.validate().unwrap_err().contains("scale"));

        c = small();
        c.time_input_dim = 15;
        assert!(c.validate().unwrap_err().contains("even"));

        c = small();
        c.cond_planes = PlaneSet::new();
        assert!(c.validate().unwrap_err().contains("empty"));
    }

    #[test]
    fn inference_graph_ends_at_the_prediction() {
        let model = build(&small(), false).unwrap();
        assert!(model.loss.is_none());
        assert_eq!(model.graph.outputs(), &[model.output]);
        let ty = &model.graph.node(model.output).ty;
        assert_eq!(ty.num_elements(), model.config.target_len());
    }

    #[test]
    fn checkpoint_weights_build_for_a_rectangular_frame() {
        let config = small();
        let extent = [32, 24];
        let model = build_for_extent(&config, false, extent).unwrap();
        assert_eq!(model.input_extent, extent);
        let ty = &model.graph.node(model.output).ty;
        assert_eq!(ty.num_elements(), config.target_len_for_extent(extent),);
    }

    #[test]
    fn hybrid_attention_is_a_bottleneck_swap_not_a_second_network() {
        let mut config = small();
        config.objective = Objective::Direct;
        config.backbone = Backbone::HybridWindowAttention {
            window: 4,
            head_dim: 8,
        };
        let model = build(&config, true).unwrap();
        let attention_blocks = model
            .params
            .iter()
            .filter(|parameter| parameter.name.ends_with("attn.q.weight"))
            .count();
        assert_eq!(attention_blocks, 2);
        assert!(
            model
                .params
                .iter()
                .any(|parameter| parameter.name.starts_with("down.")),
            "the hybrid must retain the convolutional encoder"
        );
        assert!(
            model
                .params
                .iter()
                .any(|parameter| parameter.name.starts_with("up.")),
            "the hybrid must retain the convolutional decoder"
        );

        let mut names: Vec<_> = model
            .params
            .iter()
            .map(|parameter| parameter.name.as_str())
            .collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "hybrid parameter names collide");
    }

    #[test]
    fn hybrid_attention_supports_the_deployment_extent() {
        let mut config = ModelConfig {
            batch: 1,
            ..ModelConfig::default()
        };
        config.backbone = Backbone::HybridWindowAttention {
            window: 8,
            head_dim: 16,
        };
        let model = build_for_extent(&config, false, [960, 540]).unwrap();
        assert_eq!(model.input_extent, [960, 540]);
    }

    #[test]
    fn attention_shape_errors_are_reported_without_panicking() {
        let mut config = small();
        config.backbone = Backbone::HybridWindowAttention {
            window: 0,
            head_dim: 0,
        };
        assert!(config.validate().unwrap_err().contains("window"));

        config.backbone = Backbone::HybridWindowAttention {
            window: 4,
            head_dim: 64,
        };
        assert!(config.validate().unwrap_err().contains("bottleneck"));
    }

    #[test]
    fn training_graph_ends_at_the_loss() {
        let model = build(&small(), true).unwrap();
        let loss = model.loss.expect("training graph has a loss");
        assert_eq!(model.graph.outputs(), &[loss]);
        // A loss is a scalar.
        assert_eq!(model.graph.node(loss).ty.num_elements(), 1);
    }

    #[test]
    fn every_parameter_is_declared_once() {
        let model = build(&small(), true).unwrap();
        let mut names: Vec<&str> = model.params.iter().map(|p| p.name.as_str()).collect();
        let total = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), total, "duplicate parameter names");
        assert!(total > 20, "only {total} parameters, the U-Net is too thin");

        // The head starts at zero so the first prediction is zero.
        let head = model
            .params
            .iter()
            .find(|p| p.name == "head.conv.weight")
            .expect("head weight");
        assert_eq!(head.kind, InitKind::Zeros);
    }

    #[test]
    fn direct_objective_drops_the_noise_inputs() {
        let mut c = small();
        c.objective = Objective::Direct;
        let model = build(&c, true).unwrap();
        // No timestep projection parameters at all when there is no timestep.
        assert!(
            !model.params.iter().any(|p| p.name.starts_with("time.")),
            "direct regression should not carry a timestep MLP"
        );
        assert!(!model.params.iter().any(|p| p.name.contains("time_proj")));
    }

    #[test]
    fn diffusion_objective_carries_the_timestep_mlp() {
        let model = build(&small(), true).unwrap();
        assert!(model.params.iter().any(|p| p.name == "time.in.weight"));
        assert!(model.params.iter().any(|p| p.name.contains("time_proj")));
    }
}

#[cfg(test)]
mod flop_tests {
    use super::*;

    fn default_backbone() -> ModelConfig {
        ModelConfig {
            scale: 2,
            tile: 512,
            batch: 1,
            base_channels: 64,
            level_multipliers: vec![1, 2, 4],
            blocks_per_level: 2,
            cond_planes: PlaneSet::new()
                .with(Plane::Color)
                .with(Plane::Depth)
                .with(Plane::Normal)
                .with(Plane::DiffuseAlbedo)
                .with(Plane::SpecularF0)
                .with(Plane::Roughness),
            objective: Objective::Direct,
            ..ModelConfig::default()
        }
    }

    /// Pinned against an independent hand count of the same architecture, so a
    /// change to the builder that the estimator does not follow shows up.
    #[test]
    fn the_estimate_matches_a_hand_count() {
        // 512x512 input, so 1024x1024 out.
        let gflop = default_backbone().flops(1024 * 1024);
        assert!(
            (500.0..580.0).contains(&gflop),
            "expected ~539 GFLOP for the default backbone, got {gflop:.0}"
        );
    }

    #[test]
    fn cost_scales_with_output_pixels() {
        let config = default_backbone();
        let small = config.flops(1024 * 1024);
        let big = config.flops(2048 * 2048);
        assert!(
            (big / small - 4.0).abs() < 1e-6,
            "four times the pixels should cost four times as much: {small} vs {big}",
        );
    }

    /// Halving the width should quarter the arithmetic, near enough — the stem
    /// and head scale linearly rather than quadratically, so it is not exact.
    #[test]
    fn width_dominates_the_cost() {
        let mut narrow = default_backbone();
        narrow.base_channels = 32;
        let ratio = default_backbone().flops(1 << 20) / narrow.flops(1 << 20);
        assert!(
            (3.4..4.0).contains(&ratio),
            "halving the width changed the cost by {ratio:.2}x"
        );
    }

    #[test]
    fn hybrid_candidate_stays_in_the_baseline_compute_class() {
        let conv = ModelConfig::default();
        let mut hybrid = conv.clone();
        hybrid.backbone = Backbone::HybridWindowAttention {
            window: 8,
            head_dim: 16,
        };
        let conv_cost = conv.flops(1920 * 1080);
        let hybrid_cost = hybrid.flops(1920 * 1080);
        let ratio = hybrid_cost / conv_cost;
        assert!(
            (0.9..1.1).contains(&ratio),
            "candidate arithmetic is {ratio:.2}x baseline ({hybrid_cost:.1} vs {conv_cost:.1} GFLOP)"
        );
    }
}
