//! GPU-resident preparation, local neural reconstruction, and per-lobe state.
//! The renderer-facing record methods accept GPU buffers; `process` is a
//! synchronous upload/readback convenience for offline quality tests only.
use super::*;
use blade_graphics as gpu;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    w: u32,
    h: u32,
    scale: u32,
    state_channels: u32,
    ready: u32,
    exposure: f32,
    jitter: [f32; 2],
}
#[derive(blade_macros::ShaderData)]
struct Data {
    params: Params,
    rays: gpu::BufferPiece,
    surfaces: gpu::BufferPiece,
    previous: gpu::BufferPiece,
    features: gpu::BufferPiece,
    history: gpu::BufferPiece,
    metadata: gpu::BufferPiece,
    valid: gpu::BufferPiece,
    exposure: gpu::BufferPiece,
    warp0: gpu::BufferPiece,
    warp1: gpu::BufferPiece,
    warp2: gpu::BufferPiece,
    warp3: gpu::BufferPiece,
    coeff0: gpu::BufferPiece,
    coeff1: gpu::BufferPiece,
    coeff2: gpu::BufferPiece,
    coeff3: gpu::BufferPiece,
    image: gpu::BufferPiece,
    next: gpu::BufferPiece,
    output: gpu::BufferPiece,
}

pub struct Native {
    context: Arc<gpu::Context>,
    pub session: meganeura::Session,
    pub network: graph::Network,
    config: Config,
    low: [u32; 2],
    pipelines: [gpu::ComputePipeline; 2],
    state: [gpu::Buffer; 2],
    rays: gpu::Buffer,
    surfaces: gpu::Buffer,
    output: gpu::Buffer,
    current: usize,
    ready: bool,
    timing: bool,
    last_timings: Option<[std::time::Duration; 3]>,
}
impl Native {
    pub fn new(context: Arc<gpu::Context>, config: Config, low: [u32; 2]) -> Result<Self, String> {
        Self::with_timing(context, config, low, false)
    }
    /// When enabled, `context` must have been created with GPU timing enabled.
    pub fn with_timing(
        context: Arc<gpu::Context>,
        config: Config,
        low: [u32; 2],
        timing: bool,
    ) -> Result<Self, String> {
        if timing && !context.capabilities().timing {
            return Err("GPU timestamps unavailable".into());
        }
        let network = graph::build(config, low, 0)?;
        let mut session =
            crate::gpu::inference_session_with_timing(&network.graph, Arc::clone(&context), timing);
        network.initialize(&mut session, 1);
        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("prepare.wgsl"),
            naga_module: None,
        });
        let layout = <Data as gpu::ShaderData>::layout();
        let pipelines = ["pack", "resolve"].map(|name| {
            context.create_compute_pipeline(gpu::ComputePipelineDesc {
                name,
                data_layouts: &[&layout],
                compute: shader.at(name),
            })
        });
        let n = (low[0] * low[1] * config.scale.pow(2)) as usize;
        let buffer = |name, size| {
            context.create_buffer(gpu::BufferDesc {
                name,
                size,
                memory: gpu::Memory::Shared,
            })
        };
        let state = [
            buffer(
                "recurrent-state-a",
                (n * config.state_channels() * 4) as u64,
            ),
            buffer(
                "recurrent-state-b",
                (n * config.state_channels() * 4) as u64,
            ),
        ];
        let rays = buffer(
            "observation-upload",
            (low[0] * low[1]) as u64 * std::mem::size_of::<Ray>() as u64,
        );
        let surfaces = buffer(
            "surface-upload",
            (n * std::mem::size_of::<Surface>()) as u64,
        );
        let output = buffer("reconstruction-readback", (n * 16) as u64);
        Ok(Self {
            context,
            session,
            network,
            config,
            low,
            pipelines,
            state,
            rays,
            surfaces,
            output,
            current: 0,
            ready: false,
            timing,
            last_timings: None,
        })
    }
    pub fn reset(&mut self) {
        self.ready = false;
    }
    /// Preparation, neural inference and resolve GPU pass spans for the last
    /// completed `advance`. Excludes uploads, readback and CPU/queue gaps.
    pub fn gpu_timings(&self) -> Option<[std::time::Duration; 3]> {
        self.last_timings
    }
    /// Requested resident buffer bytes, including this wrapper's offline
    /// upload/readback buffers; excludes driver objects and temporary staging.
    pub fn buffer_memory_bytes(&self) -> usize {
        let low_texels = (self.low[0] * self.low[1]) as usize;
        let high_texels = low_texels * self.config.scale.pow(2) as usize;
        self.session.memory_summary().total_allocated_bytes()
            + low_texels * std::mem::size_of::<Ray>()
            + high_texels
                * (2 * self.config.state_channels() * 4 + std::mem::size_of::<Surface>() + 16)
    }
    pub fn sync_parameters(&mut self, source: &meganeura::Session) {
        let names: Vec<_> = self
            .network
            .params
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        for (name, values) in names.iter().zip(source.read_params(&names)) {
            self.session.set_parameter(name, &values);
        }
    }
    fn data(
        &self,
        rays: gpu::BufferPiece,
        surfaces: gpu::BufferPiece,
        output: gpu::BufferPiece,
        jitter: [f32; 2],
        exposure: f32,
    ) -> Data {
        Data {
            params: Params {
                w: self.low[0],
                h: self.low[1],
                scale: self.config.scale,
                state_channels: self.config.state_channels() as u32,
                ready: u32::from(self.ready),
                exposure,
                jitter,
            },
            rays,
            surfaces,
            previous: self.state[self.current].into(),
            next: self.state[1 - self.current].into(),
            output,
            features: self.session.input_buffer("f0.features").unwrap(),
            history: self.session.input_buffer("f0.history").unwrap(),
            metadata: self.session.input_buffer("f0.metadata").unwrap(),
            valid: self.session.input_buffer("f0.valid").unwrap(),
            exposure: self.session.input_buffer("f0.exposure").unwrap(),
            warp0: self.session.input_buffer("f0.warp0").unwrap(),
            warp1: self.session.input_buffer("f0.warp1").unwrap(),
            warp2: self.session.input_buffer("f0.warp2").unwrap(),
            warp3: self.session.input_buffer("f0.warp3").unwrap(),
            coeff0: self.session.input_buffer("f0.coeff0").unwrap(),
            coeff1: self.session.input_buffer("f0.coeff1").unwrap(),
            coeff2: self.session.input_buffer("f0.coeff2").unwrap(),
            coeff3: self.session.input_buffer("f0.coeff3").unwrap(),
            image: self.session.output_buffer(2).unwrap(),
        }
    }
    fn dispatch(&self, encoder: &mut gpu::CommandEncoder, stage: usize, data: &Data) {
        let mut pass = encoder.compute("transport");
        let mut commands = pass.with(&self.pipelines[stage]);
        commands.bind(0, data);
        commands.dispatch([
            (self.low[0] * self.config.scale).div_ceil(8),
            (self.low[1] * self.config.scale).div_ceil(8),
            1,
        ]);
    }
    /// Submit this preparation before calling `session.step()` on the same context.
    pub fn record_prepare(
        &self,
        encoder: &mut gpu::CommandEncoder,
        rays: gpu::BufferPiece,
        surfaces: gpu::BufferPiece,
        jitter: [f32; 2],
        exposure: f32,
    ) {
        self.dispatch(
            encoder,
            0,
            &self.data(rays, surfaces, self.output.into(), jitter, exposure),
        );
    }
    /// Record after the network submission; output is `width*height` linear RGBA f32.
    pub fn record_resolve(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        surfaces: gpu::BufferPiece,
        output: gpu::BufferPiece,
    ) {
        self.dispatch(
            encoder,
            1,
            &self.data(self.rays.into(), surfaces, output, [0.0; 2], 1.0),
        );
        self.current = 1 - self.current;
        self.ready = true;
    }
    pub fn process(&mut self, frame: &Frame) -> Result<Vec<f32>, String> {
        self.advance(frame)?;
        // Read shared VRAM contiguously before deinterleaving on the CPU.
        let values = unsafe {
            std::slice::from_raw_parts(self.output.data().cast::<f32>(), frame.surfaces.len() * 4)
                .to_vec()
        };
        Ok(values
            .chunks_exact(4)
            .flat_map(|v| v[..3].iter().copied())
            .collect())
    }
    /// Offline recurrent step without reading displayed RGB back. Training
    /// warmup and unrolls only need state and prepared network inputs.
    pub fn advance(&mut self, frame: &Frame) -> Result<(), String> {
        self.last_timings = None;
        frame.validate(self.config)?;
        if frame.low != self.low {
            return Err("frame extent changed; recreate the reconstructor".into());
        }
        // This convenience API waits before reusing host-visible buffers.
        self.session.wait();
        unsafe {
            std::ptr::copy_nonoverlapping(
                frame.rays.as_ptr(),
                self.rays.data().cast::<Ray>(),
                frame.rays.len(),
            );
            std::ptr::copy_nonoverlapping(
                frame.surfaces.as_ptr(),
                self.surfaces.data().cast::<Surface>(),
                frame.surfaces.len(),
            );
        }
        let mut encoder = self
            .context
            .create_command_encoder(gpu::CommandEncoderDesc {
                name: "transport-quality",
                buffer_count: 2,
                manual_barriers: false,
            });
        encoder.start();
        self.record_prepare(
            &mut encoder,
            self.rays.into(),
            self.surfaces.into(),
            frame.jitter,
            frame.exposure,
        );
        let sync = self.context.submit(&mut encoder);
        if !self
            .context
            .wait_for(&sync, 60000)
            .map_err(|e| format!("{e:?}"))?
        {
            return Err("preparation timeout".into());
        }
        let preparation = self
            .timing
            .then(|| encoder.last_timing().pass_durations().map(|(_, d)| d).sum());
        self.session.step();
        self.session.wait();
        let inference = if self.timing {
            let timings = self.session.gpu_timings();
            if timings.is_empty() {
                self.context.destroy_command_encoder(&mut encoder);
                return Err("missing neural GPU timestamps".into());
            }
            timings.iter().map(|(_, d)| *d).sum()
        } else {
            std::time::Duration::ZERO
        };
        encoder.start();
        self.record_resolve(&mut encoder, self.surfaces.into(), self.output.into());
        let sync = self.context.submit(&mut encoder);
        if !self
            .context
            .wait_for(&sync, 60000)
            .map_err(|e| format!("{e:?}"))?
        {
            return Err("resolve timeout".into());
        }
        if let Some(preparation) = preparation {
            let resolve = encoder.last_timing().pass_durations().map(|(_, d)| d).sum();
            self.last_timings = Some([preparation, inference, resolve]);
        }
        self.context.destroy_command_encoder(&mut encoder);
        Ok(())
    }
    /// Offline readback after a completed process/resolve. These are the actual
    /// geometric reprojection validity, not a learned alpha or rejection test.
    pub fn read_history_validity(&self) -> Vec<f32> {
        let n = (self.low[0] * self.low[1] * self.config.scale.pow(2)) as usize;
        self.read_input("f0.valid", n)
    }
    /// Offline parity check of the actual inputs prepared by WGSL.
    pub fn read_features(&self) -> Vec<f32> {
        let lr = (self.low[0] * self.low[1]) as usize;
        self.read_input("f0.features", self.config.observation_channels() * lr)
    }
    fn read_input(&self, name: &str, len: usize) -> Vec<f32> {
        let buffer = self
            .session
            .plan()
            .input_buffers
            .iter()
            .find(|(key, _)| key == name)
            .unwrap()
            .1;
        let mut values = vec![0.0; len];
        self.session.read_buffer(buffer, &mut values);
        values
    }
    /// Offline training readback after `advance(frame)`. Maps are generated by
    /// WGSL, not reconstructed on the CPU. Phase 3 replaces this readback loop.
    pub fn read_prepared(&self, frame: &Frame) -> cpu::Prepared {
        assert_eq!(frame.low, self.low);
        let n = frame.surfaces.len();
        let state_len = n * self.config.state_channels();
        cpu::Prepared {
            features: self.read_features(),
            validity: self.read_history_validity(),
            history: self.read_input("f0.history", state_len),
            metadata: self.read_input("f0.metadata", 7 * n),
            exposure: self.read_input("f0.exposure", 1)[0],
            indices: std::array::from_fn(|k| {
                self.read_input(&format!("f0.warp{k}"), state_len)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect()
            }),
            coefficients: std::array::from_fn(|k| {
                self.read_input(&format!("f0.coeff{k}"), state_len)
            }),
        }
    }
    /// Only call after waiting for the resolve submission.
    pub fn read_state(&self) -> State {
        let n = (self.low[0] * self.low[1] * self.config.scale.pow(2)) as usize;
        if !self.ready {
            return State::default();
        }
        State {
            values: unsafe {
                std::slice::from_raw_parts(
                    self.state[self.current].data().cast::<f32>(),
                    n * self.config.state_channels(),
                )
                .to_vec()
            },
        }
    }
}
impl Drop for Native {
    fn drop(&mut self) {
        self.session.wait();
        for p in &mut self.pipelines {
            self.context.destroy_compute_pipeline(p);
        }
        for b in self
            .state
            .into_iter()
            .chain([self.rays, self.surfaces, self.output])
        {
            self.context.destroy_buffer(b);
        }
    }
}
