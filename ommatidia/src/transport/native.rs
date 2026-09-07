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
    level: u32,
    ready: u32,
    exposure: f32,
    diffuse_frames: f32,
    specular_frames: f32,
    jitter: [f32; 2],
    pad: [u32; 2],
}
#[derive(blade_macros::ShaderData)]
struct Data {
    params: Params,
    rays: gpu::BufferPiece,
    surfaces: gpu::BufferPiece,
    previous: gpu::BufferPiece,
    candidates: gpu::BufferPiece,
    features: gpu::BufferPiece,
    history: gpu::BufferPiece,
    prior: gpu::BufferPiece,
    moments: gpu::BufferPiece,
    ages: gpu::BufferPiece,
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
    pipelines: [gpu::ComputePipeline; 4],
    state: [gpu::Buffer; 2],
    moments: gpu::Buffer,
    ages: gpu::Buffer,
    rays: gpu::Buffer,
    surfaces: gpu::Buffer,
    output: gpu::Buffer,
    current: usize,
    ready: bool,
}
impl Native {
    pub fn new(context: Arc<gpu::Context>, config: Config, low: [u32; 2]) -> Result<Self, String> {
        let network = graph::build(config, low, 0)?;
        let mut session = crate::gpu::inference_session(&network.graph, Arc::clone(&context));
        network.initialize(&mut session, 1);
        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("prepare.wgsl"),
            naga_module: None,
        });
        let layout = <Data as gpu::ShaderData>::layout();
        let pipelines = ["seed", "atrous", "pack", "resolve"].map(|name| {
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
            buffer("lobe-state-a", (n * std::mem::size_of::<State>()) as u64),
            buffer("lobe-state-b", (n * std::mem::size_of::<State>()) as u64),
        ];
        let moments = buffer("lobe-moments", (n * 16) as u64);
        let ages = buffer("lobe-ages", (n * 8) as u64);
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
            moments,
            ages,
            rays,
            surfaces,
            output,
            current: 0,
            ready: false,
        })
    }
    pub fn reset(&mut self) {
        self.ready = false;
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
        level: u32,
    ) -> Data {
        Data {
            params: Params {
                w: self.low[0],
                h: self.low[1],
                scale: self.config.scale,
                level,
                ready: u32::from(self.ready),
                exposure: self.config.exposure,
                diffuse_frames: self.config.diffuse_frames,
                specular_frames: self.config.specular_frames,
                jitter,
                pad: [0; 2],
            },
            rays,
            surfaces,
            previous: self.state[self.current].into(),
            next: self.state[1 - self.current].into(),
            moments: self.moments.into(),
            ages: self.ages.into(),
            output,
            candidates: self.session.input_buffer("f0.candidates").unwrap(),
            features: self.session.input_buffer("f0.features").unwrap(),
            history: self.session.input_buffer("f0.history").unwrap(),
            prior: self.session.input_buffer("f0.prior").unwrap(),
            image: self.session.output_buffer(0).unwrap(),
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
    ) {
        self.dispatch(
            encoder,
            0,
            &self.data(rays, surfaces, self.output.into(), jitter, 0),
        );
        for level in 1..SCALES {
            self.dispatch(
                encoder,
                1,
                &self.data(rays, surfaces, self.output.into(), jitter, level as u32),
            );
        }
        self.dispatch(
            encoder,
            2,
            &self.data(rays, surfaces, self.output.into(), jitter, 0),
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
            3,
            &self.data(self.rays.into(), surfaces, output, [0.0; 2], 0),
        );
        self.current = 1 - self.current;
        self.ready = true;
    }
    pub fn process(&mut self, frame: &Frame) -> Result<Vec<f32>, String> {
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
        );
        let sync = self.context.submit(&mut encoder);
        if !self
            .context
            .wait_for(&sync, 60000)
            .map_err(|e| format!("{e:?}"))?
        {
            return Err("preparation timeout".into());
        }
        self.session.step();
        self.session.wait();
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
        let values = unsafe {
            std::slice::from_raw_parts(self.output.data().cast::<f32>(), frame.surfaces.len() * 4)
        };
        let rgb = values
            .chunks_exact(4)
            .flat_map(|v| v[..3].iter().copied())
            .collect();
        self.context.destroy_command_encoder(&mut encoder);
        Ok(rgb)
    }
    /// Offline readback after a completed resolve. No targets accepted; no
    /// history or parameters mutated. Reads actual GPU-prepared candidates.
    pub fn read_candidates(&self) -> super::oracle::Candidates {
        let n = (self.low[0] * self.low[1] * self.config.scale.pow(2)) as usize;
        let read = |name: &str, len| {
            let buffer = self
                .session
                .plan()
                .input_buffers
                .iter()
                .find(|(n, _)| n == name)
                .unwrap()
                .1;
            let mut values = vec![0.0; len];
            self.session.read_buffer(buffer, &mut values);
            values
        };
        let mut selected = vec![0.0; CANDIDATES * 2 * n];
        self.session.read_output_by_index(1, &mut selected);
        super::oracle::Candidates {
            spatial: read("f0.candidates", SCALES * 6 * n),
            history: read("f0.history", 6 * n),
            prior: read("f0.prior", CANDIDATES * 2 * n),
            selected,
        }
    }
    /// Only call after waiting for the resolve submission.
    pub fn read_state(&self) -> Vec<State> {
        let n = (self.low[0] * self.low[1] * self.config.scale.pow(2)) as usize;
        unsafe {
            std::slice::from_raw_parts(self.state[self.current].data().cast::<State>(), n).to_vec()
        }
    }
}
impl Drop for Native {
    fn drop(&mut self) {
        self.session.wait();
        for p in &mut self.pipelines {
            self.context.destroy_compute_pipeline(p);
        }
        for b in self.state.into_iter().chain([
            self.moments,
            self.ages,
            self.rays,
            self.surfaces,
            self.output,
        ]) {
            self.context.destroy_buffer(b);
        }
    }
}
