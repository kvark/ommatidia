//! One cropped BPTT graph, B GPU-resident cursors and one mean-gradient update.
use crate::{Result, sampler::Batch};
use blade_graphics as gpu;
use ommatidia::transport::{
    Config, Ray, Surface, graph,
    native::{PackInput, Packer},
};
use std::{sync::Arc, time::Instant};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    w: u32,
    h: u32,
    scale: u32,
    pad: u32,
}
#[derive(blade_macros::ShaderData)]
struct Targets {
    params: Params,
    surfaces: gpu::BufferPiece,
    rgb: gpu::BufferPiece,
    rgb_target: gpu::BufferPiece,
    albedo: gpu::BufferPiece,
    emission: gpu::BufferPiece,
}

#[derive(Clone, Copy)]
struct Upload {
    rays: gpu::BufferPiece,
    surfaces: gpu::BufferPiece,
    lobes: gpu::BufferPiece,
    rgb: gpu::BufferPiece,
}

#[derive(Default, serde::Serialize)]
pub struct Timing {
    pub upload: f64,
    pub preparation_submit: f64,
    pub step_wait: f64,
    pub carry_wait: f64,
    pub loss_read: f64,
}

pub struct Trainer {
    pub network: graph::Network,
    pub session: meganeura::Session,
    context: Arc<gpu::Context>,
    config: Config,
    low: [u32; 2],
    unroll: usize,
    batch: usize,
    packer: Packer,
    targets: gpu::ComputePipeline,
    encoder: gpu::CommandEncoder,
    state: gpu::Buffer,
    upload: gpu::Buffer,
    losses: gpu::Buffer,
    frames: Vec<Upload>,
    state_bytes: usize,
}
impl Trainer {
    pub fn new(
        context: Arc<gpu::Context>,
        config: Config,
        low: [u32; 2],
        unroll: usize,
        batch: usize,
        margin: u32,
    ) -> Result<Self> {
        if batch == 0 || batch > u32::MAX as usize / 2 {
            return Err("invalid batch size".into());
        }
        let network = graph::build_training(config, low, unroll, margin)?;
        let mut session = ommatidia::gpu::training_session(&network.graph, Arc::clone(&context));
        session.set_grad_accumulate(batch as u32);
        session.set_grad_clip_norm(1.0);
        let packer = Packer::new(Arc::clone(&context), config, low)?;
        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("targets.wgsl"),
            naga_module: None,
        });
        let targets = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "training-targets",
            data_layouts: &[&<Targets as gpu::ShaderData>::layout()],
            compute: shader.at("pack_targets"),
        });
        let pixels = (low[0] * low[1]) as usize;
        let n = pixels * config.scale.pow(2) as usize;
        let bytes = [
            pixels * size_of::<Ray>(),
            n * size_of::<Surface>(),
            6 * n * 4,
            3 * n * 4,
        ];
        let frame_bytes: usize = bytes.iter().sum();
        let buffer = |name, size| {
            context.create_buffer(gpu::BufferDesc {
                name,
                size: size as u64,
                memory: gpu::Memory::Shared,
            })
        };
        let state_bytes = config.state_channels() * n * 4;
        let state = buffer("training-cursors", state_bytes * batch);
        // Defined even for not-yet-visited cursors in an early checkpoint.
        unsafe {
            std::ptr::write_bytes(state.data(), 0, state_bytes * batch);
        }
        let upload = buffer("training-batch-upload", frame_bytes * batch * unroll);
        let losses = buffer("training-scalar-losses", batch * 4);
        let frames = (0..batch * unroll)
            .map(|i| {
                let mut offset = i * frame_bytes;
                let mut piece = |size| {
                    let p = upload.at(offset as u64);
                    offset += size;
                    p
                };
                Upload {
                    rays: piece(bytes[0]),
                    surfaces: piece(bytes[1]),
                    lobes: piece(bytes[2]),
                    rgb: piece(bytes[3]),
                }
            })
            .collect();
        let encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "training-pack-carry",
            buffer_count: (2 * batch) as u32,
            manual_barriers: false,
        });
        Ok(Self {
            network,
            session,
            context,
            config,
            low,
            unroll,
            batch,
            packer,
            targets,
            encoder,
            state,
            upload,
            losses,
            frames,
            state_bytes,
        })
    }
    pub fn initialize(&mut self, seed: u64, means: [f64; 6]) {
        self.network.initialize(&mut self.session, seed);
        let bias: Vec<_> = means
            .into_iter()
            .flat_map(|v| {
                std::iter::repeat_n(
                    v.ln().clamp(-16.0, 11.0) as f32,
                    self.config.scale.pow(2) as usize,
                )
            })
            .collect();
        self.session.set_parameter("head.radiance.bias", &bias);
    }
    pub fn share_parameters(&mut self, evaluation: &mut meganeura::Session) -> Result<()> {
        for parameter in &self.network.params {
            evaluation.share_parameter_from(&mut self.session, &parameter.name)?;
            assert!(evaluation.shares_parameter(&self.session, &parameter.name));
        }
        Ok(())
    }
    pub fn step(
        &mut self,
        batch: &Batch,
        rate: f32,
        weights: graph::LossWeights,
    ) -> Result<(f32, Timing)> {
        self.run_batch(batch, Some(rate), weights)
    }
    /// Same preparation/carry execution without changing weights, for parity checks.
    pub fn run_batch(
        &mut self,
        batch: &Batch,
        rate: Option<f32>,
        weights: graph::LossWeights,
    ) -> Result<(f32, Timing)> {
        if batch.frames.len() != self.batch
            || batch.windows.len() != self.batch
            || batch.frames.iter().any(|v| v.len() != self.unroll)
        {
            return Err("batch shape mismatch".into());
        }
        let mut timing = Timing::default();
        let start = Instant::now();
        self.session.clear_optimizer();
        self.session.zero_grad();
        weights.feed(&mut self.session);
        for ((frame, target), upload) in batch.frames.iter().flatten().zip(&self.frames) {
            let pixels = (self.low[0] * self.low[1]) as usize;
            let high_pixels = pixels * self.config.scale.pow(2) as usize;
            if frame.low != self.low
                || frame.rays.len() != pixels
                || frame.surfaces.len() != high_pixels
                || target.lobes.len() != 6 * high_pixels
                || target.rgb.len() != 3 * high_pixels
            {
                return Err("crop extent mismatch".into());
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    frame.rays.as_ptr(),
                    upload.rays.data().cast::<Ray>(),
                    frame.rays.len(),
                );
                std::ptr::copy_nonoverlapping(
                    frame.surfaces.as_ptr(),
                    upload.surfaces.data().cast::<Surface>(),
                    frame.surfaces.len(),
                );
                std::ptr::copy_nonoverlapping(
                    target.lobes.as_ptr(),
                    upload.lobes.data().cast::<f32>(),
                    target.lobes.len(),
                );
                std::ptr::copy_nonoverlapping(
                    target.rgb.as_ptr(),
                    upload.rgb.data().cast::<f32>(),
                    target.rgb.len(),
                );
            }
        }
        timing.upload = start.elapsed().as_secs_f64();
        let n = (self.low[0] * self.low[1] * self.config.scale.pow(2)) as usize;
        let mut last = None;
        for cursor in 0..self.batch {
            let start = Instant::now();
            self.encoder.start();
            for slot in 0..self.unroll {
                let upload = self.frames[cursor * self.unroll + slot];
                let frame = &batch.frames[cursor][slot].0;
                self.packer.record(
                    &mut self.encoder,
                    &self.session,
                    slot,
                    &PackInput {
                        rays: upload.rays,
                        surfaces: upload.surfaces,
                        previous: self.state.at((cursor * self.state_bytes) as u64),
                        jitter: frame.jitter,
                        exposure: frame.exposure,
                        ready: slot > 0 || !batch.windows[cursor].reset,
                    },
                );
                let input = |name| {
                    self.session
                        .input_buffer(&format!("f{slot}.{name}"))
                        .unwrap()
                };
                self.encoder.transfer("target-lobes").copy_buffer_to_buffer(
                    upload.lobes,
                    input("target"),
                    (6 * n * 4) as u64,
                );
                let data = Targets {
                    params: Params {
                        w: self.low[0],
                        h: self.low[1],
                        scale: self.config.scale,
                        pad: 0,
                    },
                    surfaces: upload.surfaces,
                    rgb: upload.rgb,
                    rgb_target: input("rgb.target"),
                    albedo: input("rgb.albedo"),
                    emission: input("rgb.emission"),
                };
                let mut pass = self.encoder.compute("target-rgb");
                let mut commands = pass.with(&self.targets);
                commands.bind(0, &data);
                commands.dispatch([
                    (self.low[0] * self.config.scale).div_ceil(8),
                    (self.low[1] * self.config.scale).div_ceil(8),
                    1,
                ]);
            }
            self.context.submit(&mut self.encoder);
            timing.preparation_submit += start.elapsed().as_secs_f64();
            let start = Instant::now();
            // The last backward ALSO accumulates its gradient, then updates once.
            // An extra optimizer-bearing step would incorrectly count B+1 grads.
            if cursor + 1 == self.batch
                && let Some(rate) = rate
            {
                self.session.set_adam(rate, 0.9, 0.999, 1e-8);
            }
            self.session.step();
            timing.step_wait += start.elapsed().as_secs_f64();
            let start = Instant::now();
            self.encoder.start();
            {
                let mut transfer = self.encoder.transfer("detached-state-carry");
                transfer.copy_buffer_to_buffer(
                    self.session.output_buffer(3).unwrap(),
                    self.state.at((cursor * self.state_bytes) as u64),
                    self.state_bytes as u64,
                );
                transfer.copy_buffer_to_buffer(
                    self.session.output_buffer(0).unwrap(),
                    self.losses.at((cursor * 4) as u64),
                    4,
                );
            }
            last = Some(self.context.submit(&mut self.encoder));
            timing.carry_wait += start.elapsed().as_secs_f64();
        }
        let start = Instant::now();
        if !self.context.wait_for(last.as_ref().unwrap(), 60_000)? {
            return Err("training GPU timeout".into());
        }
        self.session.wait();
        timing.carry_wait += start.elapsed().as_secs_f64();
        let start = Instant::now();
        let losses =
            unsafe { std::slice::from_raw_parts(self.losses.data().cast::<f32>(), self.batch) };
        if losses.iter().any(|v| !v.is_finite()) {
            return Err("training became non-finite".into());
        }
        let loss = losses.iter().sum::<f32>() / self.batch as f32;
        timing.loss_read = start.elapsed().as_secs_f64();
        Ok((loss, timing))
    }
    /// Diagnostic readback after a completed batch; never changes parameters/state.
    pub fn last_microbatch_losses(&self) -> Vec<f32> {
        // run_batch waits for the final carry submission before returning.
        unsafe { std::slice::from_raw_parts(self.losses.data().cast::<f32>(), self.batch).to_vec() }
    }
    /// Checkpoint/diagnostic readback; never called by the ordinary update loop.
    pub fn read_states(&mut self) -> Vec<f32> {
        self.session.wait();
        unsafe {
            std::slice::from_raw_parts(
                self.state.data().cast::<f32>(),
                self.state_bytes * self.batch / 4,
            )
            .to_vec()
        }
    }
    pub fn restore_states(&mut self, values: &[f32]) -> Result<()> {
        if values.len() != self.state_bytes * self.batch / 4
            || values.iter().any(|v| !v.is_finite())
        {
            return Err("invalid checkpoint cursor states".into());
        }
        self.session.wait();
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.state.data().cast::<f32>(),
                values.len(),
            );
        }
        Ok(())
    }
}
impl Drop for Trainer {
    fn drop(&mut self) {
        self.session.wait();
        self.context.destroy_compute_pipeline(&mut self.targets);
        self.context.destroy_command_encoder(&mut self.encoder);
        for buffer in [self.state, self.upload, self.losses] {
            self.context.destroy_buffer(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn target_packing_shader_validates() {
        let mut module = naga::front::wgsl::parse_str(include_str!("targets.wgsl")).unwrap();
        for (binding, (_, variable)) in module.global_variables.iter_mut().enumerate() {
            variable.binding = Some(naga::ResourceBinding {
                group: 0,
                binding: binding as u32,
            });
        }
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}
