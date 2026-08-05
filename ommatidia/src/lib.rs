//! Neural frame reconstruction from sparse samples.
//!
//! Ommatidia upscales a cheaply rendered low resolution frame into a high
//! resolution one, running the network through [meganeura] on whatever GPU the
//! host application already has. See `docs/design.md` for the formulation.
//!
//! The pieces:
//!
//! - [`dataset`] is the `.omd` training set container, written by the
//!   generator and read by the trainer.
//! - [`model`] builds the network as a meganeura graph.
//! - [`diffusion`] holds the noise schedule, the training corruption, and the
//!   sampler.
//! - [`runtime`] is the host-facing [`Upscaler`], which shares the caller's
//!   `blade_graphics::Context`.
//!
//! [meganeura]: https://github.com/kvark/meganeura

pub mod batch;
pub mod checkpoint;
pub mod dataset;
pub mod diffusion;
pub mod gpu;
pub mod model;
pub mod rng;
pub mod runtime;
pub mod transform;

pub use dataset::{Layout, Plane, PlaneSet, Sample};
pub use diffusion::{Schedule, timestep_embedding};
pub use model::{Model, ModelConfig, Objective};
pub use runtime::{FrameInputs, Upscaler, UpscalerError};
