//! Recurrent, lobe-separated neural reconstruction from sparse path samples.
//! One radiance-residual U-Net; GPU preparation and history stay scene-linear.
pub mod dataset;
pub mod gpu;
pub mod metrics;
pub mod neural;
pub mod rng;
pub mod temporal;
pub mod transform;
pub mod transport;
pub use dataset::{InputSource, Layout, Plane, PlaneSet, Sample};
