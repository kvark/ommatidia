//! Recurrent, lobe-separated neural reconstruction from sparse path samples.
//! One direct-radiance recurrent U-Net; output and radiance history are scene-linear.
pub mod dataset;
pub mod gpu;
pub mod metrics;
pub mod neural;
pub mod rng;
pub mod temporal;
pub mod transform;
pub mod transport;
pub use dataset::{InputSource, Layout, Plane, PlaneSet, Sample};
