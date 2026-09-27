//! Biased local image pyramid for the recurrent direct-radiance reconstructor.
use meganeura::{Graph, NodeId};
use std::collections::BTreeMap;

pub enum InitKind {
    Zeros,
    Constant(f32),
    Kaiming { fan_in: usize },
}
pub struct ParamInit {
    pub name: String,
    pub len: usize,
    pub kind: InitKind,
}

pub struct Network {
    pub graph: Graph,
    pub params: Vec<ParamInit>,
}
impl Network {
    /// Forward dense-convolution MACs, including every batch/unroll slot and
    /// padded kernel tap. Excludes activations, warps, preparation and backward.
    pub fn macs(&self) -> u64 {
        self.graph
            .nodes()
            .iter()
            .map(|node| match node.op {
                meganeura::graph::Op::Conv2d {
                    in_channels,
                    kernel_h,
                    kernel_w,
                    ..
                } => {
                    node.ty.shape.iter().map(|&v| v as u64).product::<u64>()
                        * u64::from(in_channels)
                        * u64::from(kernel_h)
                        * u64::from(kernel_w)
                }
                _ => 0,
            })
            .sum()
    }

    pub fn initialize(&self, session: &mut meganeura::Session, seed: u64) {
        let mut rng = crate::rng::Rng::new(seed);
        for p in &self.params {
            let values: Vec<_> = match &p.kind {
                InitKind::Zeros => vec![0.0; p.len],
                InitKind::Constant(value) => vec![*value; p.len],
                InitKind::Kaiming { fan_in } => (0..p.len)
                    .map(|_| rng.normal() * (2.0 / *fan_in as f32).sqrt())
                    .collect(),
            };
            session.set_parameter(&p.name, &values);
        }
    }
}
pub(crate) struct Builder {
    pub(crate) g: Graph,
    pub(crate) params: Vec<ParamInit>,
    shared: BTreeMap<String, NodeId>,
}
impl Builder {
    pub(crate) fn parameter(
        &mut self,
        name: &str,
        out: u32,
        input: u32,
        k: u32,
        zero: bool,
    ) -> NodeId {
        if let Some(&id) = self.shared.get(name) {
            return id;
        }
        let len = (out * input * k * k) as usize;
        let id = self.g.parameter(name, &[len]);
        self.params.push(ParamInit {
            name: name.into(),
            len,
            kind: if zero {
                InitKind::Zeros
            } else {
                InitKind::Kaiming {
                    fan_in: (input * k * k) as usize,
                }
            },
        });
        self.shared.insert(name.into(), id);
        id
    }
    pub(crate) fn conv(
        &mut self,
        x: NodeId,
        name: &str,
        shape: [u32; 3],
        out: u32,
        stride: u32,
        zero: bool,
    ) -> NodeId {
        let [input, w, h] = shape;
        let kernel = if zero { 1 } else { 3 };
        let weight = self.parameter(&format!("{name}.weight"), out, input, kernel, zero);
        let result = self.g.conv2d(
            x,
            weight,
            1,
            input,
            h,
            w,
            out,
            kernel,
            kernel,
            stride,
            kernel / 2,
        );
        let bias = self.parameter(&format!("{name}.bias"), out, 1, 1, true);
        self.g
            .add_per_channel(result, bias, out, w.div_ceil(stride) * h.div_ceil(stride))
    }
    pub(crate) fn head(
        &mut self,
        x: NodeId,
        name: &str,
        shape: [u32; 3],
        out: u32,
        bias: f32,
    ) -> NodeId {
        let output = self.conv(x, name, shape, out, 1, true);
        self.params
            .iter_mut()
            .find(|p| p.name == format!("{name}.bias"))
            .unwrap()
            .kind = InitKind::Constant(bias);
        output
    }
    fn block(&mut self, x: NodeId, name: &str, shape: [u32; 3]) -> NodeId {
        let a = self.g.silu(x);
        let a = self.conv(a, &format!("{name}.a"), shape, shape[0], 1, false);
        let a = self.g.silu(a);
        let a = self.conv(a, &format!("{name}.b"), shape, shape[0], 1, false);
        self.g.add(x, a)
    }
    pub(crate) fn encode(
        &mut self,
        input: NodeId,
        adapter: &str,
        input_channels: u32,
        low: [u32; 2],
        c: u32,
        levels: u32,
    ) -> NodeId {
        let [w, h] = low;
        let stem = self.conv(input, adapter, [input_channels, w, h], c, 1, false);
        let mut x = self.block(stem, "core.level0", [c, w, h]);
        let mut skips = vec![x];
        for level in 1..levels {
            let factor = 1 << (level - 1);
            x = self.conv(
                x,
                &format!("core.down{level}"),
                [c * factor, w / factor, h / factor],
                2 * c * factor,
                2,
                false,
            );
            x = self.block(
                x,
                &format!("core.level{level}"),
                [2 * c * factor, w / (2 * factor), h / (2 * factor)],
            );
            skips.push(x);
        }
        for level in (0..levels - 1).rev() {
            let factor = 1 << level;
            x = self
                .g
                .upsample_2x(x, 1, 2 * c * factor, h / (2 * factor), w / (2 * factor));
            x = self.g.concat(
                x,
                skips[level as usize],
                1,
                2 * c * factor,
                c * factor,
                w * h / (factor * factor),
            );
            x = self.conv(
                x,
                &format!("core.up{level}"),
                [3 * c * factor, w / factor, h / factor],
                c * factor,
                1,
                false,
            );
        }
        self.g.silu(x)
    }
    pub(crate) fn new() -> Self {
        Self {
            g: Graph::new(),
            params: Vec::new(),
            shared: BTreeMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macs_use_output_shapes_including_batch_stride_and_padding() {
        let mut graph = Graph::new();
        let input = graph.input("image", &[2 * 3 * 8 * 12]);
        let kernel = graph.parameter("kernel", &[5 * 3 * 3 * 3]);
        let result = graph.conv2d(input, kernel, 2, 3, 8, 12, 5, 3, 3, 2, 1);
        let result = graph.silu(result);
        graph.set_outputs(vec![result]);
        let network = Network {
            graph,
            params: Vec::new(),
        };
        assert_eq!(network.macs(), 2 * 5 * 4 * 6 * 3 * 3 * 3);
    }
}
