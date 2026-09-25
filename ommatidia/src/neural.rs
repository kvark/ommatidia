//! Local image pyramid for the recurrent radiance-residual reconstructor.
use meganeura::{Graph, NodeId};
use std::collections::BTreeMap;

pub enum InitKind {
    Zeros,
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
    pub fn initialize(&self, session: &mut meganeura::Session, seed: u64) {
        let mut rng = crate::rng::Rng::new(seed);
        for p in &self.params {
            let values: Vec<_> = match &p.kind {
                InitKind::Zeros => vec![0.0; p.len],
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
        let weight = self.parameter(name, out, input, kernel, zero);
        self.g.conv2d(
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
        )
    }
    fn block(&mut self, x: NodeId, name: &str, shape: [u32; 3]) -> NodeId {
        let a = self.g.silu(x);
        let a = self.conv(a, &format!("{name}.a"), shape, shape[0], 1, false);
        let a = self.g.silu(a);
        let a = self.conv(a, &format!("{name}.b"), shape, shape[0], 1, false);
        let scale = filled(&mut self.g, a, 0.1);
        let a = self.g.mul(a, scale);
        self.g.add(x, a)
    }
    pub(crate) fn encode(
        &mut self,
        input: NodeId,
        adapter: &str,
        input_channels: u32,
        low: [u32; 2],
        c: u32,
    ) -> NodeId {
        let [w, h] = low;
        let stem = self.conv(input, adapter, [input_channels, w, h], c, 1, false);
        let a = self.block(stem, "core.level0", [c, w, h]);
        let b = self.conv(a, "core.down1", [c, w, h], 2 * c, 2, false);
        let b = self.block(b, "core.level1", [2 * c, w / 2, h / 2]);
        let d = self.conv(b, "core.down2", [2 * c, w / 2, h / 2], 4 * c, 2, false);
        let d = self.block(d, "core.level2", [4 * c, w / 4, h / 4]);
        let up = self.g.upsample_2x(d, 1, 4 * c, h / 4, w / 4);
        let up = self.g.concat(up, b, 1, 4 * c, 2 * c, w * h / 4);
        let up = self.conv(up, "core.up1", [6 * c, w / 2, h / 2], 2 * c, 1, false);
        let up = self.g.upsample_2x(up, 1, 2 * c, h / 2, w / 2);
        let up = self.g.concat(up, a, 1, 2 * c, c, w * h);
        let up = self.conv(up, "core.up0", [3 * c, w, h], c, 1, false);
        self.g.silu(up)
    }
    pub(crate) fn new() -> Self {
        Self {
            g: Graph::new(),
            params: Vec::new(),
            shared: BTreeMap::new(),
        }
    }
}

fn filled(g: &mut Graph, x: NodeId, value: f32) -> NodeId {
    let shape = g.node(x).ty.shape.clone();
    g.constant(vec![value; shape.iter().product()], &shape)
}
