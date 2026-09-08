//! Exercise the field's tied multiview core above the Winograd channel threshold.
use ommatidia::{
    field::{Config, data::RenderShape, graph},
    gpu,
};

#[test]
#[ignore = "requires Vulkan or Metal"]
fn wider_field_training_weights_reload_in_inference() {
    let config = Config {
        extent: [8, 8],
        views: 2,
        channels: 16,
        hidden: 32,
        ..Config::default()
    };
    let shape = RenderShape {
        rays: 2,
        steps: 4,
        probes: 0,
    };
    let context = gpu::create_context(None, false);
    let training = graph::build_render(&config, shape, true).unwrap();
    let mut source = gpu::training_session(&training.graph, context.clone());
    training.initialize(&mut source, 7);
    let path = std::env::temp_dir().join(format!("wide-field-{}.safetensors", std::process::id()));
    source.save_checkpoint(&path).unwrap();
    let inference = graph::build_diagnostics(&config, shape).unwrap();
    let mut target = gpu::inference_session(&inference.graph, context);
    target.load_checkpoint(&path).unwrap();
    for parameter in &training.params {
        let mut a = vec![0.0; parameter.len];
        let mut b = a.clone();
        source.read_param(&parameter.name, &mut a);
        target.read_param(&parameter.name, &mut b);
        assert_eq!(a, b, "{}", parameter.name);
    }
    std::fs::remove_file(path).unwrap();
}
