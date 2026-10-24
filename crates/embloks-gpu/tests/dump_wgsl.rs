//! Writes the generated shader sources to target/wgsl (for checking them with other
//! WGSL compilers, e.g. a browser's).

#[test]
fn dump_shader_sources() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/wgsl");
    std::fs::create_dir_all(&dir).unwrap();
    for (name, source) in embloks_gpu::sim::shader_sources() {
        std::fs::write(dir.join(format!("{name}.wgsl")), source).unwrap();
    }
}
