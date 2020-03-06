use gpgpu::Handler;

fn main() {
    let src = r#"
        __kernel void main(__global float* buffer, float param) {
            buffer[get_global_id(0)] *= param;
        }
    "#;
    let gpu = Handler::new(src).unwrap();
}
