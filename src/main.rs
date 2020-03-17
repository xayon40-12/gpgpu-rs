use gpgpu::philox::*;

fn main() {
    for i in 0..100 {
        println!("{:?}", philox4x32([0,0,0,i],[0,0],7));
    }
}
