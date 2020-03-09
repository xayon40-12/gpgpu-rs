use gpgpu::Handler;
use gpgpu::descriptors::{BufferDescriptor,KernelDescriptor::*};
use gpgpu::Dim;

fn main() -> gpgpu::Result<()> {
    let src = "u[x] += p+x_size;";

    let num = 100;
    let mut gpu = Handler::builder(src)?
        .add_buffer("u",  BufferDescriptor::Data(vec![0.0;num]))
        .add_kernel("main", vec![Buffer("u"),Param("p",10.0)])
        .build()?;


    for _ in 0..10 {
        gpu.run("main",Dim::D1(num))?;
        let array = gpu.get("u")?;
        println!("{}", array[1]);
    }

    Ok(())
}
