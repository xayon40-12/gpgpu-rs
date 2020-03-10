use gpgpu::Handler;
use gpgpu::descriptors::{BufferDescriptor::*,KernelDescriptor::*};
use gpgpu::kernels::Kernel;
use gpgpu::Dim;

fn main() -> gpgpu::Result<()> {
    let src = "u[x] = p+x_size*1000+x*100;";

    let num = 8;
    let mut gpu = Handler::builder()?
        .add_buffer("u", Data((0..num).map(|i| i as f64).collect()))
        .add_buffer("u2", Len(2.0, num))
        .add_buffer("buf+", Len(0.0, num))
        .add_buffer("buf*", Len(0.0, num))
        .load_kernel("plus")
        .load_kernel("times")
        .create_kernel(Kernel {
            name: "main",
            src,
            args: vec![Buffer("u"),Param("p",10.0)]
        })
        .build()?;

    for i in 0..10 {
        gpu.run("main",Dim::D1(num),vec![Param("p", i as f64)])?;
        gpu.run("plus",Dim::D1(num),vec![BufArg("u","a"),BufArg("u2","b"),BufArg("buf+","dst")])?;
        gpu.run("times",Dim::D1(num),vec![BufArg("u","a"),BufArg("u2","b"),BufArg("buf*","dst")])?;
        println!("main : {:?}", gpu.get("u")?);
        println!("plus : {:?}", gpu.get("buf+")?);
        println!("times: {:?}", gpu.get("buf*")?);
    }

    Ok(())
}
