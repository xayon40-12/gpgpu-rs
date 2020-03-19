use gpgpu::Handler;
use gpgpu::descriptors::{BufferDescriptor::*,KernelDescriptor::*};
use gpgpu::Dim;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> gpgpu::Result<()> {
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
    let len = 1<<26;
    let mut gpu = Handler::builder()?
        .add_buffer("src", Len(time as f64,len*4))
        .add_buffer("num", Len(0.0,len*4))
        .add_buffer("sum", Len(0.0,len*4))
        .load_kernel_named("philox2x64_10","noise")
        .load_algorithm("sum")
        .build()?;

    let start = SystemTime::now();
    for _ in 0..10 {
        gpu.run("noise",Dim::D1(len),vec![Buffer("src"),BufArg("num","dst")])?;
        gpu.run_algorithm("sum",Dim::D1(len),vec![BufArg("num","src"),BufArg("sum","dst")])?;
        println!("{}", (gpu.get_first("sum")?/len as f64-0.5)*2.0);
    }
    println!("{}", SystemTime::now().duration_since(start).unwrap().as_millis());

    Ok(())
}
