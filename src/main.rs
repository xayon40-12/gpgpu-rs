use gpgpu::Handler;
use gpgpu::descriptors::{BufferConstructor::*,KernelArg::*,Type::*};
use gpgpu::Dim;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> gpgpu::Result<()> {
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
    let len = 1<<26;
    let mut gpu = Handler::builder()?
        .add_buffer("src", Len(U64(time),len*4))
        .add_buffer("num", Len(F64(0.0),len*4))
        .add_buffer("sum", Len(F64(0.0),len*4))
        .load_kernel_named("philox4x64_10","noise")
        .load_kernel_named("philox4x32_10","noise32")
        .load_algorithm("sum")
        .build()?;

    gpu.set_arg("noise",vec![Buffer("src"),BufArg("num","dst")])?;
    gpu.set_arg("noise32",vec![Buffer("src"),BufArg("num","dst")])?;

    println!("\nphilox4x64_10");
    let start = SystemTime::now();
    for _ in 0..10 {
        gpu.run("noise",Dim::D1(len))?;
        gpu.run_algorithm("sum",Dim::D1(len),vec![BufArg("num","src"),BufArg("sum","dst")])?;
        println!("{}", (gpu.get_first::<f64>("sum")?/len as f64-0.5)*2.0);
    }
    println!("{}", SystemTime::now().duration_since(start).unwrap().as_millis());

    println!("\nphilox4x32_10");
    let start = SystemTime::now();
    for _ in 0..10 {
        gpu.run("noise32",Dim::D1(len))?;
        gpu.run_algorithm("sum",Dim::D1(len),vec![BufArg("num","src"),BufArg("sum","dst")])?;
        println!("{}", (gpu.get_first::<f64>("sum")?/len as f64-0.5)*2.0);
    }
    println!("{}", SystemTime::now().duration_since(start).unwrap().as_millis());

    Ok(())
}
