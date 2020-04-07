use gpgpu::Handler;
use gpgpu::descriptors::{BufferConstructor::*,KernelArg::*,Type::*};
use gpgpu::Dim;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> gpgpu::Result<()> {
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
    let len = 1<<28;
    let n = 1<<5;
    let mut gpu = Handler::builder()?
        .add_buffer("src", Len(U64(time),len))
        .add_buffer("num", Len(F64(0.0),len))
        .load_kernel_named("philox2x64_10_normal","noise")
        .load_kernel_named("philox4x64_10_normal","noise4")
        .load_kernel_named("philox4x32_10_normal","noise32")
        .load_algorithm("sum")
        .build()?;

    gpu.set_arg("noise",&[Buffer("src"),BufArg("num","dst")])?;
    gpu.set_arg("noise4",&[Buffer("src"),BufArg("num","dst")])?;
    gpu.set_arg("noise32",&[Buffer("src"),BufArg("num","dst")])?;

    let mut s;

    println!("Generating 10^{:.2} random numbers and computing the meam:", (((len*n) as f64).ln()/10f64.ln()));

    println!("\nphilox2x64_10_normal");
    let start = SystemTime::now();
    s = 0.0;
    for _ in 0..n {
        gpu.run("noise",Dim::D1(len/2))?;
        gpu.run_algorithm("sum",Dim::D1(len),&[],&["num","num"],None)?;
        s += gpu.get_first::<f64>("num")?/len as f64;
    }
    println!("10^{}", (s/n as f64).abs().ln()/10f64.ln());
    println!("{}", SystemTime::now().duration_since(start).unwrap().as_millis());

    println!("\nphilox4x64_10_normal");
    let start = SystemTime::now();
    s = 0.0;
    for _ in 0..n {
        gpu.run("noise4",Dim::D1(len/4))?;
        gpu.run_algorithm("sum",Dim::D1(len),&[],&["num","num"],None)?;
        s += gpu.get_first::<f64>("num")?/len as f64;
    }
    println!("10^{}", (s/n as f64).abs().ln()/10f64.ln());
    println!("{}", SystemTime::now().duration_since(start).unwrap().as_millis());

    println!("\nphilox4x32_10_normal");
    let start = SystemTime::now();
    s = 0.0;
    for _ in 0..n {
        gpu.run("noise32",Dim::D1(len/2))?;
        gpu.run_algorithm("sum",Dim::D1(len),&[],&["num","num"],None)?;
        s += gpu.get_first::<f64>("num")?/len as f64;
    }
    println!("10^{}", (s/n as f64).abs().ln()/10f64.ln());
    println!("{}", SystemTime::now().duration_since(start).unwrap().as_millis());

    Ok(())
}
