use gpgpu::Handler;
use gpgpu::descriptors::{BufferConstructor::*};
use gpgpu::{Dim,DimDir::*};
use std::time::{SystemTime, UNIX_EPOCH};
use gpgpu::algorithms::{moments_to_cumulants,AlgorithmParam::*,MomentsParam,RandomType};

fn main() -> gpgpu::Result<()> {
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
    let len = 1<<17;
    let n = 1<<14;
    let nmom = 4;
    let mut gpu = Handler::builder()?
        .add_buffer("src", Len(time.into(),len))
        .add_buffer("num", Len(0.0.into(),len))
        .add_buffer("tmp", Len(0.0.into(),len))
        .add_buffer("sum", Len(0.0.into(),len))
        .add_buffer("dst", Len(0.0.into(),nmom))
        .load_algorithm_named("philox2x64_10","noise")
        .load_algorithm_named("philox4x64_10","noise4")
        .load_algorithm_named("philox4x32_10","noise32")
        .load_algorithm("moments")
        .build()?;

    println!("Generating 10^{:.2} random numbers and computing the meam:", (((len*n) as f64).ln()/10f64.ln()));

    let prm = MomentsParam{ num: nmom as u32, vect_dim: 1, packed: true };
    let rtype = RandomType::Normal;

    println!("\nphilox2x64_10_normal");
    let start = SystemTime::now();
    let mut moms = vec![0.0;nmom];
    for _ in 0..n {
        gpu.run_algorithm("noise",Dim::D1(len/2),&[],&["src","num"],Ref(&rtype))?;
        gpu.run_algorithm("moments",Dim::D1(len),&[X],&["num","tmp","sum","dst"],Ref(&prm))?;
        moms = gpu.get("dst")?.VF64().iter().enumerate().map(|(i,v)| moms[i]+v).collect();
    }
    moms = moms.iter().map(|v| v/n as f64).collect();
    println!("{:?}", moments_to_cumulants(&moms,1));
    println!("{}", SystemTime::now().duration_since(start).unwrap().as_millis());

    println!("\nphilox4x64_10_normal");
    let start = SystemTime::now();
    moms = vec![0.0;nmom];
    for _ in 0..n {
        gpu.run_algorithm("noise4",Dim::D1(len/2),&[],&["src","num"],Ref(&rtype))?;
        gpu.run_algorithm("moments",Dim::D1(len),&[X],&["num","tmp","sum","dst"],Ref(&prm))?;
        moms = gpu.get("dst")?.VF64().iter().enumerate().map(|(i,v)| moms[i]+v).collect();
    }
    moms = moms.iter().map(|v| v/n as f64).collect();
    println!("{:?}", moments_to_cumulants(&moms,1));
    println!("{}", SystemTime::now().duration_since(start).unwrap().as_millis());

    println!("\nphilox4x32_10_normal");
    let start = SystemTime::now();
    moms = vec![0.0;nmom];
    for _ in 0..n {
        gpu.run_algorithm("noise32",Dim::D1(len/2),&[],&["src","num"],Ref(&rtype))?;
        gpu.run_algorithm("moments",Dim::D1(len),&[X],&["num","tmp","sum","dst"],Ref(&prm))?;
        moms = gpu.get("dst")?.VF64().iter().enumerate().map(|(i,v)| moms[i]+v).collect();
    }
    moms = moms.iter().map(|v| v/n as f64).collect();
    println!("{:?}", moments_to_cumulants(&moms,1));
    println!("{}", SystemTime::now().duration_since(start).unwrap().as_millis());

    Ok(())
}
