use gpgpu::Handler;
use gpgpu::descriptors::{BufferConstructor::*,KernelArg::*,VecType,Type::*};
use gpgpu::kernels::Kernel;
use gpgpu::Dim;
use gpgpu::descriptors::KernelConstructor as KC;
use gpgpu::descriptors::EmptyType as EmT;

#[test]
fn simple_main() -> gpgpu::Result<()> {
    let src = "u[x] = p+x_size*1000+x*100;".to_string();

    let num = 8;
    let param_name = String::from("p");
    let mut gpu = Handler::builder()?
        .add_buffer("u", Len(F64(0.0), num))
        .create_kernel(Kernel {
            name: "_main",
            src: &src,
            args: vec![KC::Buffer("u",EmT::F64),KC::Param(&param_name,EmT::F32)]
        })
    .build()?;

    gpu.set_arg("_main",vec![Buffer("u")])?;
    for i in 0..10 {
        gpu.run_arg("_main",Dim::D1(num),vec![Param("p",F32( i as f32))])?;
        assert_eq!(gpu.get::<f64>("u")?, (0..num).map(|j| i as f64 + num as f64*1000.0 + j as f64*100.0).collect::<Vec<_>>());
    }

    Ok(())
}

#[test]
fn plus() -> gpgpu::Result<()> {
    let num = 8;
    let mut gpu = Handler::builder()?
        .add_buffer("a", Data((0..num).map(|i| i as f64).collect::<Vec<_>>().into()))
        .add_buffer("b", Len(F64(2.0), num))
        .add_buffer("dst", Len(F64(0.0), num))
        .load_kernel("plus")
        .build()?;

    gpu.run_arg("plus",Dim::D1(num),vec![Buffer("a"),Buffer("b"),Buffer("dst")])?;
    assert_eq!(gpu.get::<f64>("dst")?, (0..num).map(|i| i as f64 + 2.0).collect::<Vec<_>>());

    Ok(())
}

#[test]
fn times() -> gpgpu::Result<()> {
    let num = 8;
    let mut gpu = Handler::builder()?
        .add_buffer("a", Data(VecType::F64((0..num).map(|i| i as f64).collect())))
        .add_buffer("b", Len(F64(2.0), num))
        .add_buffer("dst", Len(F64(0.0), num))
        .load_kernel("times")
        .build()?;

    gpu.run_arg("times",Dim::D1(num),vec![Buffer("a"),Buffer("b"),Buffer("dst")])?;
    assert_eq!(gpu.get::<f64>("dst")?, (0..num).map(|i| i as f64 * 2.0).collect::<Vec<_>>());

    Ok(())
}

#[test]
fn sum() -> gpgpu::Result<()> {
    let num = 8;
    let mut gpu = Handler::builder()?
        .add_buffer("src", Data(VecType::F64((0..num).map(|i| i as f64).collect())))
        .add_buffer("dst", Len(F64(0.0), num))
        .load_algorithm("sum")
        .build()?;

    gpu.run_algorithm("sum",Dim::D1(num),vec![Buffer("src"),Buffer("dst")])?;
    assert_eq!(gpu.get::<f64>("dst")?[0], (0..num).map(|i| i as f64).fold(0.0,|i,a| i+a));

    Ok(())
}

