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
    let x = 8;
    let y = 3;
    let num = x*y;
    let v = (0..num).map(|i| i as f64).collect::<Vec<_>>();
    let mut gpu = Handler::builder()?
        .add_buffer("src", Data(VecType::F64(v.clone())))
        .add_buffer("dst", Len(F64(0.0), num))
        .load_algorithm("sum")
        .build()?;

    gpu.run_algorithm("sum",Dim::D2(x,y),vec![Buffer("src"),Buffer("dst")])?;
    assert_eq!(gpu.get::<f64>("dst")?.chunks(x).map(|b| b[0]).collect::<Vec<_>>(), v.chunks(x).map(|b| b.iter().fold(0.0,|i,a| i+a)).collect::<Vec<_>>());

    Ok(())
}

#[test]
fn correlation() -> gpgpu::Result<()> {
    let x = 8;
    let y = 3;
    let num = x*y;
    let mut gpu = Handler::builder()?
        .add_buffer("src", Data(VecType::F64((0..num).map(|i| (i%x) as f64).collect())))
        .add_buffer("dst", Len(F64(0.0), num))
        .load_algorithm("correlation")
        .build()?;

    gpu.run_algorithm("correlation",Dim::D1(num),vec![Buffer("src"),Buffer("dst")])?;
    assert_eq!(gpu.get::<f64>("dst")?, (0..num).map(|i| (i%x) as f64 * (x/2) as f64).collect::<Vec<_>>());

    Ok(())
}

#[test]
fn moments() -> gpgpu::Result<()> {
    let x = 8;
    let y = 3;
    let num = x*y;
    let n = 4;
    let mut gpu = Handler::builder()?
        .add_buffer("src", Data(VecType::F64((0..num).map(|i| (i%x) as f64).collect())))
        .add_buffer("tmp", Len(F64(0.0), num))
        .add_buffer("sum", Len(F64(0.0), num))
        .add_buffer("dst", Len(F64(0.0), n*y))
        .load_algorithm("moments")
        .build()?;

    gpu.run_algorithm("moments",Dim::D2(x,y),vec![Buffer("src"),Buffer("tmp"),Buffer("sum"),Buffer("dst"),Param("n",U32(n as u32))])?;
    assert_eq!(gpu.get::<f64>("dst")?, (0..num)
        .map(|i| {
            let i = (i%x) as f64;
            (i,i*i,i*i*i,i*i*i*i)
        })
        .collect::<Vec<_>>().chunks(x)
        .map(|c| c.into_iter().fold([0.0,0.0,0.0,0.0],|[a,b,c,d],(e,f,g,h)| [a+e,b+f,c+g,d+h]).iter().map(|i| i/x as f64).collect::<Vec<f64>>())
        .flatten().collect::<Vec<f64>>()
    );

    Ok(())
}

#[test]
#[should_panic(expected = "Cannot add two algorithms with the same name \"sum\", already added by algorithm \"moments\".")]
fn load_algorithm_already_created() {
    let _gpu = Handler::builder().unwrap()
        .load_algorithm("moments")
        .load_algorithm_named("correlation","sum")
        .build().unwrap();
}

#[test]
#[should_panic(expected = "Cannot add two kernels with the same name \"times\", already added by algorithm \"moments\".")]
fn load_kernel_already_created() {
    let _gpu = Handler::builder().unwrap()
        .load_algorithm("moments")
        .load_kernel_named("divides","times")
        .build().unwrap();
}


#[test]
fn data_file() {
    use gpgpu::data_file::DataFile;

    let mut file = String::new();
    let (x,y,z) = (10,10,10);
    for i in 0..x {
        for j in 0..y {
            for k in 0..z {
                file += &format!("{} {} {} {}\n",i,j,k,i+x*(j+y*k));
            }
        }
    }
    let data = DataFile::from_column(&file);
    for k in 0..z {
        for i in 0..x {
            for j in 0..y {
                assert_eq!((i+x*(j+y*k)) as f64,data.get(&[i as f64,j as f64,k as f64]));
            }
        }
    }
}
