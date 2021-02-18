use gpgpu::Handler;
use gpgpu::descriptors::{
    BufferConstructor::*,
    KernelArg::*,
    VecTypes::*,
    Types::*,
    KernelConstructor::*,
    ConstructorTypes::*,
};
use gpgpu::kernels::Kernel;
use gpgpu::{Dim,DimDir::*};
use gpgpu::algorithms::{AlgorithmParam::*,MomentsParam,RandomType};
use gpgpu::philox::*;

#[test]
fn simple_main() -> gpgpu::Result<()> {
    let num = 8;
    let mut gpu = Handler::builder()?
        .add_buffer("u", Len(F64(0.0), num))
        .create_kernel(&Kernel {
            name: "_main",
            src: "u[x] = p+x_size*1000+x*100;",
            args: vec![KCBuffer("u",CF64),KCParam("p",CF32)],
            needed: vec![],
        })
    .build()?;

    gpu.set_arg("_main",&[Buffer("u")])?;
    for i in 0..10 {
        gpu.run_arg("_main",Dim::D1(num),&[Param("p",F32( i as f32))])?;
        assert_eq!(gpu.get("u")?.VF64(), (0..num).map(|j| i as f64 + num as f64*1000.0 + j as f64*100.0).collect::<Vec<_>>());
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

    gpu.run_arg("plus",Dim::D1(num),&[Buffer("a"),Buffer("b"),Buffer("dst")])?;
    assert_eq!(gpu.get("dst")?.VF64(), (0..num).map(|i| i as f64 + 2.0).collect::<Vec<_>>());

    Ok(())
}

#[test]
fn times() -> gpgpu::Result<()> {
    let num = 8;
    let mut gpu = Handler::builder()?
        .add_buffer("a", Data(VF64((0..num).map(|i| i as f64).collect())))
        .add_buffer("b", Len(F64(2.0), num))
        .add_buffer("dst", Len(F64(0.0), num))
        .load_kernel("times")
        .build()?;

    gpu.run_arg("times",Dim::D1(num),&[Buffer("a"),Buffer("b"),Buffer("dst")])?;
    assert_eq!(gpu.get("dst")?.VF64(), (0..num).map(|i| i as f64 * 2.0).collect::<Vec<_>>());

    Ok(())
}

#[test]
fn sum() -> gpgpu::Result<()> {
    let x = 97;
    let y = 107;
    let num = x*y;
    let v = (0..num).map(|i| i as f64).collect::<Vec<_>>();
    let mut gpu = Handler::builder()?
        .add_buffer("src", Data(VF64(v.clone())))
        .add_buffer("tmp", Len(F64(0.0), num))
        .add_buffer("dstx", Len(F64(0.0), y))
        .add_buffer("dsty", Len(F64(0.0), x))
        .add_buffer("dstxy", Len(F64(0.0), 1))
        .load_algorithm("sum")
        .build()?;

    gpu.run_algorithm("sum",Dim::D2(x,y),&[X,Y],&["src","tmp","dstxy"],Nothing)?;
    assert_eq!(gpu.get_first("dstxy")?.F64(), v.iter().fold(0.0,|a,i| i+a), "dim XY");

    gpu.run_algorithm("sum",Dim::D2(x,y),&[X],&["src","tmp","dstx"],Nothing)?;
    gpu.get_first("dstx")?.F64();
    assert_eq!(gpu.get("dstx")?.VF64(), v.chunks(x).map(|b| b.iter().fold(0.0,|a,i| i+a)).collect::<Vec<_>>(), "dim X");
    gpu.run_algorithm("sum",Dim::D2(x,y),&[Y],&["src","tmp","dsty"],Nothing)?;
    assert_eq!(gpu.get("dsty")?.VF64(), v.chunks(x).fold(vec![0f64;x],|a,c| a.iter().enumerate().map(|(i,v)| v+c[i]).collect::<Vec<_>>()), "dim Y");



    Ok(())
}

/* #[test]
fn window_sum() -> gpgpu::Result<()> {
    //TODO
    Ok(())
} */

#[test]
fn min() -> gpgpu::Result<()> {
    let x = 8;
    let y = 3;
    let num = x*y;
    let v = (0..num).map(|i| i as f64).collect::<Vec<_>>();
    let mut gpu = Handler::builder()?
        .add_buffer("src", Data(VF64(v.clone())))
        .add_buffer("tmp", Len(F64(0.0), num))
        .add_buffer("dst", Len(F64(0.0), y))
        .load_algorithm("min")
        .build()?;

    gpu.run_algorithm("min",Dim::D2(x,y),&[X],&["src","tmp","dst"],Nothing)?;
    assert_eq!(gpu.get("dst")?.VF64(), v.chunks(x).map(|b| b.iter().fold(std::f64::MAX,|a,&i| if i<a { i } else { a })).collect::<Vec<_>>());

    Ok(())
}

#[test]
fn max() -> gpgpu::Result<()> {
    let x = 8;
    let y = 3;
    let num = x*y;
    let v = (0..num).map(|i| i as f64).collect::<Vec<_>>();
    let mut gpu = Handler::builder()?
        .add_buffer("src", Data(VF64(v.clone())))
        .add_buffer("tmp", Len(F64(0.0), num))
        .add_buffer("dst", Len(F64(0.0), y))
        .load_algorithm("max")
        .build()?;

    gpu.run_algorithm("max",Dim::D2(x,y),&[X],&["src","tmp","dst"],Nothing)?;
    assert_eq!(gpu.get("dst")?.VF64(), v.chunks(x).map(|b| b.iter().fold(std::f64::MIN,|a,&i| if i>a { i } else { a })).collect::<Vec<_>>());

    Ok(())
}

#[test]
fn correlation() -> gpgpu::Result<()> {
    let x = 8;
    let y = 3;
    let num = x*y;
    let mut gpu = Handler::builder()?
        .add_buffer("src", Data(VF64((0..num).map(|i| i as f64).collect())))
        .add_buffer("dst", Len(F64(0.0), num))
        .load_algorithm("correlation")
        .build()?;

    gpu.run_algorithm("correlation",Dim::D2(x,y),&[X],&["src","dst"],Nothing)?;
    assert_eq!(gpu.get("dst")?.VF64(), (0..num).map(|i| i as f64 * ((i/x)*x+x/2) as f64).collect::<Vec<_>>(),"dim X");
    gpu.run_algorithm("correlation",Dim::D2(x,y),&[Y],&["src","dst"],Nothing)?;
    assert_eq!(gpu.get("dst")?.VF64(), (0..num).map(|i| i as f64 * ((i%x)+x*(y/2)) as f64).collect::<Vec<_>>(),"dim Y");
    gpu.run_algorithm("correlation",Dim::D2(x,y),&[X,Y],&["src","dst"],Nothing)?;
    assert_eq!(gpu.get("dst")?.VF64(), (0..num).map(|i| i as f64 * (x*y/2) as f64).collect::<Vec<_>>(),"dim XY");

    Ok(())
}

fn ft(tab: &[[f64;2]]) -> Vec<[f64;2]> {
    #[allow(non_snake_case)]
    let N = tab.len();
    (0..N).map(|k|
        tab.iter().enumerate().fold([0.0,0.0].into(), |a: [f64;2],(i,v)| {
            let x = -2.0*std::f64::consts::PI*k as f64*i as f64/N as f64;
            let c = f64::cos(x);
            let s = f64::sin(x);
            [a[0]+c*v[0]-s*v[1],a[1]+s*v[0]+c*v[1]].into()
        })
    ).collect()
}

fn cut(x: f64) -> f64 {
    let pow = 10.0f64.powf(10.0);
    ((x*pow) as u64) as f64/pow
}

fn cut2(x: [f64;2]) -> [f64;2] {
    [cut(x[0]),cut(x[1])].into()
}

#[test]
fn fft() -> gpgpu::Result<()> {
    let x = 1<<2;
    let y = 1<<3;
    let num = x*y;
    let mut gpu = Handler::builder()?
        .add_buffer("src", Data(VF64_2((0..num).map(|i| [i as f64,0.0].into()).collect())))
        .add_buffer("tmp", Len(F64_2([0.0,0.0].into()), num))
        .add_buffer("dst", Len(F64_2([0.0,0.0].into()), num))
        .load_algorithm("FFT")
        .build()?;

    gpu.run_algorithm("FFT",Dim::D2(x,y),&[X],&["src","tmp","dst"],Nothing)?;
    assert_eq!(gpu.get("dst")?.VF64_2().iter().map(|&d| cut2(d)).collect::<Vec<_>>()
        , (0..num).map(|i| [i as f64,0.0].into()).collect::<Vec<[f64;2]>>()
        .chunks(x).flat_map(|c| ft(c)).map(|d| cut2([d[0]/x as f64,d[1]/x as f64].into())).collect::<Vec<_>>(),"dim X");

    gpu.run_algorithm("FFT",Dim::D2(x,y),&[Y],&["src","tmp","dst"],Nothing)?;
    assert_eq!(gpu.get("dst")?.VF64_2().iter().map(|&d| cut2(d)).collect::<Vec<_>>(),
        {let tmp = (0..num).map(|i| [((i%y)*x+i/y) as f64,0.0].into()).collect::<Vec<[f64;2]>>()
        .chunks(y).flat_map(|c| ft(c)).map(|d| [d[0]/y as f64,d[1]/y as f64].into()).collect::<Vec<[f64;2]>>();
        (0..num).map(|i| cut2(tmp[(i*y)%(x*y)+i/x])).collect::<Vec<[f64;2]>>()
        },"dim Y");

    gpu.run_algorithm("FFT",Dim::D2(x,y),&[Y,X],&["src","tmp","dst"],Nothing)?;
    let para = gpu.get("dst")?.VF64_2().iter().map(|&d| cut2(d)).collect::<Vec<_>>();
    let local = {
            let tmpp = (0..num).map(|i| [i as f64,0.0].into()).collect::<Vec<[f64;2]>>()
            .chunks(x).flat_map(|c| ft(c)).map(|d| [d[0]/x as f64,d[1]/x as f64].into()).collect::<Vec<[f64;2]>>();
            let tmp = (0..num).map(|i| tmpp[((i%y)*x+i/y)]).collect::<Vec<[f64;2]>>()
            .chunks(y).flat_map(|c| ft(c)).map(|d| [d[0]/y as f64,d[1]/y as f64].into()).collect::<Vec<[f64;2]>>();
            (0..num).map(|i| cut2(tmp[(i*y)%(x*y)+i/x])).collect::<Vec<[f64;2]>>()
        };
    assert_eq!(para,local,"dim XY");

    Ok(())
}

#[test]
fn moments() -> gpgpu::Result<()> {
    let x = 8;
    let y = 3;
    let z = 4;
    let num = x*y*z;
    let n = 4;
    let mut gpu = Handler::builder()?
        .add_buffer("src", Data(VF64((0..num).map(|i| (i%x) as f64).collect())))
        .add_buffer("tmp", Len(F64(0.0), num))
        .add_buffer("sum", Len(F64(0.0), num))
        .add_buffer("dstx", Len(F64(0.0), n*y*z))
        .add_buffer("dsty", Len(F64(0.0), n*x*z))
        .add_buffer("dstxy", Len(F64(0.0), n*z))
        .add_buffer("dstxyz", Len(F64(0.0), n))
        .load_algorithm("moments")
        .build()?;

    let pow = (0..num)
        .map(|i| {
            let i = (i%x) as f64;
            (i,i*i,i*i*i,i*i*i*i)
        })
        .collect::<Vec<_>>();

    let prm = MomentsParam{ num: n as u32, vect_dim: 1, packed: true };

    gpu.run_algorithm("moments",Dim::D3(x,y,z),&[X],&["src","tmp","sum","dstx"],Ref(&prm))?;
    assert_eq!(gpu.get("dstx")?.VF64(), pow
        .chunks(x)
        .map(|c| c.into_iter().fold([0.0,0.0,0.0,0.0],|[a,b,c,d],(e,f,g,h)| [a+e,b+f,c+g,d+h]).iter().map(|i| i/x as f64).collect::<Vec<f64>>())
        .flatten().collect::<Vec<f64>>()
    ,"x");
    gpu.run_algorithm("moments",Dim::D3(x,y,z),&[Y],&["src","tmp","sum","dsty"],Ref(&prm))?;
    assert_eq!(gpu.get("dsty")?.VF64(), pow
        .chunks(x*z)
        .fold(vec![(0.0,0.0,0.0,0.0);x*z],|a,c| a.iter().enumerate().map(|(i,v)| (v.0+c[i].0,v.1+c[i].1,v.2+c[i].2,v.3+c[i].3)).collect())
        .iter().map(|&(a,b,c,d)| vec![a,b,c,d]).flatten().map(|i| i/y as f64).collect::<Vec<_>>()
    ,"y");
    gpu.run_algorithm("moments",Dim::D3(x,y,z),&[X,Y,Z],&["src","tmp","sum","dstxyz"],Ref(&prm))?;
    assert_eq!(gpu.get("dstxyz")?.VF64(), pow.iter()
        .fold([0.0,0.0,0.0,0.0],|a,c| [a[0]+c.0,a[1]+c.1,a[2]+c.2,a[3]+c.3])
        .iter().map(|i| i/(x*y*z) as f64).collect::<Vec<_>>()
    ,"xyz");

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
fn data_file() -> gpgpu::Result<()> {
    use gpgpu::data_file::{DataFile,Format};

    let mut file = String::new();
    let l = 1<<5;
    let (x,y,z) = (l,l,l);
    for i in 0..x {
        for k in 0..z {
            for j in 0..y {
                file += &format!("{} {} {} {}\n",i,j,k,i+x*(j+y*k));
            }
        }
    }
    let data = DataFile::parse(Format::Column(&file));
    let mut gpu = Handler::builder()?
        .load_data("data",Format::Column(&file),false,Some("databuf"))
        .add_buffer("u", Len(F64(0.0),x*y*z))
        .create_kernel(&Kernel {
            name: "kern",
            args: vec![KCBuffer("u",CF64),KCBuffer("databuf",CF64)],
            src: "
                u[x+x_size*(y+y_size*z)] = data(x,y,z,databuf);
            ",
            needed: vec![],
        })
    .build()?;

    gpu.run_arg("kern",Dim::D3(x,y,z),&[Buffer("u"),Buffer("databuf")])?;

    let gpudata = gpu.get("u")?.VF64();
    for i in 0..x {
        for j in 0..y {
            for k in 0..z {
                let id = i+x*(j+y*k);
                assert_eq!(id as f64,data.get(&[i as f64,j as f64,k as f64]));
                assert_eq!(id as f64,gpudata[id]);
            }
        }
    }

    Ok(())
}

#[test]
fn data_file_interpolated() -> gpgpu::Result<()> {
    use gpgpu::data_file::{DataFile,Format};

    let mut file = String::new();
    let (x,y,z) = (3,5,107);
    for i in 0..=x {
        for j in 0..=y {
            for k in 0..=z {
                file += &format!("{} {} {} {}\n",i,j,k,i+j+k);
            }
        }
    }

    let data = DataFile::parse(Format::Column(&file));

    let t = 3;
    let (x,y,z) = (x*t,y*t,z*t);
    let mut gpu = Handler::builder()?
        .load_data("data",Format::Column(&file),true,None)
        .add_buffer("u", Len(F64(0.0),x*y*z))
        .create_kernel(&Kernel {
            name: "kern",
            args: vec![KCBuffer("u",CF64)],
            src: &format!("
                double t = {};
                u[x+x_size*(y+y_size*z)] = data(x/t,y/t,z/t);
            ",t),
            needed: vec![],
        })
    .build()?;

    gpu.run_arg("kern",Dim::D3(x,y,z),&[Buffer("u")])?;

    let gpudata = gpu.get("u")?.VF64();

    for i in 0..x {
        for j in 0..y {
            for k in 0..z {
                let l = i as f64/t as f64;
                let m = j as f64/t as f64;
                let n = k as f64/t as f64;
                let id = l+m+n;
                assert_eq!(
                    format!("{:.10}",id) ,
                    format!("{:.10}",data.get_interpolated(&[l,m,n]))
                );
                assert_eq!(
                    format!("{:.10}",id) ,
                    format!("{:.10}",gpudata[i+x*(j+y*k)])
                );
            }
        }
        println!("");
    }

    Ok(())
}

#[test]
fn function_test() -> gpgpu::Result<()> {
    let num = 8;
    let mut gpu = Handler::builder()?
        .add_buffer("u", Data(VF64((0..num).map(|i| i as f64).collect())))
        .load_function("swap")
        .create_kernel(&Kernel {
            name: "_main",
            src: "swap(&u[x*2],&u[x*2+1]);",
            args: vec![KCBuffer("u",CF64)],
            needed: vec![],
        })
    .build()?;

    gpu.run_arg("_main",Dim::D1(num/2),&[Buffer("u")])?;
    assert_eq!(gpu.get("u")?.VF64(), (0..num).map(|j| (j + (j+1)%2 -j%2) as f64 ).collect::<Vec<_>>());

    Ok(())
}

#[test]
fn random_philoxrs_gpualgorithm() -> gpgpu::Result<()> {
    let seed = 0u64;
    let len = 4;
    let mut gpu = Handler::builder()?
        .add_buffer("src", Len(seed.into(),len))
        .add_buffer("src1", Len(seed.into(),len))
        .add_buffer("src2", Len(seed.into(),len))
        .add_buffer("num", Len(0.0.into(),len))
        .add_buffer("num1", Len(0.0.into(),len))
        .add_buffer("num2", Len(0.0.into(),len))
        .load_algorithm("philox2x64_10")
        .load_algorithm("philox4x64_10")
        .load_algorithm("philox4x32_10")
        .build()?;

    let rtype = RandomType::Uniform;


    gpu.run_algorithm("philox2x64_10",Dim::D1(len),&[],&["src","num"],Ref(&rtype))?;
    let gres = gpu.get("num")?.VF64();
    let counter = [seed;2];
    let res0 = philox2x64(counter, 0, 10);
    let res1 = philox2x64(counter, 1, 10);
    assert_eq!(gres,[res0[0],res0[1],res1[0],res1[1]],"philox2x64_10");
    
    gpu.run_algorithm("philox4x64_10",Dim::D1(len),&[],&["src1","num1"],Ref(&rtype))?;
    let gres = gpu.get("num1")?.VF64();
    let counter = [seed;4];
    let res = philox4x64(counter, [0,0], 10);
    assert_eq!(gres,res,"philox4x64_10");

    gpu.run_algorithm("philox4x32_10",Dim::D1(len),&[],&["src2","num2"],Ref(&rtype))?;
    let gres = gpu.get("num2")?.VF64();
    let counter = unsafe { std::mem::transmute([seed;2]) };
    let res0 = philox4x32(counter, [0,0], 10);
    let res1 = philox4x32(counter, [0,1], 10);
    assert_eq!(gres,[res0[0],res0[1],res1[0],res1[1]],"philox4x32_10");

    Ok(())
}
