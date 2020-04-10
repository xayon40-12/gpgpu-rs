use gpgpu::Handler;
use gpgpu::descriptors::{BufferConstructor::*,KernelArg::*,VecType,Type::*};
use gpgpu::kernels::Kernel;
use gpgpu::{Dim,DimDir::*};
use gpgpu::descriptors::KernelConstructor as KC;
use gpgpu::descriptors::EmptyType as EmT;

use ocl::prm::*;

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
            args: vec![KC::Buffer("u",EmT::F64),KC::Param(&param_name,EmT::F32)],
            needed: vec![],
        })
    .build()?;

    gpu.set_arg("_main",&[Buffer("u")])?;
    for i in 0..10 {
        gpu.run_arg("_main",Dim::D1(num),&[Param("p",F32( i as f32))])?;
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

    gpu.run_arg("plus",Dim::D1(num),&[Buffer("a"),Buffer("b"),Buffer("dst")])?;
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

    gpu.run_arg("times",Dim::D1(num),&[Buffer("a"),Buffer("b"),Buffer("dst")])?;
    assert_eq!(gpu.get::<f64>("dst")?, (0..num).map(|i| i as f64 * 2.0).collect::<Vec<_>>());

    Ok(())
}

#[test]
fn sum() -> gpgpu::Result<()> {
    let x = 97;
    let y = 107;
    let num = x*y;
    let v = (0..num).map(|i| i as f64).collect::<Vec<_>>();
    let mut gpu = Handler::builder()?
        .add_buffer("src", Data(VecType::F64(v.clone())))
        .add_buffer("tmp", Len(F64(0.0), num))
        .add_buffer("dstx", Len(F64(0.0), y))
        .add_buffer("dsty", Len(F64(0.0), x))
        .add_buffer("dstxy", Len(F64(0.0), 1))
        .load_algorithm("sum")
        .build()?;

    gpu.run_algorithm("sum",Dim::D2(x,y),&[X,Y],&["src","tmp","dstxy"],None)?;
    assert_eq!(gpu.get_first::<f64>("dstxy")?, v.iter().fold(0.0,|a,i| i+a), "dim XY");

    gpu.run_algorithm("sum",Dim::D2(x,y),&[X],&["src","tmp","dstx"],None)?;
    gpu.get_first::<f64>("dstx")?;
    assert_eq!(gpu.get::<f64>("dstx")?, v.chunks(x).map(|b| b.iter().fold(0.0,|a,i| i+a)).collect::<Vec<_>>(), "dim X");
    gpu.run_algorithm("sum",Dim::D2(x,y),&[Y],&["src","tmp","dsty"],None)?;
    assert_eq!(gpu.get::<f64>("dsty")?, v.chunks(x).fold(vec![0f64;x],|a,c| a.iter().enumerate().map(|(i,v)| v+c[i]).collect::<Vec<_>>()), "dim Y");



    Ok(())
}

#[test]
fn min() -> gpgpu::Result<()> {
    let x = 8;
    let y = 3;
    let num = x*y;
    let v = (0..num).map(|i| i as f64).collect::<Vec<_>>();
    let mut gpu = Handler::builder()?
        .add_buffer("src", Data(VecType::F64(v.clone())))
        .add_buffer("tmp", Len(F64(0.0), num))
        .add_buffer("dst", Len(F64(0.0), y))
        .load_algorithm("min")
        .build()?;

    gpu.run_algorithm("min",Dim::D2(x,y),&[X],&["src","tmp","dst"],None)?;
    assert_eq!(gpu.get::<f64>("dst")?, v.chunks(x).map(|b| b.iter().fold(std::f64::MAX,|a,&i| if i<a { i } else { a })).collect::<Vec<_>>());

    Ok(())
}

#[test]
fn max() -> gpgpu::Result<()> {
    let x = 8;
    let y = 3;
    let num = x*y;
    let v = (0..num).map(|i| i as f64).collect::<Vec<_>>();
    let mut gpu = Handler::builder()?
        .add_buffer("src", Data(VecType::F64(v.clone())))
        .add_buffer("tmp", Len(F64(0.0), num))
        .add_buffer("dst", Len(F64(0.0), y))
        .load_algorithm("max")
        .build()?;

    gpu.run_algorithm("max",Dim::D2(x,y),&[X],&["src","tmp","dst"],None)?;
    assert_eq!(gpu.get::<f64>("dst")?, v.chunks(x).map(|b| b.iter().fold(std::f64::MIN,|a,&i| if i>a { i } else { a })).collect::<Vec<_>>());

    Ok(())
}

#[test]
fn correlation() -> gpgpu::Result<()> {
    let x = 8;
    let y = 3;
    let num = x*y;
    let mut gpu = Handler::builder()?
        .add_buffer("src", Data(VecType::F64((0..num).map(|i| i as f64).collect())))
        .add_buffer("dst", Len(F64(0.0), num))
        .load_algorithm("correlation")
        .build()?;

    gpu.run_algorithm("correlation",Dim::D2(x,y),&[X],&["src","dst"],None)?;
    assert_eq!(gpu.get::<f64>("dst")?, (0..num).map(|i| i as f64 * ((i/x)*x+x/2) as f64).collect::<Vec<_>>(),"dim X");
    gpu.run_algorithm("correlation",Dim::D2(x,y),&[Y],&["src","dst"],None)?;
    assert_eq!(gpu.get::<f64>("dst")?, (0..num).map(|i| i as f64 * ((i%x)+x*(y/2)) as f64).collect::<Vec<_>>(),"dim Y");
    gpu.run_algorithm("correlation",Dim::D2(x,y),&[X,Y],&["src","dst"],None)?;
    assert_eq!(gpu.get::<f64>("dst")?, (0..num).map(|i| i as f64 * (x*y/2) as f64).collect::<Vec<_>>(),"dim XY");

    Ok(())
}

fn ft(tab: &[Double2]) -> Vec<Double2> {
    #[allow(non_snake_case)]
    let N = tab.len();
    (0..N).map(|k|
        tab.iter().enumerate().fold([0.0,0.0].into(), |a: Double2,(i,v)| {
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

fn cut2(x: Double2) -> Double2 {
    [cut(x[0]),cut(x[1])].into()
}

#[test]
fn fft() -> gpgpu::Result<()> {
    let x = 1<<2;
    let y = 1<<3;
    let num = x*y;
    let mut gpu = Handler::builder()?
        .add_buffer("src", Data(VecType::F64_2((0..num).map(|i| [i as f64,0.0].into()).collect())))
        .add_buffer("tmp", Len(F64_2([0.0,0.0].into()), num))
        .add_buffer("dst", Len(F64_2([0.0,0.0].into()), num))
        .load_algorithm("FFT")
        .build()?;

    gpu.run_algorithm("FFT",Dim::D2(x,y),&[X],&["src","tmp","dst"],None)?;
    assert_eq!(gpu.get::<Double2>("dst")?.iter().map(|&d| cut2(d)).collect::<Vec<_>>()
        , (0..num).map(|i| [i as f64,0.0].into()).collect::<Vec<Double2>>()
        .chunks(x).flat_map(|c| ft(c)).map(|d| cut2([d[0]/x as f64,d[1]/x as f64].into())).collect::<Vec<_>>(),"dim X");

    gpu.run_algorithm("FFT",Dim::D2(x,y),&[Y],&["src","tmp","dst"],None)?;
    assert_eq!(gpu.get::<Double2>("dst")?.iter().map(|&d| cut2(d)).collect::<Vec<_>>(),
        {let tmp = (0..num).map(|i| [((i%y)*x+i/y) as f64,0.0].into()).collect::<Vec<Double2>>()
        .chunks(y).flat_map(|c| ft(c)).map(|d| [d[0]/y as f64,d[1]/y as f64].into()).collect::<Vec<Double2>>();
        (0..num).map(|i| cut2(tmp[(i*y)%(x*y)+i/x])).collect::<Vec<Double2>>()
        },"dim Y");

    gpu.run_algorithm("FFT",Dim::D2(x,y),&[Y,X],&["src","tmp","dst"],None)?;
    let para = gpu.get::<Double2>("dst")?.iter().map(|&d| cut2(d)).collect::<Vec<_>>();
    let local = {
            let tmpp = (0..num).map(|i| [i as f64,0.0].into()).collect::<Vec<Double2>>()
            .chunks(x).flat_map(|c| ft(c)).map(|d| [d[0]/x as f64,d[1]/x as f64].into()).collect::<Vec<Double2>>();
            let tmp = (0..num).map(|i| tmpp[((i%y)*x+i/y)]).collect::<Vec<Double2>>()
            .chunks(y).flat_map(|c| ft(c)).map(|d| [d[0]/y as f64,d[1]/y as f64].into()).collect::<Vec<Double2>>();
            (0..num).map(|i| cut2(tmp[(i*y)%(x*y)+i/x])).collect::<Vec<Double2>>()
        };
    assert_eq!(para,local,"dim XY");

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
        .add_buffer("dstsum", Len(F64(0.0), num))
        .add_buffer("dstx", Len(F64(0.0), n*y))
        .add_buffer("dsty", Len(F64(0.0), n*x))
        .add_buffer("dstxy", Len(F64(0.0), n))
        .load_algorithm("moments")
        .build()?;

    let pow = (0..num)
        .map(|i| {
            let i = (i%x) as f64;
            (i,i*i,i*i*i,i*i*i*i)
        })
        .collect::<Vec<_>>();

    gpu.run_algorithm("moments",Dim::D2(x,y),&[X],&["src","tmp","sum","dstsum","dstx"],Some(&(n as u32)))?;
    assert_eq!(gpu.get::<f64>("dstx")?, pow
        .chunks(x)
        .map(|c| c.into_iter().fold([0.0,0.0,0.0,0.0],|[a,b,c,d],(e,f,g,h)| [a+e,b+f,c+g,d+h]).iter().map(|i| i/x as f64).collect::<Vec<f64>>())
        .flatten().collect::<Vec<f64>>()
    );
    gpu.run_algorithm("moments",Dim::D2(x,y),&[Y],&["src","tmp","sum","dstsum","dsty"],Some(&(n as u32)))?;
    assert_eq!(gpu.get::<f64>("dsty")?, pow
        .chunks(x)
        .fold(vec![(0.0,0.0,0.0,0.0);x],|a,c| a.iter().enumerate().map(|(i,v)| (v.0+c[i].0,v.1+c[i].1,v.2+c[i].2,v.3+c[i].3)).collect())
        .iter().map(|&(a,b,c,d)| vec![a,b,c,d]).flatten().map(|i| i/y as f64).collect::<Vec<_>>()
    );
    gpu.run_algorithm("moments",Dim::D2(x,y),&[X,Y],&["src","tmp","sum","dstsum","dstxy"],Some(&(n as u32)))?;
    assert_eq!(gpu.get::<f64>("dstxy")?, pow.iter()
        .fold([0.0,0.0,0.0,0.0],|a,c| [a[0]+c.0,a[1]+c.1,a[2]+c.2,a[3]+c.3])
        .iter().map(|i| i/(x*y) as f64).collect::<Vec<_>>()
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
fn data_file() -> gpgpu::Result<()> {
    use gpgpu::data_file::{DataFile,Format};

    let mut file = String::new();
    let (x,y,z) = (4,4,4);
    for i in 0..x {
        for k in 0..z {
            for j in 0..y {
                file += &format!("{} {} {} {}\n",i,j,k,i+x*(j+y*k));
            }
        }
    }
    let data = DataFile::parse(Format::Column(&file));
    let mut gpu = Handler::builder()?
        .load_data("data",Format::Column(&file),false)
        .add_buffer("u", Len(F64(0.0),x*y*z))
        .create_kernel(Kernel {
            name: "kern",
            args: vec![KC::Buffer("u",EmT::F64)],
            src: "
                double coord[] = {x,y,z};
                u[x+x_size*(y+y_size*z)] = data(coord);
            ",
            needed: vec![],
        })
    .build()?;

    gpu.run_arg("kern",Dim::D3(x,y,z),&[Buffer("u")])?;

    let gpudata = gpu.get::<f64>("u")?;
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
    let (x,y,z) = (2,2,2);
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
        .load_data("data",Format::Column(&file),true)
        .add_buffer("u", Len(F64(0.0),x*y*z))
        .create_kernel(Kernel {
            name: "kern",
            args: vec![KC::Buffer("u",EmT::F64)],
            src: &format!("
                double t = {};
                double coord[] = {{x/t,y/t,z/t}};
                u[x+x_size*(y+y_size*z)] = data(coord);
            ",t),
            needed: vec![],
        })
    .build()?;

    gpu.run_arg("kern",Dim::D3(x,y,z),&[Buffer("u")])?;

    let gpudata = gpu.get::<f64>("u")?;

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
        .add_buffer("u", Data(VecType::F64((0..num).map(|i| i as f64).collect())))
        .load_function("swap")
        .create_kernel(Kernel {
            name: "_main",
            src: "swap(&u[x*2],&u[x*2+1]);",
            args: vec![KC::Buffer("u",EmT::F64)],
            needed: vec![],
        })
    .build()?;

    gpu.run_arg("_main",Dim::D1(num/2),&[Buffer("u")])?;
    assert_eq!(gpu.get::<f64>("u")?, (0..num).map(|j| (j + (j+1)%2 -j%2) as f64 ).collect::<Vec<_>>());

    Ok(())
}
