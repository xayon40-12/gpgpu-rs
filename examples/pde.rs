use gpgpu::integrators::{*,pde_ir::{*,PDETokens::*,DiffDir::*},create_euler_pde};
use gpgpu::dim::{Dim::*,DimDir::*};
use gpgpu::descriptors::{BufferConstructor::*,ConstructorTypes::*,Types::*,VecTypes::*,FunctionConstructor::*};
use gpgpu::functions::Function;
use gpgpu::Handler;

fn main() -> gpgpu::Result<()> {
    println!("\n------------------------------------------------------\n");
    pde_generator_test()?;
    println!("\n------------------------------------------------------\n");
    simple_int()?;
    println!("\n------------------------------------------------------\n");
    diffusion_int()?;
    println!("\n------------------------------------------------------\n");
    diffusion_int_pde_gen()?;
    println!("\n------------------------------------------------------\n");
    Ok(())
}

#[allow(non_snake_case)]
fn pde_generator_test() -> gpgpu::Result<()> {
    let u = Indx(IndexingTypes::new_vector(2,"u","b"));
    let T = Indx(IndexingTypes::new_scalar(1,"T","b"));
    let v = Vect(&[&Const(1.0),&Const(2.0)]);
    let f = Forward(vec![X,Y]);
    let b = Backward(vec![X,Y]);
    let F = Forward(vec![X]);
    let B = Backward(vec![X]);
    println!("{}\n", Add(&Mul(&Symb("a"),&Const(1.into())),&Sub(&Div(&Symb("b"),&Symb("c")),&Symb("d"))).to_ocl());
    println!("{}\n", &Mul(&v,&Diff(&Diff(&u,&f),&b)).to_ocl());
    println!("{}\n", &Mul(&Diff(&Diff(&u,&f),&b),&v).to_ocl());
    println!("{}\n", &Func("cos",&Mul(&u,&v)).to_ocl());
    println!("{}\n", &Diff(&Diff(&T,&F),&B).to_ocl());
    Ok(())
}

fn simple_int() -> gpgpu::Result<()> {
    let l = 2;
    let m = 1000;
    let mut gpu = Handler::builder()?
        .add_buffer("u", Len(0.0.into(),l))
        .add_buffer("v", Len(3.0.into(),l))
        .add_buffer("dst", Len(0.0.into(),l))
        .create_algorithm(create_euler_pde("simple",2.0/m as f64,vec![
                SPDE{ dvar: "u".into(), expr: "v[_i]".into() },
                SPDE{ dvar: "v".into(), expr: "a".into() },
        ],vec![("a".into(),CF64)]))
        .build()?;

    let args = vec![("a".to_string(),F64(1.0))];
    for _ in 0..m {
        gpu.run_algorithm("simple",D1(l),&[X],&["dst","u","v"],Some(&args))?;
    }
    println!("dst: {:?}", gpu.get("dst")?.VF64());
    println!("u:   {:?}", gpu.get("u")?.VF64());
    println!("v:   {:?}", gpu.get("v")?.VF64());
    Ok(())
}

use std::io::Write;
use std::time::SystemTime;
fn diffusion_int() -> gpgpu::Result<()> {
    let l = 1<<24;
    let t = 10.0;
    let dt = 0.01;
    let mut gpu = Handler::builder()?
        .add_buffer("u", Data(VF64((0..l).map(|i| i as _).collect())))
        .add_buffer("dst", Len(0.0.into(),l))
        .create_algorithm(create_euler_pde("diffusion",dt,vec![
                SPDE{ dvar: "u".into(), expr: "D*(u[(x+1)%x_size]-2*u[x]+u[(x-1)%x_size])*ivdx*ivdx".into() },
        ],vec![("D".into(),CF64),("ivdx".into(),CF64)]))
        .build()?;

    let args = vec![("D".to_string(),F64(5.0)),("ivdx".to_string(),F64(0.17))];
    let m = (t/dt) as usize;
    let start = SystemTime::now();
    for i in 0..m {
        if i%(m/100) == 0 { print!(" {}%\r",i*100/m); std::io::stdout().lock().flush().unwrap(); }
        gpu.run_algorithm("diffusion",D1(l),&[X],&["dst","u"],Some(&args))?;
    }
    println!("{} s / {} steps / {} elements", SystemTime::now().duration_since(start).unwrap().as_millis() as f64/1000.0, m, l);
    println!("u[0]: {:?} <-> {}", gpu.get_first("u")?.F64(), (l-1) as f64/2.0);
    Ok(())
}

#[allow(non_snake_case)]
fn diffusion_int_pde_gen() -> gpgpu::Result<()> {
    let u = Indx(IndexingTypes::new_scalar(1,"u","b"));
    let D = Symb("D");
    let f = Forward(vec![X]);
    let b = Backward(vec![X]);
    let expr = Mul(&D,&Diff(&Diff(&u,&f),&b)).to_ocl();

    let l = 1<<24;
    let t = 10.0;
    let dt = 0.01;
    let mut gpu = Handler::builder()?
        .add_buffer("u", Data(VF64((0..l).map(|i| i as _).collect())))
        .add_buffer("dst", Len(0.0.into(),l))
        .create_function(&Function {
            name: "b",
            args: vec![FCParam("x",CU32),FCGlobalPtr("u",CF64)],
            ret_type: Some(CF64),
            src: "return u[x%get_global_size(0)];",
            needed: vec![],
        })
        .create_algorithm(create_euler_pde("diffusion",dt,vec![
                SPDE{ dvar: "u".into(), expr },
        ],vec![("D".into(),CF64),("ivdx".into(),CF64)]))
        .build()?;

    let args = vec![("D".to_string(),F64(5.0)),("ivdx".to_string(),F64(0.17))];
    let m = (t/dt) as usize;
    let start = SystemTime::now();
    for i in 0..m {
        if i%(m/100) == 0 { print!(" {}%\r",i*100/m); std::io::stdout().lock().flush().unwrap(); }
        gpu.run_algorithm("diffusion",D1(l),&[X],&["dst","u"],Some(&args))?;
    }
    println!("{} s / {} steps / {} elements", SystemTime::now().duration_since(start).unwrap().as_millis() as f64/1000.0, m, l);
    println!("u[0]: {:?} <-> {}", gpu.get_first("u")?.F64(), (l-1) as f64/2.0);
    Ok(())
}
