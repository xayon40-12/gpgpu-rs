use gpgpu::integrators::{*,PDETokens::*,DiffDir::*,create_euler_pde};
use gpgpu::dim::{Dim::*,DimDir::*};
use gpgpu::descriptors::{BufferConstructor::*,ConstructorTypes::*,Types::*,VecTypes::*};
use gpgpu::Handler;

fn main() -> gpgpu::Result<()> {
    println!("\n------------------------------------------------------\n");
    pde_generator_test()?;
    println!("\n------------------------------------------------------\n");
    simple_int()?;
    println!("\n------------------------------------------------------\n");
    diffusion_int()?;
    println!("\n------------------------------------------------------\n");
    Ok(())
}

#[allow(non_snake_case)]
fn pde_generator_test() -> gpgpu::Result<()> {
    let u = Indx(IndexingTypes::new_vector(2,"u","bound"));
    let T = Indx(IndexingTypes::new_scalar(1,"T","bound"));
    let v = Vect(vec![1.into(),2.into()]);
    let f = Forward(vec![X,Y]);
    let b = Backward(vec![X,Y]);
    let F = Forward(vec![X]);
    let B = Backward(vec![X]);
    println!("{}\n", Add(&Mul(&Symb("a"),&Const(1.into())),&Sub(&Div(&Symb("b"),&Symb("c")),&Symb("d"))).to_ocl());
    println!("{}\n", &Mul(&v,&Diff(&Diff(&u,&f),&b)).to_ocl());
    println!("{}\n", &Mul(&Diff(&Diff(&u,&f),&b),&v).to_ocl());
    println!("{}\n", &Mul(&u,&v).to_ocl());
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
                SPDE{ dependant_var: "u".into(), expr: "v[_i]".into() },
                SPDE{ dependant_var: "v".into(), expr: "a".into() },
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

fn diffusion_int() -> gpgpu::Result<()> {
    let l = 1<<8;
    let t = 1000.0;
    let dt = 0.01;
    let mut gpu = Handler::builder()?
        .add_buffer("u", Data(VF64((0..l).map(|i| i as _).collect())))
        .add_buffer("dst", Len(0.0.into(),l))
        .create_algorithm(create_euler_pde("diffusion",dt,vec![
                SPDE{ dependant_var: "u".into(), expr: "D*(u[(x+1)%x_size]-2*u[x]+u[(x-1)%x_size])".into() },
        ],vec![("D".into(),CF64)]))
        .build()?;

    let args = vec![("D".to_string(),F64(5.0))];
    let m = (t/dt) as usize;
    for _ in 0..m {
        gpu.run_algorithm("diffusion",D1(l),&[X],&["dst","u"],Some(&args))?;
    }
    println!("u[0]: {:?} <-> {}", gpu.get_first("u")?.F64(), (l-1) as f64/2.0);
    Ok(())
}
