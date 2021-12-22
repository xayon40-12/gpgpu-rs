#[macro_use]
extern crate gpgpu;
use gpgpu::algorithms::AlgorithmParam::*;
use gpgpu::descriptors::{
    BufferConstructor::*, ConstructorTypes::*, FunctionConstructor::*, Types::*, VecTypes::*,
};
use gpgpu::dim::{Dim::*, DimDir::*};
use gpgpu::functions::Function;
use gpgpu::integrators::{create_euler_pde, *};
use gpgpu::pde_parser::pde_ir::ir_helper::*;
use gpgpu::Handler;

fn main() -> gpgpu::Result<()> {
    println!("\n------------------------------------------------------\n");
    pde_generator_test()?;
    println!("\n------------------------------------------------------\n");
    simple_int()?;
    println!("\n------------------------------------------------------\n");
    diffusion_int_vect()?;
    println!("\n------------------------------------------------------\n");
    diffusion_int()?;
    println!("\n------------------------------------------------------\n");
    diffusion_int_pde_gen()?;
    println!("\n------------------------------------------------------\n");
    fix_small()?;
    println!("\n------------------------------------------------------\n");
    Ok(())
}

#[allow(non_snake_case)]
fn pde_generator_test() -> gpgpu::Result<()> {
    let u = Indexable::new_vector(2, 2, 2, "u", "b");
    let u34 = Indexable::new_vector(3, 3, 5, "u", "b");
    let u34s = Indexable::new_slice(3, 3, 5, &[1..5], "u", "b");
    let T = Indexable::new_scalar(1, 1, "T", "b");
    let t = Indexable::new_scalar(2, 2, "t", "b");
    let v = vect![Const(1.0), Const(2.0)];
    let f = Forward(vec![X, Y]);
    let b = Backward(vec![X, Y]);
    let F = Forward(vec![X]);
    let B = Backward(vec![X]);
    println!("{:?}\n", (symb("a") * 1.0 / "c").to_ocl());
    println!("{:?}\n", (&v * diff(diff(&u, &f), &b)).to_ocl());
    println!("{:?}\n", (diff(diff(&u, &f), &b) * &v).to_ocl());
    println!("{:?}\n", func("cos", vec![&u * &v]).to_ocl());
    println!("{:?}\n", diff(diff(&T, &F), &B).to_ocl());
    println!("{:?}\n", diff(&t, &f).to_ocl());
    println!("{:?}\n", u34.to_ocl());
    println!("{:?}\n", diff(&u34s, &Forward(vec![X, X, Z, Y])).to_ocl());
    Ok(())
}

fn simple_int() -> gpgpu::Result<()> {
    let l = 2;
    let m = 1000;
    let dt = 2.0 / m as f64;
    let mut gpu = Handler::builder()?
        .add_buffer("u", Len(0.0.into(), l))
        .add_buffer("v", Len(3.0.into(), l))
        .add_buffer("swu", Len(0.0.into(), l))
        .add_buffer("swv", Len(0.0.into(), l))
        .create_algorithm(create_euler_pde(
            "simple",
            dt,
            vec![
                SPDE {
                    dvar: "u".into(),
                    expr: vec!["v[_i]".into()],
                },
                SPDE {
                    dvar: "v".into(),
                    expr: vec!["a".into()],
                },
            ],
            None,
            vec![("a".into(), CF64), ("t".into(), CF64)],
        ))
        .build()?;

    let args = vec![("a".to_string(), F64(1.0))];
    let mut ip = IntegratorParam {
        t: 0.0,
        increment_name: "t".into(),
        args,
    };
    let bufs = ["u", "swu", "v", "swv"];
    for _ in 0..m {
        gpu.run_algorithm("simple", D1(l), &[], &bufs, Ref(&ip))?;
        ip.t += dt;
    }
    println!("simple_int");
    println!("u:   {:?}", gpu.get("u")?.VF64());
    println!("v:   {:?}", gpu.get("v")?.VF64());
    Ok(())
}

use std::io::Write;
use std::time::SystemTime;
fn diffusion_int() -> gpgpu::Result<()> {
    let l = 1 << 18;
    let t = 7.0e-4; // t should be a lot lager in order to have good result be it would take a lot longer
    let dt = 3e-7;
    let mut gpu = Handler::builder()?
        .add_buffer("u", Data(VF64((0..l).map(|i| i as _).collect())))
        .add_buffer("swu", Len(0.0.into(), l))
        .create_algorithm(create_euler_pde(
            "diffusion",
            dt,
            vec![SPDE {
                dvar: "u".into(),
                expr: vec!["D*(u[(x+1)%x_size]-2*u[x]+u[(x-1)%x_size])*ivdx*ivdx".into()],
            }],
            None,
            vec![
                ("D".into(), CF64),
                ("ivdx".into(), CF64),
                ("t".into(), CF64),
            ],
        ))
        .build()?;

    let args = vec![("D".to_string(), F64(2.0)), ("ivdx".to_string(), F64(10.0))];
    let mut ip = IntegratorParam {
        t: 0.0,
        increment_name: "t".into(),
        args,
    };
    let bufs = ["u", "swu"];
    let m = (t / dt) as usize;
    let start = SystemTime::now();
    for i in 0..m {
        if i % (m / 100) == 0 {
            print!(" {}%\r", i * 100 / m);
            std::io::stdout().lock().flush().unwrap();
        }
        gpu.run_algorithm("diffusion", D1(l), &[], &bufs, Ref(&ip))?;
        ip.t += dt;
    }
    println!("diffusion_int");
    println!(
        "{} s / {} steps / {} elements",
        SystemTime::now().duration_since(start).unwrap().as_millis() as f64 / 1000.0,
        m,
        l
    );
    println!(
        "u[0]: {:?} <-> {}",
        gpu.get_first("u")?.F64(),
        (l - 1) as f64 / 2.0
    );
    Ok(())
}

fn diffusion_int_vect() -> gpgpu::Result<()> {
    let l = 1 << 19;
    let lv = 3 * l;
    let t = 10.0;
    let dt = 0.01;
    let mut gpu = Handler::builder()?
        .add_buffer("u", Data(VF64((0..lv).map(|i| (i % 3) as _).collect())))
        .add_buffer("swu", Len(0.0.into(), lv))
        .create_algorithm(create_euler_pde(
            "diffusion",
            dt,
            vec![SPDE {
                dvar: "u".into(),
                expr: vec![
                    "u[2+_i]-2*u[0+_i]+u[1+_i]".into(),
                    "u[0+_i]-2*u[1+_i]+u[2+_i]".into(),
                    "u[1+_i]-2*u[2+_i]+u[0+_i]".into(),
                ],
            }],
            None,
            vec![("t".into(), CF64)],
        ))
        .build()?;

    let mut ip = IntegratorParam {
        t: 0.0,
        increment_name: "t".into(),
        args: vec![],
    };
    let bufs = ["u", "swu"];
    let m = (t / dt) as usize;
    let start = SystemTime::now();
    for i in 0..m {
        if i % (m / 100) == 0 {
            print!(" {}%\r", i * 100 / m);
            std::io::stdout().lock().flush().unwrap();
        }
        gpu.run_algorithm("diffusion", D1(l), &[], &bufs, Ref(&ip))?;
        ip.t += dt;
    }
    println!("diffusion_int_vect");
    println!(
        "{} s / {} steps / {} elements",
        SystemTime::now().duration_since(start).unwrap().as_millis() as f64 / 1000.0,
        m,
        l
    );
    println!("u[0]: {:?} <-> 1", gpu.get_firsts("u", 9)?.VF64());
    Ok(())
}

#[allow(non_snake_case)]
fn diffusion_int_pde_gen() -> gpgpu::Result<()> {
    let u = Indexable::new_scalar(1, 1, "u", "b");
    let D = symb("D");
    let f = Forward(vec![X]);
    let b = Backward(vec![X]);
    let expr = (D * diff(diff(u, f), b)).to_ocl();
    println!("{:?}", expr);

    let l = 1 << 21;
    let t = 10.0;
    let dt = 0.01;
    let mut gpu = Handler::builder()?
        .add_buffer("u", Data(VF64((0..l).map(|i| i as _).collect())))
        .add_buffer("swu", Len(0.0.into(), l))
        .create_function(&Function {
            name: "b",
            args: vec![
                FCParam("x", CU32),
                FCParam("w", CU32),
                FCParam("w_size", CU32),
                FCGlobalPtr("u", CF64),
            ],
            ret_type: Some(CF64),
            src: "return u[x%get_global_size(0)];",
            needed: vec![],
        })
        .create_algorithm(create_euler_pde(
            "diffusion",
            dt,
            vec![SPDE {
                dvar: "u".into(),
                expr,
            }],
            None,
            vec![
                ("D".into(), CF64),
                ("ivdx".into(), CF64),
                ("t".into(), CF64),
            ],
        ))
        .build()?;

    let args = vec![("D".to_string(), F64(2.0)), ("ivdx".to_string(), F64(2.0))];
    let mut ip = IntegratorParam {
        t: 0.0,
        increment_name: "t".into(),
        args,
    };
    let bufs = ["u", "swu"];
    let m = (t / dt) as usize;
    let start = SystemTime::now();
    for i in 0..m {
        if i % (1 + m / 100) == 0 {
            print!(" {}%\r", i * 100 / m);
            std::io::stdout().lock().flush().unwrap();
        }
        gpu.run_algorithm("diffusion", D1(l), &[], &bufs, Ref(&ip))?;
        ip.t += dt;
    }
    println!("diffusion_int_pde_gen");
    println!(
        "{} s / {} steps / {} elements",
        SystemTime::now().duration_since(start).unwrap().as_millis() as f64 / 1000.0,
        m,
        l
    );
    println!(
        "u[0]: {:?} <-> {}",
        gpu.get_first("u")?.F64(),
        (l - 1) as f64 / 2.0
    );
    Ok(())
}

#[allow(non_snake_case)]
fn fix_small() -> gpgpu::Result<()> {
    let u = Indexable::new_scalar(1, 1, "u", "b");
    let D = symb("D");
    let f = Forward(vec![X]);
    let b = Backward(vec![X]);
    let expr = (D * diff(diff(u, f), b)).to_ocl();
    println!("{:?}", expr);

    let l = 1 << 21;
    let t = 10.0;
    let dt = 0.01;
    let mut gpu = Handler::builder()?
        .add_buffer("u", Data(VF64((0..l).map(|i| i as _).collect())))
        .create_function(&Function {
            name: "b",
            args: vec![
                FCParam("x", CU32),
                FCParam("w", CU32),
                FCParam("w_size", CU32),
                FCGlobalPtr("u", CF64),
            ],
            ret_type: Some(CF64),
            src: "return u[x%get_global_size(0)];",
            needed: vec![],
        })
        .create_algorithm(create_euler_pde(
            "diffusion",
            dt,
            vec![SPDE {
                dvar: "u".into(),
                expr,
            }],
            None,
            vec![
                ("D".into(), CF64),
                ("ivdx".into(), CF64),
                ("t".into(), CF64),
            ],
        ))
        .build()?;

    let args = vec![("D".to_string(), F64(2.0)), ("ivdx".to_string(), F64(2.0))];
    let mut ip = IntegratorParam {
        t: 0.0,
        increment_name: "t".into(),
        args,
    };
    let bufs = ["u", "swu"];
    let m = (t / dt) as usize;
    let start = SystemTime::now();
    for i in 0..m {
        if i % (1 + m / 100) == 0 {
            print!(" {}%\r", i * 100 / m);
            std::io::stdout().lock().flush().unwrap();
        }
        gpu.run_algorithm("diffusion", D1(l), &[], &bufs, Ref(&ip))?;
        ip.t += dt;
    }
    println!("diffusion_int_pde_gen");
    println!(
        "{} s / {} steps / {} elements",
        SystemTime::now().duration_since(start).unwrap().as_millis() as f64 / 1000.0,
        m,
        l
    );
    println!(
        "u[0]: {:?} <-> {}",
        gpu.get_first("u")?.F64(),
        (l - 1) as f64 / 2.0
    );
    Ok(())
}
