use crate::descriptors::{KernelConstructor::*,ConstructorTypes::*,KernelArg::*};
use crate::kernels::{Kernel};
use crate::algorithms::{SAlgorithm,SNeeded::{*,self},AlgorithmParam};
use crate::Handler;
use crate::dim::{Dim::{*,self},DimDir};
use serde::{Serialize,Deserialize};
use crate::descriptors::{Types,ConstructorTypes};

pub mod pde_ir;

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct SPDE {
    pub dvar: String,
    pub expr: Vec<String>,//one String for each dimension of the vectorial pde
}

pub struct IntegratorParam {
    pub t: f64,
    pub swap: usize,
    pub args: Vec<(String,Types)>,
}

// Each PDE must be first order in time. A higher order PDE can be cut in multiple first order PDE.
// Example: d2u/dt2 + du/dt = u   =>   du/dt = z, dz/dt = u.
// It is why the parameter pdes is a Vec.
pub fn create_euler_pde<'a>(name: &'a str, dt: f64, pdes: Vec<SPDE>, needed_buffers: Option<Vec<String>>, params: Vec<(String,ConstructorTypes)>) -> SAlgorithm {
    multistages_algorithm(name, &pdes, needed_buffers, params, dt, vec![vec![1.0]])
}

fn multistages_kernels(name: &str, pdes: &Vec<SPDE>, needed_buffers: &Option<Vec<String>>, params: Vec<(String,ConstructorTypes)>, stages: Vec<Vec<f64>>) -> Vec<SNeeded> {
    let mut args = vec![KCBuffer("dst",CF64)];
    args.extend(pdes.iter().map(|pde| KCBuffer(&pde.dvar,CF64)));
    if let Some(ns) = &needed_buffers { 
        args.extend(ns.iter().map(|n| KCBuffer(&n,CF64)));
    }
    args.extend(params.iter().map(|t| KCParam(&t.0,t.1)));
    let mut needed = pdes.iter().map(|d| {
        let mut id = "x+x_size*(y+y_size*z)".to_string();
        let len = d.expr.len();
        if len > 1 {
            id = format!("{}*({})", len, id);
        }
        let mut expr = String::new();
        for i in 0..len {
            expr += &format!("    dst[{i}+_i] = {};\n", &d.expr[i], i = i);
        }
        NewKernel((&Kernel {
            name: &format!("{}_{}", &name, &d.dvar),
            args: args.clone(),
            src: &format!("    uint _i = {};\n{}", id, expr),
            needed: vec![],
        }).into())
    }).collect::<Vec<_>>();
    for (i,v) in stages.iter().enumerate() {
        let c = v.iter().fold(0.0, |a,i| a+i);
        let mut args = vec![KCBuffer("dst",CF64)];
        let argnames = (0..v.len()).map(|i| format!("src{}", i)).collect::<Vec<_>>();
        let mut src = String::new();
        for (i,v) in v.iter().enumerate() {
            args.push(KCBuffer(&argnames[i],CF64));
            src = format!("{}*{}[i] + ", v, &argnames[i]);
        }
        src = format!("    uint i = x+x_size*(y+y_size*z);\n    dst[i] = {};", &src[..src.len()-2]);

        needed.push(NewKernel((&Kernel {
            name: &format!("stage{}", i),
            args,
            src: &src,
            needed: vec![],
        }).into()));
    }

    needed.push(NewKernel((&Kernel {
        name: "muladd",
        args: vec![KCBuffer("dst",CF64),KCBuffer("a",CF64),KCBuffer("b",CF64),KCParam("c",CF64)],
        src: "    uint i = x+x_size*(y+y_size*z);\n    dst[i] = a[i] + c*b[i];",
        needed: vec![],
    }).into()));
    needed
}

fn multistages_algorithm(name: &str, pdes: &Vec<SPDE>, needed_buffers: Option<Vec<String>>, params: Vec<(String,ConstructorTypes)>, dt: f64, stages: Vec<Vec<f64>>) -> SAlgorithm {
    let name = name.to_string();
    let vars = pdes.iter().map(|d| (format!("{}_{}", &name, &d.dvar),d.dvar.clone())).collect::<Vec<_>>();
    let mut len = 2*vars.len();
    if let Some(ns) = &needed_buffers { 
        len += ns.len();
    }
    for (i,v) in stages.iter().enumerate() {
        if v.len() != i+1 { panic!("In multisatges algorithm the coefficients must be given as a vector for each stages in the order, the first stage does not need a coefficent nor an empty, then each stage needs one more coefficient (3 coefficient for the fourth stage for instance) and the las vector correspond to how to sum each of the computed stages at the end thus it needs as many coefficient as there are stages.") }
    }

    len += stages.len()-1;
    let nb_stages = stages.len();
    let needed = multistages_kernels(&name, &pdes, &needed_buffers, params, stages);
    SAlgorithm {
        name: name.clone(),
        callback: std::rc::Rc::new(move |h: &mut Handler, dim: Dim, _dimdir: &[DimDir], bufs: &[&str], mut other: AlgorithmParam| {
            // bufs[0] = dst
            // bufs[1,2,...] = differential equation buffer holders in the same order as giver for
            // create_euler function
            // bufs[i] must write in bufs[i-1]
            let _dim: [usize; 3] = dim.into();
            let d = _dim.iter().fold(1, |a,i| a*i);
            if bufs.len() != len { panic!("Multistages algorithm \"{}\" must be given {} buffer arguments.", &name, &len); }
            let IntegratorParam{ref mut t,ref mut swap,args: iargs} = other
                .downcast_mut("There must be an Mut(&mut IntegratorParam) given as optional argument in Multistages integrator algorithm.");
            let mut args = vec![BufArg(&bufs[1-*swap],"dst")];
            for i in 0..vars.len() {
                args.push(BufArg(&bufs[2*i+*swap],&vars[i].1));
            }
            if let Some(ns) = &needed_buffers {
                let mut i = 2*vars.len();
                for n in ns {
                    args.push(BufArg(&bufs[i],&n));
                    i+=1;
                }
            }
            args.extend(iargs.iter().map(|i| Param(&i.0,i.1)));
            for i in (0..vars.len()).rev() {
                args[0] = BufArg(&bufs[2*i+1-*swap],"dst");
                h.run_arg(&vars[i].0,dim,&args)?;
                if nb_stages == 1 {
                    h.run_arg("muladd",D1(d),&[BufArg(&bufs[2*i+*swap],"a"),BufArg(&bufs[2*i+1-*swap],"b"),BufArg(&bufs[2*i+1-*swap],"dst"),Param("c",dt.into())])?;
                } else {
                    panic!("Multistages not handled yet.");
                }
            }

            *t += dt;
            *swap = 1-*swap;
            Ok(None)
        }),
        needed
    }
}
