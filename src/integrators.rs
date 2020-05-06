use crate::descriptors::{KernelConstructor::*,ConstructorTypes::*,KernelArg::*};
use crate::kernels::{Kernel};
use crate::algorithms::{SAlgorithm,SNeeded::*,AlgorithmParam};
use crate::Handler;
use crate::dim::{Dim,DimDir};
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
    let name = name.to_string();
    let mut args = vec![KCBuffer("dst",CF64)];
    args.extend(pdes.iter().map(|pde| KCBuffer(&pde.dvar,CF64)));
    let vars = pdes.iter().map(|d| (format!("{}_{}", &name, &d.dvar),d.dvar.clone())).collect::<Vec<_>>();
    let mut len = vars.len()+1;
    if let Some(ns) = &needed_buffers { 
        args.extend(ns.iter().map(|n| KCBuffer(&n,CF64)));
        len += ns.len();
    }
    args.extend(params.iter().map(|t| KCParam(&t.0,t.1)));
    let needed = pdes.iter().map(|d| {
        let mut id = "x+x_size*(y+y_size*z)".to_string();
        let len = d.expr.len();
        if len > 1 {
            id = format!("{}*({})", len, id);
        }
        let expr = if len == 1 {
            format!("    dst[_i] = {}[_i] + {}*({});\n", &d.dvar, dt, &d.expr[0])
        } else {
            let mut expr = String::new();
            for i in 0..len {
                expr += &format!("    dst[{i}+_i] = {}[{i}+_i] + {}*({});\n", &d.dvar, dt, &d.expr[i], i = i);
            }
            expr
        };
        NewKernel((&Kernel {
            name: &format!("{}_{}", &name, &d.dvar),
            args: args.clone(),
            src: &format!("    uint _i = {};\n{}", id, expr),
            needed: vec![],
        }).into())
    }).collect::<Vec<_>>();
    SAlgorithm {
        name: name.clone(),
        callback: std::rc::Rc::new(move |h: &mut Handler, dim: Dim, _dimdir: &[DimDir], bufs: &[&str], mut other: AlgorithmParam| {
            // bufs[0] = dst
            // bufs[1,2,...] = differential equation buffer holders in the same order as giver for
            // create_euler function
            // bufs[i] must write in bufs[i-1]
            let num = len;
            if bufs.len() != num { panic!("Euler algorithm \"{}\" must be given {} buffer arguments.", &name, &num); }
            let IntegratorParam{ref mut t,swap,args: iargs} = other
                .downcast_mut("There must be an IntegratorParam struct given as optional argument in Euler integrator algorithm.");
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
            }

            *t += dt;
            Ok(None)
        }),
        needed
    }
}
