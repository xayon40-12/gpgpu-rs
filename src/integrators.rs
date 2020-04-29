use crate::descriptors::{KernelConstructor::*,ConstructorTypes::*,KernelArg::*};
use crate::kernels::{Kernel};
use crate::algorithms::{SAlgorithm,SNeeded::*};
use crate::Handler;
use crate::dim::{Dim,DimDir};
use std::any::Any;
use serde::{Serialize,Deserialize};
use crate::descriptors::{Types,ConstructorTypes};

pub mod pde_ir;
use pde_ir::*;

#[derive(Clone,Debug)]
pub struct PDE<'a> {
    pub dvar: &'a str,
    pub expr: PDETokens<'a>,
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct SPDE {
    pub dvar: String,
    pub expr: Vec<String>,
}

impl<'a> From<&PDE<'a>> for SPDE {
    fn from(de: &PDE) -> SPDE {
        SPDE {
            dvar: de.dvar.into(),
            expr: de.expr.to_ocl(),
        }
    }
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
                expr += &format!("    dst[{}+{}*(_i)] = {}[_i] + {}*({});\n", i, len, &d.dvar, dt, &d.expr[0]);
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
        callback: std::rc::Rc::new(move |h: &mut Handler, dim: Dim, _dimdir: &[DimDir], bufs: &[&str], other: Option<&dyn Any>| {
            // bufs[0] = dst
            // bufs[1,2,...] = differential equation buffer holders in the same order as giver for
            // create_euler function
            // bufs[i] must write in bufs[i-1]
            let num = len;
            if bufs.len() != num { panic!("Euler algorithm \"{}\" must be given {} buffer arguments.", &name, &num); }
            let mut args = vec![BufArg(&bufs[0],"dst")];
            for i in 0..vars.len() {
                args.push(BufArg(&bufs[i+1],&vars[i].1));
            }
            if let Some(ns) = &needed_buffers {
                let mut i = vars.len()+1;
                for n in ns {
                    args.push(BufArg(&bufs[i],&n));
                    i+=1;
                }
            }
            let mut t = None;
            if let Some(params) = other {
                if let Some(time) = params.downcast_ref::<f64>() {
                    t = Some(*time);
                } else if let Some(params) = params.downcast_ref::<Vec<(String,Types)>>() {
                    args.extend(params.iter().map(|i| Param(&i.0,i.1)));
                } else {
                    let params = params.downcast_ref::<(f64,Vec<(String,Types)>)>().expect(&format!("Parameters of \"{}\" Euler Algorithm must be (f64,Vec<(String,Types)>) or f64.",&name));
                    t = Some(params.0);
                    args.extend(params.1.iter().map(|i| Param(&i.0,i.1)));
                }
            } else {
                panic!("There must be at least the time variable given (an f64)");
            }
            for i in (0..vars.len()).rev() {
                h.run_arg(&vars[i].0,dim,&args)?;
                h.copy(bufs[0],bufs[i+1])?;
            }

            if let Some(t) = t {
                Ok(Some(Box::new(t+dt)))
            } else {
                Ok(None)
            }
        }),
        needed
    }
}
