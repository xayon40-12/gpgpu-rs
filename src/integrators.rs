use crate::descriptors::{KernelConstructor::*,ConstructorTypes::*,KernelArg::*};
use crate::functions::Needed::*;
use crate::kernels::{Kernel};
use crate::algorithms::{SAlgorithm,SNeeded::*};
use crate::Handler;
use crate::dim::{Dim,DimDir};
use std::any::Any;

pub struct DE<'a> {
    dependant_var: &'a str,
    expr: &'a str,
}

pub struct SDE {
    dependant_var: String,
    expr: String,
}

impl<'a> From<&DE<'a>> for SDE {
    fn from(de: &DE) -> SDE {
        SDE {
            dependant_var: de.dependant_var.into(),
            expr: de.expr.into(),
        }
    }
}

pub fn create_euler<'a>(name: &'a str, dt: f64, de: Vec<SDE>) -> SAlgorithm {
    let name = name.to_string();
    let needed = de.iter().map(|d| {
        NewKernel((&Kernel {
            name: &format!("{}_{}", &name, &d.dependant_var),
            args: vec![KCBuffer("dst",CF64),KCBuffer(&d.dependant_var,CF64)],
            src: &format!("uint id = x+x_size*(y+y_size*z); dst[id] = {}[id] +{}*({})", d.dependant_var, dt, d.expr),
            needed: vec![FuncName("mid")],
        }).into())
    }).collect::<Vec<_>>();
    let vars = de.iter().map(|d| (format!("{}_{}", &name, &d.dependant_var),d.dependant_var.clone())).collect::<Vec<_>>();
    SAlgorithm {
        name: name.clone(),
        callback: std::rc::Rc::new(move |h: &mut Handler, dim: Dim, _dimdir: &[DimDir], bufs: &[&str], _other: Option<&dyn Any>| {
            // bufs[0] = dst
            // bufs[1,2,...] = differential equation buffer holders in the same order as giver for
            // create_euler function
            // bufs[i] must write in bufs[i-1]
            let num = vars.len()+1;
            if bufs.len() != num { panic!("Euler algorithm \"{}\" must be given {} buffer arguments.", &name, &num); }
            let mut i = num;
            for (name,var) in vars.iter().rev() {
                i -= 1;
                h.run_arg(name,dim,&[BufArg(bufs[i-1],"dst"),BufArg(bufs[i],var)])?;
            }

            Ok(())
        }),
        needed
    }
}
