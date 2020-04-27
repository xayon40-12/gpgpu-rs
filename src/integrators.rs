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

#[derive(Clone)]
pub struct PDE<'a> {
    pub dependant_var: &'a str,
    pub expr: PDETokens<'a>,
}

#[derive(Clone,Serialize,Deserialize)]
pub struct SPDE {
    pub dependant_var: String,
    pub expr: String,
}

impl<'a> From<&PDE<'a>> for SPDE {
    fn from(de: &PDE) -> SPDE {
        SPDE {
            dependant_var: de.dependant_var.into(),
            expr: de.expr.to_ocl(),
        }
    }
}

// Each PDE must be first order in time. A higher order PDE can be cut in multiple first order PDE.
// Example: d2u/dt2 + du/dt = u   =>   du/dt = z, dz/dt = u.
// It is why the parameter pdes is a Vec.
pub fn create_euler_pde<'a>(name: &'a str, dt: f64, pdes: Vec<SPDE>, params: Vec<(String,ConstructorTypes)>) -> SAlgorithm {
    let name = name.to_string();
        let mut args = vec![KCBuffer("dst",CF64)];
        args.extend(pdes.iter().map(|pde| KCBuffer(&pde.dependant_var,CF64)));
        args.extend(params.iter().map(|t| KCParam(&t.0,t.1)));
    let needed = pdes.iter().map(|d| {
        NewKernel((&Kernel {
            name: &format!("{}_{}", &name, &d.dependant_var),
            args: args.clone(),
            src: &format!("    uint _i = x+x_size*(y+y_size*z);\n    dst[_i] = {}[_i] + {}*({});", d.dependant_var, dt, d.expr),
            needed: vec![],
        }).into())
    }).collect::<Vec<_>>();
    let vars = pdes.iter().map(|d| (format!("{}_{}", &name, &d.dependant_var),d.dependant_var.clone())).collect::<Vec<_>>();
    SAlgorithm {
        name: name.clone(),
        callback: std::rc::Rc::new(move |h: &mut Handler, dim: Dim, _dimdir: &[DimDir], bufs: &[&str], other: Option<&dyn Any>| {
            // bufs[0] = dst
            // bufs[1,2,...] = differential equation buffer holders in the same order as giver for
            // create_euler function
            // bufs[i] must write in bufs[i-1]
            let num = vars.len()+1;
            if bufs.len() != num { panic!("Euler algorithm \"{}\" must be given {} buffer arguments.", &name, &num); }
            let mut args = vec![BufArg(&bufs[0],"dst")];
            for i in 0..vars.len() {
                args.push(BufArg(&bufs[i+1],&vars[i].1));
            }
            if let Some(params) = other {
                args.extend(params.downcast_ref::<Vec<(String,Types)>>().expect(&format!("Parameters of \"{}\" Euler Algorithm must be Vec<(String,Types)>.",&name)).iter().map(|i| Param(&i.0,i.1)));
            }
            for i in (0..vars.len()).rev() {
                h.run_arg(&vars[i].0,dim,&args)?;
                h.copy(bufs[0],bufs[i+1])?;
            }

            Ok(())
        }),
        needed
    }
}
