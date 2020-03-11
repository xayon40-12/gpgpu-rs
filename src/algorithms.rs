use crate::{Handler,kernels::{self,Kernel}};
use crate::Dim::{self,*};
use crate::descriptors::KernelDescriptor::*;
use std::collections::HashMap;
use std::rc::Rc;

type Callback = Rc<(dyn Fn(&mut Handler) -> crate::Result<()>)>;

pub struct Algorithm<S: Into<String>+Clone> {
    pub name: S,
    pub callback: Callback,
    pub needed_kernels: Vec<S>
}

pub fn convert<S: Into<String>+Clone>(a: &Algorithm<S>) -> Algorithm<String> {
    Algorithm {
        name: a.name.clone().into(),
        callback: a.callback.clone(),
        needed_kernels: a.needed_kernels.iter().map(|s| s.clone().into()).collect()
    }
}

pub fn algorithms() -> HashMap<String,Algorithm<&'static str>> {
    vec![
        Algorithm {
            name: "sum",
            callback: Rc::new(|h: &mut Handler| {
                let dim = h.kernel_dim("algo_sum");
                let mut spacing = 1;
                match dim {
                    D1(x) => {
                        while spacing<x {
                            spacing *= 2;
                            let l = x/spacing + if x%spacing > 1 { 1 } else { 0 };
                            h.run("algo_sum", D1(l), vec![Param("spacing",(spacing-1) as f64)])?;
                        }
                    },
                    _ => panic!("Dimensions higher than one are not handled yet.")
                }
                Ok(())
            }),
            needed_kernels: vec!["algo_sum"]
        }
    ].into_iter().map(|a| (a.name.clone().into(),a)).collect()
}
