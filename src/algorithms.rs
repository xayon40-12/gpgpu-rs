use crate::{Handler,kernels::Kernel};
use crate::Dim::{self,*};
use crate::descriptors::KernelDescriptor::{self,*};
use std::collections::HashMap;
use std::rc::Rc;

pub type Callback = Rc<(dyn Fn(&mut Handler, Dim, Vec<KernelDescriptor>) -> crate::Result<()>)>;

#[derive(Clone)]
pub struct Algorithm {
    pub name: &'static str,
    pub callback: Callback,
    pub kernels: Vec<Kernel>
}

pub fn algorithms() -> HashMap<&'static str,Algorithm> {
    vec![
        Algorithm {
            name: "sum",
            callback: Rc::new(|h: &mut Handler, dim: Dim, desc: Vec<KernelDescriptor>| {
                let mut bufsrc: Option<KernelDescriptor> = None;
                let mut bufdst: Option<KernelDescriptor> = None;
                for d in desc {
                    let n = match d {
                        KernelDescriptor::Buffer(n) => n,
                        KernelDescriptor::BufArg(_,m) => m,
                        KernelDescriptor::Param(_,_) => panic!("No parameters should be given for algorithm \"sum\"")
                    };
                    if n == "src" { bufsrc = Some(d); }
                    else if n == "dst" { bufdst = Some(d); }
                }
                let bufsrc = bufsrc.expect("No source buffer given for algorithm \"sum\"");
                let bufdst = bufdst.expect("No destination buffer given for algorithm \"sum\"");
                let mut spacing = 2;
                let id = |x,spacing| x/spacing + if x%spacing > 1 { 1 } else { 0 };
                match dim {
                    D1(x) => {
                        if x<=1 { return Ok(()); }
                        let l = id(x,spacing);
                        h.run("algo_sum_src", D1(l), vec![bufsrc,bufdst.clone()])?;
                        if spacing<x {
                            spacing *= 2;
                            let l = id(x,spacing);
                            h.run("algo_sum", D1(l), vec![Param("spacing",spacing as f64),bufdst])?;
                        }
                        while spacing<x {
                            spacing *= 2;
                            let l = id(x,spacing);
                            h.run("algo_sum", D1(l), vec![Param("spacing",spacing as f64)])?;
                        }
                    },
                    _ => panic!("Dimensions higher than one are not handled yet.")
                }
                Ok(())
            }),
            kernels: vec![
                Kernel {
                    name: "algo_sum_src",
                    args: vec![Buffer("src"),Buffer("dst")],
                    src: "dst[x*2] = src[x*2]+src[x*2+1];"
                },
                Kernel {
                    name: "algo_sum",
                    args: vec![Param("spacing",0f64),Buffer("dst")],
                    src: "long s=spacing; dst[x*s] = dst[x*s]+dst[x*s+s/2];"
                },
            ]
        },
    ].into_iter().map(|a| (a.name,a)).collect()
}
