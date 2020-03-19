use crate::{Handler,kernels::Kernel};
use crate::Dim::{self,*};
use crate::descriptors::KernelDescriptor::{self,*};
use std::collections::HashMap;
use std::rc::Rc;

pub type Callback = Rc<(dyn Fn(&mut Handler, Dim, Vec<KernelDescriptor>) -> crate::Result<()>)>;

#[derive(Clone)]
pub struct Algorithm<'a> {
    pub name: &'a str,
    pub callback: Callback,
    pub kernels: Vec<Kernel<'a>>
}

pub fn algorithms<'a>() -> HashMap<&'static str,Algorithm<'a>> {
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
                let x = match dim {
                    D1(x) => x,
                    D2(x,y) => x*y,
                    D3(x,y,z) => x*y*z
                };
                let len = |spacing| x/spacing + if x%spacing > 1 { 1 } else { 0 };
                if x<=1 { return Ok(()); }
                let l = len(spacing);
                h.run("algo_sum_src", D1(l), vec![bufsrc,bufdst.clone()])?;
                if spacing<x {
                    spacing *= 2;
                    let l = len(spacing);
                    h.run("algo_sum", D1(l), vec![Param("spacing",spacing as f64),bufdst])?;
                }
                while spacing<x {
                    spacing *= 2;
                    let l = len(spacing);
                    h.run("algo_sum", D1(l), vec![Param("spacing",spacing as f64)])?;
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
