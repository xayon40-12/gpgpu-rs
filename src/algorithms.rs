use crate::{Handler,kernels::Kernel};
use crate::Dim::{self,*};
use crate::descriptors::KernelArg::{self,*};
use crate::descriptors::Type::*;
use std::collections::HashMap;
use std::rc::Rc;
use crate::descriptors::KernelConstructor as KC;
use crate::descriptors::EmptyType as EmT;

pub type Callback = Rc<(dyn Fn(&mut Handler, Dim, Vec<KernelArg>) -> crate::Result<()>)>;

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
            callback: Rc::new(|h: &mut Handler, dim: Dim, desc: Vec<KernelArg>| {
                let mut bufsrc: Option<KernelArg> = None;
                let mut bufdst: Option<KernelArg> = None;
                for d in desc {
                    let n = match d {
                        KernelArg::Buffer(n) => n,
                        KernelArg::BufArg(_,m) => m,
                        KernelArg::Param(_,_) => panic!("No parameters should be given for algorithm \"sum\"")
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
                h.run_arg("algo_sum_src", D1(l), vec![bufsrc,bufdst.clone()])?;
                if spacing<x {
                    spacing *= 2;
                    let l = len(spacing);
                    h.run_arg("algo_sum", D1(l), vec![Param("s",U64(spacing as u64)),bufdst])?;
                }
                while spacing<x {
                    spacing *= 2;
                    let l = len(spacing);
                    h.run_arg("algo_sum", D1(l), vec![Param("s",U64(spacing as u64))])?;
                }
                Ok(())
            }),
            kernels: vec![
                Kernel {
                    name: "algo_sum_src",
                    args: vec![KC::Buffer("src",EmT::F64),KC::Buffer("dst",EmT::F64)],
                    src: "dst[x*2] = src[x*2]+src[x*2+1];"
                },
                Kernel {
                    name: "algo_sum",
                    args: vec![KC::Param("s",EmT::U64),KC::Buffer("dst",EmT::F64)],
                    src: "dst[x*s] = dst[x*s]+dst[x*s+s/2];"
                },
            ]
        },
    ].into_iter().map(|a| (a.name,a)).collect()
}
