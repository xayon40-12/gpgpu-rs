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
        // sum each elements. With D1 apply on whole buffer, with D2 apply on all y sub-buffers of
        // size x (where x and y are the first and second dimensions).
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
                let (x,d): (usize,Box<dyn Fn(usize) -> Dim>) = match dim {
                    D1(x) => (x,Box::new(|l| D1(l))),
                    D2(x,y) => (x,Box::new(move |l| D2(l,y))),
                    _ => panic!("Dimension for algorithm \"sum\" should be either D1 or D2.")
                };
                let len = |spacing| x/spacing + if x%spacing > 1 { 1 } else { 0 };
                if x<=1 { return Ok(()); }
                let l = len(spacing);
                h.run_arg("algo_sum_src", d(l), vec![bufsrc,bufdst.clone(),Param("xs",U64(x as u64))])?;
                if spacing<x {
                    spacing *= 2;
                    let l = len(spacing);
                    h.run_arg("algo_sum", d(l), vec![Param("s",U64(spacing as u64)),bufdst,Param("xs",U64(x as u64))])?;
                }
                while spacing<x {
                    spacing *= 2;
                    let l = len(spacing);
                    h.run_arg("algo_sum", d(l), vec![Param("s",U64(spacing as u64)),Param("xs",U64(x as u64))])?;
                }
                Ok(())
            }),
            kernels: vec![
                Kernel {
                    name: "algo_sum_src",
                    args: vec![KC::Buffer("src",EmT::F64),KC::Buffer("dst",EmT::F64),KC::Param("xs",EmT::U64)],
                    src: "dst[x*2+y*xs] = src[x*2+y*xs]+src[x*2+y*xs+1];"
                },
                Kernel {
                    name: "algo_sum",
                    args: vec![KC::Param("s",EmT::U64),KC::Buffer("dst",EmT::F64),KC::Param("xs",EmT::U64)],
                    src: "dst[x*s+y*xs] = dst[x*s+y*xs]+dst[x*s+y*xs+s/2];"
                },
            ]
        },
        Algorithm {
            name: "moments",
            callback: Rc::new(|h: &mut Handler, dim: Dim, desc: Vec<KernelArg>| {
                if desc.len() < 4 || desc.len() > 5 {
                    panic!("Algorithm \"moment\" takes 4 or 5 arguments, {} given.", desc.len());
                }
                let bufsrc = desc[0].clone();
                let buftmp = desc[1].clone();
                let bufsum = desc[2].clone();
                let bufdst = desc[3].clone();
                let num: u32 = if desc.len() == 5 {
                    if let Param("n",U32(num)) = desc[5] {
                        if num < 1 { panic!("There must be at least one moment calculated in \"moments\" algorithm."); }
                        num
                    } else {
                        panic!("Fifth parameter of \"moment\" algorithm must be U32.");
                    }
                } else {
                    4
                };
                let x = match dim {
                    D1(x) => x,
                    D2(x,y) => x*y,
                    D3(x,y,z) => x*y*z
                };

                h.run_arg("times",D1(x),vec![bufsrc.clone(),bufsrc.clone(),buftmp.clone()])?;
                h.run_algorithm("sum",D1(x),vec![buftmp.clone(),bufsum.clone()])?;
                h.run_arg("move_0_to_i",D1(1),vec![bufsum.clone(),bufdst.clone(),Param("i",U64(0))])?;
                h.set_arg("times",vec![buftmp.clone(),bufsrc.clone(),buftmp.clone()])?;
                for i in 1..num {
                    h.run("time",D1(x))?;
                    h.run_algorithm("sum",D1(x),vec![buftmp.clone(),bufsum.clone()])?;
                    h.run_arg("move_0_to_i",D1(1),vec![bufsum.clone(),bufdst.clone(),Param("i",U64(i as u64))])?;
                }
                h.run_arg("cdivide",D1(num as _),vec![bufdst.clone(),Param("c",F64(num as f64))])?;

                Ok(())
            }),
            kernels: vec![
                Kernel {
                    name: "move_0_to_i",
                    args: vec![KC::Buffer("src",EmT::F64),KC::Buffer("dst",EmT::F64),KC::Param("i",EmT::U64)],
                    src: "dst[i] = src[0];"
                },
            ]
        },
    ].into_iter().map(|a| (a.name,a)).collect()
}
