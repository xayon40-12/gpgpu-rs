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

macro_rules! ifs {
    ($desc:ident, $alg:expr, $num:literal) => {
        if $desc.len() != $num {
            panic!("Algorithm \"{}\" takes {} arguments, {} given.", $alg, $num, $desc.len());
        }
    };
    ($desc:ident, $alg:expr, $min:literal..$max:literal) => {
        if $desc.len() < $min || $desc.len() > $max {
            panic!("Algorithm \"{}\" takes {} arguments, {} given.", $alg, stringify!($min..$max), $desc.len());
        }
    };
}

macro_rules! bufs {
    ($desc:ident, $alg:expr, $num:literal, $($arg:ident) +) => {
        ifs!($desc,$alg,$num);
        bufs!($desc,$alg,$($arg)+);
    };
    ($desc:ident, $alg:expr, $min:literal..$max:literal, $($arg:ident) +) => {
        ifs!($desc,$alg,$min..$max);
        bufs!($desc,$alg,$($arg)+);
    };
    ($desc:ident, $alg:expr, $($arg:ident) +) => {
        let mut __i = 0;
        $(
            let $arg = if let KernelArg::Buffer(s) = $desc[__i].clone() { s } else { panic!("Parameter {} (start at 0) of algorithms \"{}\" must be \"Buffer\"", __i, $alg) };
            __i += 1;
        )+
    };
}

macro_rules! dim1or2 {
    ($alg:expr, $dim:ident, $x:ident $d:ident) => {
        let ($x,$d): (usize,Box<dyn Fn(usize) -> Dim>) = match $dim {
            D1(x) => (x,Box::new(|l| D1(l))),
            D2(x,y) => (x,Box::new(move |l| D2(l,y))),
            _ => panic!("Dimension for algorithm \"{}\" should be either D1 or D2.", $alg)
        };
    }
}

pub fn algorithms<'a>() -> HashMap<&'static str,Algorithm<'a>> {
    vec![
        // sum each elements. With D1 apply on whole buffer, with D2 apply on all y sub-buffers of
        // size x (where x and y are the first and second dimensions).
        Algorithm {
            name: "sum",
            callback: Rc::new(|h: &mut Handler, dim: Dim, desc: Vec<KernelArg>| {
                bufs!(desc, "sum", 2,
                    src
                    dst
                );
                let mut spacing = 2;
                dim1or2!("sum",dim,x d);
                let len = |spacing| x/spacing + if x%spacing > 1 { 1 } else { 0 };
                if x<=1 { return Ok(()); }
                let l = len(spacing);
                h.run_arg("algo_sum_src", d(l), vec![BufArg(src,"src"),BufArg(dst,"dst").clone(),Param("xs",U64(x as u64))])?;
                if spacing<x {
                    spacing *= 2;
                    let l = len(spacing);
                    h.run_arg("algo_sum", d(l), vec![Param("s",U64(spacing as u64)),BufArg(dst,"dst"),Param("xs",U64(x as u64))])?;
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
        // Compute correlation. With D1 apply on whole buffer, with D2 apply on all y sub-buffers of
        // size x (where x and y are the first and second dimensions).
        Algorithm {
            name: "correlation",
            callback: Rc::new(|h: &mut Handler, dim: Dim, desc: Vec<KernelArg>| {
                bufs!(desc, "correlation", 2,
                    src
                    dst
                );
                dim1or2!("correlation",dim,x d);
                h.run_arg("correlation", d(x), vec![BufArg(src,"src"),BufArg(dst,"dst")])?;
                Ok(())
            }),
            kernels: vec![
                Kernel {
                    name: "correlation",
                    args: vec![KC::Buffer("src",EmT::F64),KC::Buffer("dst",EmT::F64)],
                    src: "dst[x+y*x_size] = src[x+y*x_size]*src[x_size/2+y*x_size];"
                }
            ]
        },
        // Compute moments. With D1 apply on whole buffer, with D2 apply on all y sub-buffers of
        // size x (where x and y are the first and second dimensions).
        Algorithm {
            name: "moments",
            callback: Rc::new(|h: &mut Handler, dim: Dim, desc: Vec<KernelArg>| {
                bufs!(desc, "moments", 4..5,
                    src
                    tmp
                    sum
                    dst
                );
                let num: u32 = if desc.len() == 5 {
                    if let Param("n",U32(num)) = desc[4] {
                        if num < 1 { panic!("There must be at least one moment calculated in \"moments\" algorithm."); }
                        num
                    } else {
                        panic!("Fifth parameter of \"moment\" algorithm must be U32.");
                    }
                } else {
                    4
                };
                let (x,y,l) = match dim {
                    D1(x) => (x,1,x),
                    D2(x,y) => (x,y,x*y),
                    _ => panic!("Dimension should be either D1 or D2 for algorithm \"moments\"")
                };

                h.run_algorithm("sum",dim,vec![Buffer(src),Buffer(sum)])?;
                h.set_arg("move_0_to_i",vec![BufArg(sum,"src"),BufArg(dst,"dst"),Param("i",U64(0)),Param("xs",U64(x as u64)),Param("n",U64(num as u64))]);
                h.run("move_0_to_i",D2(1,y))?;
                if num >= 1 {
                    h.run_arg("times",D1(l),vec![BufArg(src,"a"),BufArg(src,"b"),BufArg(tmp,"dst")])?;
                    h.run_algorithm("sum",dim,vec![Buffer(tmp),Buffer(sum)])?;
                    h.run_arg("move_0_to_i",D2(1,y),vec![Param("i",U64(1))])?;
                    h.set_arg("times",vec![BufArg(tmp,"a")])?;
                }
                for i in 2..num {
                    h.run("times",D1(l))?;
                    h.run_algorithm("sum",dim,vec![Buffer(tmp),Buffer(sum)])?;
                    h.run_arg("move_0_to_i",D2(1,y),vec![Param("i",U64(i as u64))])?;
                }
                h.run_arg("cdivides",D1(l as _),vec![BufArg(dst,"src"),Param("c",F64(x as f64)),BufArg(dst,"dst")])?;

                Ok(())
            }),
            kernels: vec![
                Kernel {
                    name: "move_0_to_i",
                    args: vec![KC::Buffer("src",EmT::F64),KC::Buffer("dst",EmT::F64),KC::Param("i",EmT::U64),KC::Param("xs",EmT::U64),KC::Param("n",EmT::U64)],
                    src: "dst[y*n+i] = src[y*xs];"
                },
            ]
        },
    ].into_iter().map(|a| (a.name,a)).collect()
}
