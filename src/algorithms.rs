use crate::{Handler,kernels::Kernel};
use crate::Dim::{self,*};
use crate::descriptors::KernelArg::*;
use crate::descriptors::{Type::*,VecType};
use std::collections::HashMap;
use std::rc::Rc;
use crate::descriptors::KernelConstructor as KC;
use crate::descriptors::EmptyType as EmT;
use std::any::Any;
use crate::DimDir::{*,self};

//Fn(handler: &mut Handler, dim: Dim, dim_dir: &[DimDir], buf_names: Vec<String>, other_args: Option<&dyn Any>)
pub type Callback = Rc<(dyn Fn(&mut Handler, Dim, &[DimDir], &[&str], Option<&dyn Any>) -> crate::Result<Option<Vec<VecType>>>)>;

#[derive(Clone)]
pub struct Algorithm<'a> { //TODO use one SC for each &'a str
    pub name: &'a str,
    pub callback: Callback,
    pub needed: Vec<Needed<'a>>
}

#[derive(Clone)]
pub enum Needed<'a> { //TODO use one SC for each &'a str
    KernelName(&'a str),
    AlgorithmName(&'a str),
    NewKernel(Kernel<'a>)
}
use Needed::*;

macro_rules! ifs {
    ($bufs:ident, $alg:expr, $num:literal) => {
        if $bufs.len() != $num {
            panic!("Algorithm \"{}\" takes {} buffer names, {} given.", $alg, $num, $bufs.len());
        }
    };
    ($bufs:ident, $alg:expr, $min:literal..$max:literal) => {
        if $bufs.len() < $min || $bufs.len() > $max {
            panic!("Algorithm \"{}\" takes {} buffer names, {} given.", $alg, stringify!($min..$max), $bufs.len());
        }
    };
}

macro_rules! bufs {
    ($bufs:ident, $alg:expr, $num:literal, $($arg:ident) +) => {
        ifs!($bufs,$alg,$num);
        bufs!($bufs,$alg,$($arg)+);
    };
    ($bufs:ident, $alg:expr, $min:literal..$max:literal, $($arg:ident) +) => {
        ifs!($bufs,$alg,$min..$max);
        bufs!($bufs,$alg,$($arg)+);
    };
    ($bufs:ident, $alg:expr, $($arg:ident) +) => {
        let mut __i = 0;
        $(
            let $arg = $bufs[__i];
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

macro_rules! callback_gen {
    ($h:ident, $dim:ident, $dimdir:pat, $bufs:ident, $other:pat, $body:tt) => {
        Rc::new(|$h: &mut Handler, $dim: Dim, $dimdir: &[DimDir], $bufs: &[&str], $other: Option<&dyn Any>| $body)
    };
    (simple $name:literal) => {
        callback_gen!(iner a $name);
    };
    (log $name:literal) => {
        callback_gen!(iner a|b $name);
    };
    (iner $($type:meta)|+ $name:literal) => {
        Rc::new(|h: &mut Handler, dim: Dim, dirs: &[DimDir], bufs: &[&str], _: Option<&dyn Any>| {
                bufs!(bufs, "sum", 2,
                    src
                    dst
                );
                let len = |spacing,x| x/spacing + if x%spacing > 1 { 1 } else { 0 };
                let (d, dims): (Box<dyn Fn(usize, DimDir) -> Dim>, _) = match dim {
                    D1(x) => {
                        if x<=1 { panic!("Each given dim in algorithm \"{}\" must be strictly greater than 1, given (x: {})", $name, x); }
                        (Box::new(move |s,dir| match dir {
                            X => D1(len(s,x)),
                            _ => panic!("Direction {:?} does not exist for in dimension {:?} for algorithm \"{}\""),
                        }),
                        vec![(x,X)])
                    },
                    D2(x,y) => {
                        if x<=1 || y<=1 { panic!("Each given dim in algorithm \"{}\" must be strictly greater than 1, given (x: {}, y: {})", $name, x, y); }
                        (Box::new(move |s,dir| match dir {
                            X => D2(len(s,x),y),
                            Y => D2(x,len(s,y)),
                            _ => panic!("Direction {:?} does not exist for in dimension {:?} for algorithm \"{}\"",dir,dim,$name),
                        }),
                        dirs.into_iter().map(|dir| match dir {
                            X => (x,X),
                            Y => (y,Y),
                            _ => panic!(),
                        }).collect())
                    },
                    _ => panic!("not")
                };
                #[allow(unused)]
                for (x,dir) in dims {
                    callback_gen!($($type)|+ $name, h, d, dir, src, dst, x);
                }
                Ok(None)
            })
    };
    ($a:meta $name:literal, $h:ident, $d:ident, $dir:ident, $src:ident, $dst:ident, $x:ident) => {
        $h.run_arg(concat!("algo_",$name), $d(1,$dir), &[BufArg(&$src,"src"),BufArg(&$dst,"dst")])?;
    };
    ($a:meta|$b:meta $name:literal, $h:ident, $d:ident, $dir:ident, $src:ident, $dst:ident, $x:ident) => {
        let mut spacing = 2;
        $h.run_arg(concat!("algo_",$name), $d(spacing,$dir), &[Param("s",U64(spacing as u64)),BufArg(&$src,"src"),BufArg(&$dst,"dst"),Param("xs",U64($x as u64))])?;
        $h.set_arg(concat!("algo_",$name), &[BufArg(&$dst,"src")])?;
        while spacing<$x {
            spacing *= 2;
            $h.run_arg(concat!("algo_",$name), $d(spacing,$dir), &[Param("s",U64(spacing as u64))])?;
        }
    };
}

pub fn algorithms<'a>() -> HashMap<&'static str,Algorithm<'a>> {
    vec![
        // sum each elements. With D1 apply on whole buffer, with D2 apply on all y sub-buffers of
        // size x (where x and y are the first and second dimensions).
        Algorithm {
            name: "sum",
            callback: callback_gen!(log "sum"),
            needed: vec![//TODO add other direction algorithms
                NewKernel(Kernel {
                    name: "algo_sum",
                    args: vec![KC::ConstBuffer("src",EmT::F64),KC::Buffer("dst",EmT::F64),KC::Param("s",EmT::U64),KC::Param("xs",EmT::U64)],
                    src: "dst[x*s+y*xs] = src[x*s+y*xs]+src[x*s+y*xs+s/2];",
                    needed: vec![],
                }),
            ]
        },
        // find min each elements. With D1 apply on whole buffer, with D2 apply on all y sub-buffers of
        // size x (where x and y are the first and second dimensions).
        Algorithm {
            name: "min",
            callback: callback_gen!(log "min"),
            needed: vec![
                NewKernel(Kernel {
                    name: "algo_min",
                    args: vec![KC::ConstBuffer("src",EmT::F64),KC::Buffer("dst",EmT::F64),KC::Param("s",EmT::U64),KC::Param("xs",EmT::U64)],
                    src: "dst[x*s+y*xs] = (src[x*s+y*xs]<src[x*s+y*xs+s/2])?src[x*s+y*xs]:src[x*s+y*xs+s/2];",
                    needed: vec![],
                }),
            ]
        },
        // find max each elements. With D1 apply on whole buffer, with D2 apply on all y sub-buffers of
        // size x (where x and y are the first and second dimensions).
        Algorithm {
            name: "max",
            callback: callback_gen!(log "max"),
            needed: vec![
                NewKernel(Kernel {
                    name: "algo_max",
                    args: vec![KC::ConstBuffer("src",EmT::F64),KC::Buffer("dst",EmT::F64),KC::Param("s",EmT::U64),KC::Param("xs",EmT::U64)],
                    src: "dst[x*s+y*xs] = (src[x*s+y*xs]>src[x*s+y*xs+s/2])?src[x*s+y*xs]:src[x*s+y*xs+s/2];",
                    needed: vec![],
                }),
            ]
        },
        // Compute correlation. With D1 apply on whole buffer, with D2 apply on all y sub-buffers of
        // size x (where x and y are the first and second dimensions).
        Algorithm {
            name: "correlation",
            callback: callback_gen!(simple "correlation"),
            needed: vec![
                NewKernel(Kernel {
                    name: "algo_correlation",
                    args: vec![KC::ConstBuffer("src",EmT::F64),KC::Buffer("dst",EmT::F64)],
                    src: "dst[x+y*x_size] = src[x+y*x_size]*src[x_size/2+y*x_size];",
                    needed: vec![],
                })
            ]
        },
        // Compute moments. With D1 apply on whole buffer, with D2 apply on all y sub-buffers of
        // size x (where x and y are the first and second dimensions).
        Algorithm {
            name: "moments",
            callback: callback_gen!(h,dim,dirs,bufs,param, {
                bufs!(bufs, "moments", 4..5,
                    src
                    tmp
                    sum
                    dst
                );
                let num: u32 = if let Some(p) = param {
                    *p.downcast_ref().expect("Optional parameter of \"moment\" algorithm must be U32.")
                } else {
                    4
                };
                if num < 1 { panic!("There must be at least one moment calculated in \"moments\" algorithm."); }

                let (x,y,l) = match dim {
                    D1(x) => (x,1,x),
                    D2(x,y) => (x,y,x*y),
                    _ => panic!("Dimension should be either D1 or D2 for algorithm \"moments\"")
                };

                h.run_algorithm("sum",dim,dirs,&[src,sum],None)?;
                h.set_arg("move_0_to_i",&[BufArg(&sum,"src"),BufArg(&dst,"dst"),Param("i",U64(0)),Param("xs",U64(x as u64)),Param("n",U64(num as u64))])?;
                h.run("move_0_to_i",D2(1,y))?;
                if num >= 1 {
                    h.run_arg("times",D1(l),&[BufArg(&src,"a"),BufArg(&src,"b"),BufArg(&tmp,"dst")])?;
                    h.run_algorithm("sum",dim,dirs,&[tmp,sum],None)?;
                    h.run_arg("move_0_to_i",D2(1,y),&[Param("i",U64(1))])?;
                    h.set_arg("times",&[BufArg(&tmp,"a")])?;
                }
                for i in 2..num {
                    h.run("times",D1(l))?;
                    h.run_algorithm("sum",dim,dirs,&[tmp,sum],None)?;
                    h.run_arg("move_0_to_i",D2(1,y),&[Param("i",U64(i as u64))])?;
                }
                h.run_arg("cdivides",D1(l as _),&[BufArg(&dst,"src"),Param("c",F64(x as f64)),BufArg(&dst,"dst")])?;

                Ok(None)
            }),
            needed: vec![
                NewKernel(Kernel {
                    name: "move_0_to_i",
                    args: vec![KC::ConstBuffer("src",EmT::F64),KC::Buffer("dst",EmT::F64),KC::Param("i",EmT::U64),KC::Param("xs",EmT::U64),KC::Param("n",EmT::U64)],
                    src: "dst[y*n+i] = src[y*xs];",
                    needed: vec![],
                }),
                KernelName("times"),
                KernelName("cdivides"),
                AlgorithmName("sum"),
            ]
        },
        #[allow(unused)] //TODO remove when algorithm FFT is finished
        Algorithm {
            name: "FFT",
            callback: callback_gen!(h,dim,_,bufs,_, {
                bufs!(bufs, "FFT", 3,
                    src
                    tmp
                    dst
                );

                let x = match dim {
                    Dim::D1(x) => x,
                    _ => panic!("Dimensions higher than one are not handled for \"FFT\"")
                };
                if !x.is_power_of_two() { panic!("FFT dimensions must be power of two."); }

                let mut i = 0;
                let mut lnx = 0;
                while (1<<lnx) < x { lnx += 1; }
                let m = lnx%2;
                let sd = [&tmp,&dst];
                h.run_arg("FFT",Dim::D1(x),&[BufArg(&src,"src"),BufArg(sd[(i+m)%2],"dst"),Param("i",U64(i as u64))]);
                while (1<<i) < x {
                    i += 1;
                    h.run_arg("FFT",Dim::D1(x),&[BufArg(sd[(i+m+1)%2],"src"),BufArg(sd[(i+m)%2],"dst"),Param("i",U64(i as u64))]);
                }
                Ok(None)
            }),
            needed: vec![
                NewKernel(Kernel {
                    name: "FFT",
                    args: vec![KC::ConstBuffer("src",EmT::F64_2),KC::Buffer("dst",EmT::F64_2),KC::Param("i",EmT::U64)],
                    src: "
                        ulong Ni = x_size>>i;
                        ulong u = x/Ni;
                        double2 e;
                        e.y = sincos(-2*M_PI*u/(1<<i),&e.x);
                        double2 z = src[x+(u+1)*Ni];
                        double2 c = (double2)(z.x*e.x-z.y*e.y,z.x*e.y+z.y*e.x);
                        dst[x] = src[x+u*Ni] + c;
                    ",
                    needed: vec![],
                }),
                ]
        },
        ].into_iter().map(|a| (a.name,a)).collect()
}

#[allow(non_snake_case)]
fn C(n: usize, k: usize) -> usize {
    if k==0 || k==n {
        1
    } else {
        C(n-1,k-1) + C(n-1,k)
    }
}

// Only for D1
pub fn moments_to_cumulants<'a>(moments: &'a [f64]) -> Vec<f64> {
    let len = moments.len();
    let mut cumulants = vec![0.0; len];
    for n in 0..len {
        let mut m = 0.0;
        for k in 0..n {
            m += C(n-1,k-1) as f64*cumulants[k]*moments[n-k];
        }
        cumulants[n] = moments[n] - m;
    }

    cumulants
}
