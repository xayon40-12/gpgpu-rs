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

macro_rules! callback_gen {
    ($h:ident, $dim:ident, $dimdir:pat, $bufs:ident, $other:pat, $body:tt) => {
        Rc::new(|$h: &mut Handler, $dim: Dim, $dimdir: &[DimDir], $bufs: &[&str], $other: Option<&dyn Any>| $body)
    };
}

macro_rules! algo_gen {
    (center $name:literal, $Eb:ident|$Tb:ident $Ep:ident|$Ep_3:ident|$Tp:ident, $src:literal) => {
        algo_gen!(iner a $name, $src, $Eb|$Tb $Ep|$Ep_3|$Tp);
    };
    (log $name:literal, $Eb:ident|$Tb:ident $Ep:ident|$Ep_3:ident|$Tp:ident, $src:literal) => {
        algo_gen!(iner a|b $name, $src, $Eb|$Tb $Ep|$Ep_3|$Tp);
    };
    (iner $($type:meta)|+ $name:literal, $src:literal, $Eb:ident|$Tb:ident $Ep:ident|$Ep_3:ident|$Tp:ident) => {
        #[allow(unused)]
        Algorithm {
            name: $name,
            callback: Rc::new(|h: &mut Handler, dim: Dim, dirs: &[DimDir], bufs: &[&str], _: Option<&dyn Any>| {
                algo_gen!(init $($type)|+, bufs, $name, src tmp dst);
                let len = |spacing,x| x/spacing + if x%spacing > 1 { 1 } else { 0 };
                let (d, dims, size): (Box<dyn Fn(usize, DimDir) -> Dim>, _, _) = match dim {
                    D1(x) => {
                        if x<=1 { panic!("Each given dim in algorithm \"{}\" must be strictly greater than 1, given (x: {})", $name, x); }
                        (Box::new(move |s,dir| match dir {
                            X => D1(len(s,x)),
                            _ => panic!("Direction {:?} does not exist for in dimension {:?} for algorithm \"{}\""),
                        }),
                        vec![(x,X)],
                        [x as $Tp,1,1])
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
                        }).collect(),
                        [x as $Tp,y as $Tp,1])
                    },
                    _ => panic!("not")
                };
                algo_gen!(doing $($type)|+ $name,$Eb|$Tb $Ep|$Ep_3|$Tp, h, d dims size, dirs, src tmp dst);
                Ok(None)
            }),
            needed: algo_gen!(nedeed $($type)|+,
                NewKernel(Kernel {
                    name: concat!("algo_",$name),
                    args: vec![KC::ConstBuffer("src",EmT::$Eb),KC::Buffer("dst",EmT::$Eb),KC::Param("s",EmT::$Ep),KC::Param("size",EmT::$Ep_3),KC::Param("dim",EmT::U8)],
                    src: algo_gen!(src $($type)|+ $src),
                    needed: vec![],
                })
            )
        }
    };
    (nedeed $a:meta, $kern:expr) => {
        vec![$kern]
    };
    (nedeed $a:meta|$b:meta, $kern:expr) => {
        vec![$kern,KernelName("move")]
    };
    (init $a:meta, $bufs:ident, $name:literal, $src:ident $tmp:ident $dst:ident) => {
        bufs!($bufs, $name, 2,
            $src
            $dst
        );
    };
    (init $a:meta|$b:meta, $bufs:ident, $name:literal, $src:ident $tmp:ident $dst:ident) => {
        bufs!($bufs, $name, 3,
            $src
            $tmp
            $dst
        );
    };
    (doing $a:meta $name:literal, $Eb:ident|$Tb:ident $Ep:ident|$Ep_3:ident|$Tp:ident, $h:ident, $d:ident $dims:ident $size:ident, $dirs:ident, $src:ident $tmp:ident $dst:ident) => {
        for (x,dir) in $dims {
            $h.run_arg(concat!("algo_",$name), $d(1,dir), &[BufArg(&$src,"src"),BufArg(&$dst,"dst"),Param("dim",U8(dir as u8))])?;
        }
    };
    (doing $a:meta|$b:meta $name:literal, $Eb:ident|$Tb:ident $Ep:ident|$Ep_3:ident|$Tp:ident, $h:ident, $d:ident $dims:ident $size:ident, $dirs:ident, $src:ident $tmp:ident $dst:ident) => {
        $h.copy::<$Tb>($src,$tmp)?;
        for (x,dir) in $dims {
            let mut spacing = 2;
            $h.run_arg(concat!("algo_",$name), $d(spacing,dir), &[Param("s",$Ep(spacing as $Tp)),BufArg(&$tmp,"src"),BufArg(&$tmp,"dst"),Param("size",$Ep_3($size.into())),Param("dim",U8(dir as u8))])?;
            while spacing<x {
                spacing *= 2;
                $h.run_arg(concat!("algo_",$name), $d(spacing,dir), &[Param("s",$Ep(spacing as $Tp))])?;
            }
        }
        let dims = $dirs.iter().fold($size.clone(), |mut a,dir| { a[*dir as usize] = 1; a });
        $h.run_arg("move",dims.into(),&[BufArg(&$tmp,"src"),BufArg(&$dst,"dst"),Param("size",$Ep_3($size.into())),Param("offset",U32(0))])?
    };
    (src $a:meta $src:literal) => {
        concat!("
            y *= x_size;
            z *= x_size*y_size;
            long xp = ((long[3]){x_size/2,x,x})[dim];
            long yp = ((long[3]){y,x_size*(y_size/2),y})[dim];
            long zp = ((long[3]){z,z,x_size*y_size*(z_size/2)})[dim];
        ",$src)
    };
    (src $a:meta|$b:meta $src:literal) => {
        concat!("
            x *= ((long[3]){s,1,1})[dim];
            y *= size.x*((long[3]){1,s,1})[dim];
            z *= size.x*size.y*((long[3]){1,1,s})[dim];
            long xp = x+((long[3]){s/2,0,0})[dim];
            long yp = y+((long[3]){0,size.x*s/2,0})[dim];
            long zp = z+((long[3]){0,0,size.x*size.y*s/2})[dim];
        ",$src)
    };

}

pub fn algorithms<'a>() -> HashMap<&'static str,Algorithm<'a>> {
    vec![
        // sum each elements. With D1 apply on whole buffer, with D2 apply on all y sub-buffers of
        // size x (where x and y are the first and second dimensions).
        algo_gen!(log "sum",F64|f64 U32|U32_3|u32,"dst[x+y+z] = src[x+y+z]+src[xp+yp+zp];"),
        // find min each elements. With D1 apply+z on whole buffer, with D2 apply+z on all y+z sub-buffers of
        // size x (where x and y+z are the first and second dimensions).
        algo_gen!(log "min",F64|f64 U32|U32_3|u32,"dst[x+y+z] = (src[x+y+z]<src[xp+yp+zp])?src[x+y+z]:src[xp+yp+zp];"),
        // find max each elements. With D1 apply+z on whole buffer, with D2 apply+z on all y+z sub-buffers of
        // size x (where x and y+z are the first and second dimensions).
        algo_gen!(log "max",F64|f64 U32|U32_3|u32,"dst[x+y+z] = (src[x+y+z]>src[xp+yp+zp])?src[x+y+z]:src[xp+yp+zp];"),
        // Compute correlation. With D1 apply+z on whole buffer, with D2 apply+z on all y+z sub-buffers of
        // size x (where x and y are the first and second dimensions).
        algo_gen!(center "correlation",F64|f64 U32|U32_3|u32,"dst[x+y+z] = src[x+y+z]*src[xp+yp+zp];"),
        // Compute moments. With D1 apply on whole buffer, with D2 apply on all y sub-buffers of
        // size x (where x and y are the first and second dimensions).
        Algorithm {
            name: "moments",
            callback: callback_gen!(h,dim,dirs,bufs,param, {
                bufs!(bufs, "moments", 5..6,
                    src
                    tmp
                    sum
                    dstsum
                    dst
                );
                let num: u32 = if let Some(p) = param {
                    *p.downcast_ref().expect("Optional parameter of \"moment\" algorithm must be U32.")
                } else {
                    4
                };
                if num < 1 { panic!("There must be at least one moment calculated in \"moments\" algorithm."); }

                let (l,size) = match dim {
                    D1(x) => (x,[x as u32,1,1]),
                    D2(x,y) => (x*y,[x as u32,y as u32,1]),
                    D3(x,y,z) => (x*y*z,[x as u32,y as u32,z as u32])
                };

                h.run_algorithm("sum",dim,dirs,&[src,sum,dstsum],None)?;
                let sumsize = dirs.iter().fold(size.clone(), |mut a,dir| { a[*dir as usize] = 1; a });
                let sumlen = sumsize[0]*sumsize[1]*sumsize[2];
                h.set_arg("smove",&[BufArg(&dstsum,"src"),BufArg(&dst,"dst"),Param("size",U32_3([num,sumlen,1].into())),Param("offset",U32(0))])?;
                h.run("smove",D2(1,sumlen as usize))?;
                if num >= 1 {
                    h.run_arg("times",D1(l),&[BufArg(&src,"a"),BufArg(&src,"b"),BufArg(&tmp,"dst")])?;
                    h.run_algorithm("sum",dim,dirs,&[tmp,sum,dstsum],None)?;
                    h.run_arg("smove",D2(1,sumlen as usize),&[Param("offset",U32(1))])?;
                    h.set_arg("times",&[BufArg(&tmp,"a")])?;
                }
                for i in 2..num {
                    h.run("times",D1(l))?;
                    h.run_algorithm("sum",dim,dirs,&[tmp,sum,dstsum],None)?;
                    h.run_arg("smove",D2(1,sumlen as usize),&[Param("offset",U32(i as u32))])?;
                }
                h.run_arg("cdivides",D1((num*sumlen) as usize),&[BufArg(&dst,"src"),Param("c",F64((l/sumlen as usize) as f64)),BufArg(&dst,"dst")])?;

                Ok(None)
            }),
            needed: vec![
                KernelName("times"),
                KernelName("cdivides"),
                KernelName("smove"),
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
