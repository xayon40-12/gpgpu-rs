use crate::{Handler,kernels::{Kernel,SKernel}};
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
pub struct SAlgorithm {
    pub name: String,
    pub callback: Callback,
    pub needed: Vec<SNeeded>,
}

impl<'a> From<&Algorithm<'a>> for SAlgorithm {
    fn from(f: &Algorithm<'a>) -> Self {
        SAlgorithm {
            name: f.name.into(),
            callback: f.callback.clone(),
            needed: f.needed.iter().map(|i| i.into()).collect(),
        }
    }
}

#[derive(Clone)]
pub enum Needed<'a> { //TODO use one SC for each &'a str
    KernelName(&'a str),
    AlgorithmName(&'a str),
    NewKernel(Kernel<'a>)
}
use Needed::*;
use crate::functions::Needed::*;

#[derive(Clone)]
pub enum SNeeded {
    KernelName(String),
    AlgorithmName(String),
    NewKernel(SKernel)
}

impl<'a> From<&Needed<'a>> for SNeeded {
    fn from(n: &Needed<'a>) -> Self {
        match n {
            Needed::KernelName(n) => SNeeded::KernelName((*n).into()),
            Needed::AlgorithmName(n) => SNeeded::AlgorithmName((*n).into()),
            Needed::NewKernel(k) => SNeeded::NewKernel(k.into()),
        }
    }
}

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

macro_rules! center {
    (nedeed $kern:expr) => {
        vec![$kern]
    };
    (kern_needed) => {
        vec![]
    };
    (doing $name:literal, $Eb:ident|$Tb:ident $Ebp:ident $Ep:ident|$Ep_3:ident, $h:ident, $dim:ident , $dirs:ident, $bufs:ident) => {
        bufs!($bufs, $name, 2,
            src
            dst
        );
        let mut dirs = [0u8,0,0];
        if $dirs.len() < 1 { panic!("There must be at least one direction given for algorithm \"{}\"", concat!("algo_",$name)); }
        $dirs.iter().for_each(|d| dirs[*d as usize] = 1);
        $h.run_arg(concat!("algo_",$name), $dim, &[BufArg(&src,"src"),BufArg(&dst,"dst"),Param("dir",U8_3(dirs.into()))])?;
    };
    (src $src:literal) => {
        concat!("
            y *= x_size;
            z *= x_size*y_size;
            uint xp = (uint[2]){x,x_size/2}[dir.x];
            uint yp = (uint[2]){y,x_size*(y_size/2)}[dir.y];
            uint zp = (uint[2]){z,x_size*y_size*(z_size/2)}[dir.z];
            uint id = x+y+z;
            uint idp = xp+yp+zp;
        ",$src)
    };
    (args $Eb:ident|$Tb:ident $Ep:ident|$Ep_3:ident) => {
        vec![KC::Buffer("src",EmT::$Eb),KC::Buffer("dst",EmT::$Eb),KC::Param("dir",EmT::U8_3)]
    };
}

macro_rules! logreduce {
    (nedeed $kern:expr) => {
        vec![$kern,KernelName("move")]
    };
    (kern_needed) => {
        vec![]
    };
    (doing $name:literal, $Eb:ident|$Tb:ident $Ebp:ident $Ep:ident|$Ep_3:ident, $h:ident, $dim:ident, $dirs:ident, $bufs:ident) => {
        bufs!($bufs, $name, 3,
            src
            tmp
            dst
        );

        let len = |spacing,x| x/spacing + if (x/spacing)*spacing+spacing/2 < x { 1 } else { 0 };
        let (d, dims, size): (Box<dyn Fn(usize, DimDir, [usize;3]) -> Dim>, _, _) = match $dim {
            D1(x) => {
                if x<=1 { panic!("Each given dim in algorithm \"{}\" must be strictly greater than 1, given (x: {})", $name, x); }
                (Box::new(move |s,dir,_| match dir {
                    X => D1(len(s,x)),
                    _ => panic!("Direction \"{:?}\" is not accessible with dimension \"{:?}\" in algorithm \"{}\"", dir, $dim, $name),
                }),
                vec![(x,X)],
                [x as _,1,1])
            },
            D2(x,y) => {
                if x<=1 || y<=1 { panic!("Each given dim in algorithm \"{}\" must be strictly greater than 1, given (x: {}, y: {})", $name, x, y); }
                (Box::new(move |s,dir,size| match dir {
                    X => D2(len(s,x),size[1]),
                    Y => D2(size[0],len(s,y)),
                    _ => panic!("Direction {:?} does not exist for in dimension {:?} for algorithm \"{}\"",dir,$dim,$name),
                }),
                $dirs.into_iter().map(|dir| match dir {
                    X => (x,X),
                    Y => (y,Y),
                    _ => panic!("Direction \"{:?}\" is not accessible with dimension \"{:?}\" in algorithm \"{}\"", dir, $dim, $name)
                }).collect(),
                [x as _,y as _,1])
            },
            D3(x,y,z) => {
                if x<=1 || y<=1 || z<=1 { panic!("Each given dim in algorithm \"{}\" must be strictly greater than 1, given (x: {}, y: {}, z: {})", $name, x, y, z); }
                (Box::new(move |s,dir,size| match dir {
                    X => D3(len(s,x),size[1],size[2]),
                    Y => D3(size[0],len(s,y),size[2]),
                    Z => D3(size[0],size[1],len(s,z)),
                }),
                $dirs.into_iter().map(|dir| match dir {
                    X => (x,X),
                    Y => (y,Y),
                    Z => (z,Z),
                }).collect(),
                [x as _,y as _,z as _])
            },
        };

        $h.copy::<$Tb>(src,tmp)?;
        let mut size_reduce = [size[0] as usize, size[1] as usize, size[2] as usize];
        for (x,dir) in dims {
            let mut spacing = 2;
            $h.run_arg(concat!("algo_",$name), d(spacing,dir,size_reduce), &[Param("s",$Ep(spacing as _)),BufArg(&tmp,"src"),BufArg(&tmp,"dst"),Param("size",$Ep_3(size.into())),Param("dir",U8(dir as u8))])?;
            while spacing<x {
                spacing *= 2;
                $h.run_arg(concat!("algo_",$name), d(spacing,dir,size_reduce), &[Param("s",$Ep(spacing as _))])?;
            }
            size_reduce[dir as usize] = 1;
        }
        let dims = $dirs.iter().fold(size.clone(), |mut a,dir| { a[*dir as usize] = 1; a });
        $h.run_arg("move",dims.into(),&[BufArg(&tmp,"src"),BufArg(&dst,"dst"),Param("size",$Ep_3(size.into())),Param("offset",U32(0))])?
    };
    (src $src:literal) => {
        concat!("
            x *= (uint[3]){s,1,1}[dir];
            y *= size.x*(uint[3]){1,s,1}[dir];
            z *= size.x*size.y*(uint[3]){1,1,s}[dir];
            uint xp = x+(uint[3]){s/2,0,0}[dir];
            uint yp = y+(uint[3]){0,size.x*s/2,0}[dir];
            uint zp = z+(uint[3]){0,0,size.x*size.y*s/2}[dir];
            uint id = x+y+z;
            uint idp = xp+yp+zp;
        ",$src)
    };
    (args $Eb:ident|$Tb:ident $Ep:ident|$Ep_3:ident) => {
        vec![KC::Buffer("src",EmT::$Eb),KC::Buffer("dst",EmT::$Eb),KC::Param("s",EmT::$Ep),KC::Param("size",EmT::$Ep_3),KC::Param("dir",EmT::U8)]
    };
}

macro_rules! log {
    (nedeed $kern:expr) => {
        vec![$kern,KernelName("cdivides")]
    };
    (kern_needed) => {
        vec![FuncName("c_exp".into()),FuncName("c_times".into())]
    };
    (doing $name:literal, $Eb:ident|$Tb:ident $Ebp:ident $Ep:ident|$Ep_3:ident, $h:ident, $dim:ident, $dirs:ident, $bufs:ident) => {
        bufs!($bufs, $name, 3,
            src
            tmp
            dst
        );
        let (dims, size) = match $dim {
            D1(x) => {
                if x<=1 { panic!("Each given dim in algorithm \"{}\" must be strictly greater than 1, given (x: {})", $name, x); }
                ($dirs.into_iter().map(|dir| match dir {
                    X => (x,X),
                    _ => panic!("Direction \"{:?}\" is not accessible with dimension \"{:?}\" in algorithm \"{}\"", dir, $dim, $name)
                }).collect::<Vec<_>>(),
                [x as _,1,1])
            },
            D2(x,y) => {
                if x<=1 || y<=1 { panic!("Each given dim in algorithm \"{}\" must be strictly greater than 1, given (x: {}, y: {})", $name, x, y); }
                ($dirs.into_iter().map(|dir| match dir {
                    X => (x,X),
                    Y => (y,Y),
                    _ => panic!("Direction \"{:?}\" is not accessible with dimension \"{:?}\" in algorithm \"{}\"", dir, $dim, $name),
                }).collect(),
                [x as _,y as _,1])
            },
            D3(x,y,z) => {
                if x<=1 || y<=1 || z<=1 { panic!("Each given dim in algorithm \"{}\" must be strictly greater than 1, given (x: {}, y: {}, z: {})", $name, x, y, z); }
                ($dirs.into_iter().map(|dir| match dir {
                    X => (x,X),
                    Y => (y,Y),
                    Z => (z,Z),
                }).collect(),
                [x as _,y as _,z as _])
            },
        };
        let l = size[0]*size[1]*size[2];
        let mut j = 0;
        let mut lnx = 0;
        while (1<<lnx) < $dirs.iter().fold(1,|a,i| a*size[*i as usize]) { lnx += 1; }
        let m = lnx%2;
        let mut begi = m+1;
        let sd = [&tmp,&dst,&src];
        for (x,dir) in dims {
            if !x.is_power_of_two() { panic!("In algorithm \"{}\", dimensions must be power of two.",$name); }

            let mut i = 1; j += 1;
            $h.run_arg(concat!("algo_",$name),$dim,&[BufArg(sd[(j+m)%2+begi],"src"),BufArg(sd[(j+m+1)%2],"dst"),Param("i",$Ep(i as _)),Param("dir",U8(dir as u8))]);
            while (1<<i) < x {
                i += 1; j += 1;
                $h.run_arg(concat!("algo_",$name),$dim,&[BufArg(sd[(j+m)%2],"src"),BufArg(sd[(j+m+1)%2],"dst"),Param("i",$Ep(i as _))]);
            }
            $h.run_arg("cdivides",D1((l*2) as _),&[BufArg(sd[(j+m+1)%2],"src"),Param("c",$Ebp(x as _)),BufArg(&sd[(j+m+1)%2],"dst")]);
            begi = 0;
        }
    };
    (src $src:literal) => {
        concat!("
            uint Ni = ((uint[3]){x_size,y_size,z_size}[dir])>>i;
            uint u  = ((uint[3]){x,y,z}[dir])/Ni;
            uint uNi = u*Ni;

            uint xx  = (x+(uint[3]){uNi,0,0}[dir])%x_size;
            uint yy  = (y+(uint[3]){0,uNi,0}[dir])%y_size;
            uint zz  = (z+(uint[3]){0,0,uNi}[dir])%z_size;

            uint xp = (xx+(uint[3]){Ni,0,0}[dir])%x_size;
            uint yp = (yy+(uint[3]){0,Ni,0}[dir])%y_size;
            uint zp = (zz+(uint[3]){0,0,Ni}[dir])%z_size;

            y *= x_size;
            z *= x_size*y_size;
            yy *= x_size;
            zz *= x_size*y_size;
            yp *= x_size;
            zp *= x_size*y_size;

            uint id = x+y+z;
            uint ida = xx+yy+zz;
            uint idb = xp+yp+zp;
        ",$src)
    };
    (args $Eb:ident|$Tb:ident $Ep:ident|$Ep_3:ident) => {
        vec![KC::Buffer("src",EmT::$Eb),KC::Buffer("dst",EmT::$Eb),KC::Param("i",EmT::$Ep),KC::Param("dir",EmT::U8)]
    };
}

macro_rules! algo_gen {
    ($algo_macro:ident $name:literal, $Eb:ident|$Tb:ident $Ep:ident|$Ep_3:ident, $src:literal) => {
        algo_gen!($algo_macro $name, $Eb|$Tb nop $Ep|$Ep_3, $src);
    };
    ($algo_macro:ident $name:literal, $Eb:ident|$Tb:ident $Ebp:ident $Ep:ident|$Ep_3:ident, $src:literal) => {
        #[allow(unused)]
        Algorithm {
            name: $name,
            callback: Rc::new(|h: &mut Handler, dim: Dim, dirs: &[DimDir], bufs: &[&str], _: Option<&dyn Any>| {
                $algo_macro!(doing $name, $Eb|$Tb $Ebp $Ep|$Ep_3, h, dim, dirs, bufs);
                Ok(None)
            }),
            needed: $algo_macro!(nedeed
                NewKernel(Kernel {
                    name: concat!("algo_",$name),
                    args: $algo_macro!(args $Eb|$Tb $Ep|$Ep_3),
                    src: $algo_macro!(src $src),
                    needed: $algo_macro!(kern_needed),
                })
            )
        }
    };

}

pub fn algorithms() -> HashMap<&'static str,Algorithm<'static>> {
    vec![
        // sum each elements.
        algo_gen!(logreduce "sum",F64|f64 U32|U32_3,"dst[id] = src[id]+src[idp];"),
        // find min value.
        algo_gen!(logreduce "min",F64|f64 U32|U32_3,"dst[id] = (src[id]<src[idp])?src[id]:src[idp];"),
        // find max value.
        algo_gen!(logreduce "max",F64|f64 U32|U32_3,"dst[id] = (src[id]>src[idp])?src[id]:src[idp];"),
        // Compute correlation.
        algo_gen!(center "correlation",F64|f64 U32|U32_3,"dst[id] = src[id]*src[idp];"),
        // Compute the FFT
        algo_gen!(log "FFT",F64_2|Double2 F64 U32|U32_3,"
            dst[id] = src[ida] + c_times(src[idb],c_exp(-2*M_PI*u/(1<<i)));
        "),
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
pub fn moments_to_cumulants(moments: &[f64]) -> Vec<f64> {
    let len = moments.len();
    let mut cumulants = vec![0.0; len];
    for n in 0..len {
        let mut m = 0.0;
        for k in 0..n {
            m += C(n,k) as f64*cumulants[k]*moments[n-k-1];
        }
        cumulants[n] = moments[n] - m;
    }

    cumulants
}
