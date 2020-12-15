use crate::{Handler,kernels::{Kernel,SKernel}};
use crate::Dim::{self,*};
use crate::descriptors::KernelArg::*;
use crate::descriptors::Types::*;
use std::collections::HashMap;
use std::rc::Rc;
use crate::descriptors::KernelConstructor::*;
use crate::descriptors::ConstructorTypes::*;
use std::any::Any;
use crate::DimDir::{*,self};

pub enum AlgorithmParam<'a> {
    Mut(&'a mut dyn Any),
    Ref(&'a dyn Any),
    Nothing,
}
use AlgorithmParam::*;

impl<'a> AlgorithmParam<'a> {
    pub fn downcast_ref<'b,T: 'static>(&self, error_msg: &'b str) -> &T {
        if let Ref(r) = self {
            r.downcast_ref().expect(error_msg)
        } else {
            panic!("{}",error_msg)
        }
    }

    pub fn downcast_mut<'b,T: 'static>(&mut self, error_msg: &'b str) -> &mut T {
        if let Mut(r) = self {
            r.downcast_mut().expect(error_msg)
        } else {
            panic!("{}",error_msg)
        }
    }
}

//Fn(handler: &mut Handler, dim: Dim, dim_dir: &[DimDir], buf_names: Vec<String>, other_args: Option<&dyn Any>)
pub type Callback = Rc<dyn Fn(&mut Handler, Dim, &[DimDir], &[&str], AlgorithmParam) -> crate::Result<Option<Box<dyn Any>>>>;

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

#[derive(Clone,Debug)]
pub enum Needed<'a> { //TODO use one SC for each &'a str
    KernelName(&'a str),
    AlgorithmName(&'a str),
    NewKernel(Kernel<'a>)
}
use Needed::*;
use crate::functions::Needed::*;

#[derive(Clone,Debug)]
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

#[derive(Clone,Copy,Debug)]
pub enum Packing {
    Packed([u32;4]),
    Unpacked([u32;4]),
}

impl Packing {
    pub fn new(size: [u32;4], packed: bool) -> Packing {
        if packed {
            Packing::Packed(size)
        } else {
            Packing::Unpacked(size)
        }
    }
}

#[derive(Clone,Copy,Debug)]
pub struct Window {
    pub offset: usize,
    pub len: usize,
}

#[derive(Clone,Debug)]
pub struct ReduceParam {
    pub vect_dim: u32,
    pub dst_size: Option<Packing>,
    pub window: Option<Vec<Window>>,
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

macro_rules! param {
    (logreduce $other:ident, $name:literal) => {
        if let Ref(o) = $other {
            if let Some(ap) = o.downcast_ref::<ReduceParam>() {
                let ap = ap.clone();
                if ap.vect_dim<1 { panic!("Vectorial dimension given as parameter of algorithm \"{}\" must be greater or equal to 1.",$name) }
                ap
            } else {
                panic!("The optional parameter of algorithm \"{}\" must be of type ReduceParam.",$name)
            }
        } else {
            ReduceParam { vect_dim: 1, dst_size: None, window: None }
        }
    };
    (log $other:ident, $name:literal) => {
        if let Ref(o) = $other {
            if let Some(&w) = o.downcast_ref::<u32>() {
                if w<1 { panic!("Vectorial dimension given as parameter of algorithm \"{}\" must be greater or equal to 1.",$name) }
                w
            } else {
                panic!("The optional parameter of algorithm \"{}\" must be of type u32.",$name)
            }
        } else {
            1
        }
    };
    (center $other:ident, $name:literal) => {
        param!(log $other, $name)
    }
}

macro_rules! callback_gen {
    ($h:ident, $dim:ident, $dimdir:pat, $bufs:ident, $other:pat, $body:tt) => {
        Rc::new(|$h: &mut Handler, $dim: Dim, $dimdir: &[DimDir], $bufs: &[&str], $other: AlgorithmParam| $body)
    };
}

macro_rules! center {
    (nedeed $kern:expr) => {
        vec![$kern]
    };
    (kern_needed) => {
        vec![]
    };
    (doing $name:literal, $Eb:ident|$Ebp:ident $Ep:ident|$Ep_:ident, $h:ident, $dim:ident , $dirs:ident, $bufs:ident, $other:ident) => {
        bufs!($bufs, $name, 2,
            src
            dst
        );
        let w = param!(center $other, $name);
        let mut dirs = [0u8,0,0,0];
        if $dirs.len() < 1 { panic!("There must be at least one direction given for algorithm \"{}\"", concat!("algo_",$name)); }
        $dirs.iter().for_each(|d| dirs[*d as usize] = 1);
        $h.run_arg(concat!("algo_",$name), $dim, &[BufArg(&src,"src"),BufArg(&dst,"dst"),Param("dir",U8_4(dirs.into())),Param("w",U8(w as u8))])?;
    };
    (src $src:literal) => {
        concat!("
    y *= x_size;
    z *= x_size*y_size;
    uint xp = (uint[2]){x,x_size/2}[dir.x];
    uint yp = (uint[2]){y,x_size*(y_size/2)}[dir.y];
    uint zp = (uint[2]){z,x_size*y_size*(z_size/2)}[dir.z];
    uint id = w*(x+y+z);
    uint idp = w*(xp+yp+zp);
    for(int _iw_ = 0; _iw_<w; _iw_++) {
        ",$src,"
        id++; idp++;
    }")
    };
    (args $CEb:ident|$CEp:ident|$CEp_:ident) => {
        vec![KCBuffer("src",$CEb),KCBuffer("dst",$CEb),KCParam("dir",CU8_4),KCParam("w",CU8)]
    };
}

macro_rules! logreduce {
    (nedeed $kern:expr) => {
        vec![$kern,KernelName("dmove"),KernelName("rdmove"),KernelName("move"),KernelName("omove")]
    };
    (kern_needed) => {
        vec![]
    };
    (doing $name:literal, $Eb:ident|$Ebp:ident $Ep:ident|$Ep_:ident, $h:ident, $dim:ident, $dirs:ident, $bufs:ident, $other:ident) => {
        bufs!($bufs, $name, 3,
            src
            tmp
            dst
        );
        let ap = param!(logreduce $other, $name);
        let w = ap.vect_dim;


        let mut dim: [usize;3] = $dim.into();
        let mut offs = [0;4];
        if let Some(window) = &ap.window {
            if window.len() != $dirs.len() { panic!("The window dimensions must be in the same number as directions and in the same order.") }
            for (Window{offset: off,len},dir) in window.iter().zip($dirs.iter()) {
                let d = *dir as usize;
                if *len > dim[d] { panic!("Each len in window must be less or equal to the corresponding dim.") }
                if off+len > dim[d] { panic!("The window must be inside the range of the corresponding dim (offset+len<=dim).") }
                dim[d] = *len;
                offs[d] = *off as u32;
            }
        }
        let len = |spacing,x| x/spacing + if (x/spacing)*spacing+spacing/2 < x { 1 } else { 0 };
        let (d, dims, mut size): (Box<dyn Fn(usize, DimDir, [usize;3]) -> Dim>, _, _) = match dim.into() {
            D1(x) => {
                if x<1 { panic!("Each given dim in algorithm \"{}\" must be strictly greater than 1, given (x: {})", $name, x); }
                (Box::new(move |s,dir,_| match dir {
                    X => D1(len(s,x)),
                    _ => panic!("Direction \"{:?}\" is not accessible with dimension \"{:?}\" in algorithm \"{}\"", dir, $dim, $name),
                }),
                vec![(x,X)],
                [x as _,1,1,0])
            },
            D2(x,y) => {
                if x<1 || y<1 { panic!("Each given dim in algorithm \"{}\" must be strictly greater than 1, given (x: {}, y: {})", $name, x, y); }
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
                [x as _,y as _,1,0])
            },
            D3(x,y,z) => {
                if x<1 || y<1 || z<1 { panic!("Each given dim in algorithm \"{}\" must be strictly greater than 1, given (x: {}, y: {}, z: {})", $name, x, y, z); }
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
                [x as _,y as _,z as _,0])
            },
        };

        if ap.window.is_some() {
            let size: [usize; 3] = $dim.into();
            let size = [size[0] as u32, size[1] as u32];
            $h.run_arg("omove", dim.into(), &[BufArg(src,"src"),BufArg(tmp,"dst"),Param("size",size.into()),Param("offsets",offs.into()),Param("vect_dim",(w as u32).into())])?;
        } else {
            $h.copy(src,tmp)?;
        }
        let mut size_reduce = [size[0] as usize, size[1] as usize, size[2] as usize];
        for (x,dir) in dims {
            if x == 1 { continue; }
            let mut spacing = 2;
            $h.run_arg(concat!("algo_",$name), d(spacing,dir,size_reduce), &[Param("s",$Ep(spacing as _)),BufArg(&tmp,"src"),BufArg(&tmp,"dst"),Param("size",$Ep_(size.into())),Param("dir",U8(dir as u8)),Param("w",U8(w as u8))])?;
            while spacing<x {
                spacing *= 2;
                $h.run_arg(concat!("algo_",$name), d(spacing,dir,size_reduce), &[Param("s",$Ep(spacing as _))])?;
            }
            size_reduce[dir as usize] = 1;
        }
        let mut dims = $dirs.iter().fold([size[0],size[1],size[2]], |mut a,dir| { a[*dir as usize] = 1; a });
        dims[0] *= w as u32;
        size[0] *= w as u32;
        if let Some(dst_size) =  ap.dst_size {
            use Packing::*;
            match dst_size {
                Packed(dst_size) =>
                    $h.run_arg("dmove",dims.into(),&[BufArg(&tmp,"src"),BufArg(&dst,"dst"),Param("size",$Ep_(size.into())),Param("dst_size",dst_size.into()),Param("vect_dim",(w as u32).into())])?,
                Unpacked(dst_size) => 
                    $h.run_arg("rdmove",dims.into(),&[BufArg(&tmp,"src"),BufArg(&dst,"dst"),Param("size",$Ep_(size.into())),Param("dst_size",dst_size.into()),Param("vect_dim",(w as u32).into())])?,
            }
        } else {
            $h.run_arg("move",dims.into(),&[BufArg(&tmp,"src"),BufArg(&dst,"dst"),Param("size",$Ep_(size.into())),Param("offset",U32(0)),Param("vect_dim",(w as u32).into())])?
        }
    };
    (src $src:literal) => {
        concat!("
    x *= (uint[3]){s,1,1}[dir];
    y *= size.x*(uint[3]){1,s,1}[dir];
    z *= size.x*size.y*(uint[3]){1,1,s}[dir];
    uint xp = x+(uint[3]){s/2,0,0}[dir];
    uint yp = y+(uint[3]){0,size.x*s/2,0}[dir];
    uint zp = z+(uint[3]){0,0,size.x*size.y*s/2}[dir];
    uint id = w*(x+y+z);
    uint idp = w*(xp+yp+zp);
    for(int _iw_ = 0; _iw_<w; _iw_++) {
        ",$src,"
        id++; idp++;
    }")
    };
    (args $CEb:ident|$CEp:ident|$CEp_:ident) => {
        vec![KCBuffer("src",$CEb),KCBuffer("dst",$CEb),KCParam("s",$CEp),KCParam("size",$CEp_),KCParam("dir",CU8),KCParam("w",CU8)]
    };
}

macro_rules! log {
    (nedeed $kern:expr) => {
        vec![$kern,KernelName("cdivides")]
    };
    (kern_needed) => {
        vec![FuncName("c_exp".into()),FuncName("c_times".into())]
    };
    (doing $name:literal, $Eb:ident|$Ebp:ident $Ep:ident|$Ep_:ident, $h:ident, $dim:ident, $dirs:ident, $bufs:ident, $other:ident) => {
        bufs!($bufs, $name, 3,
            src
            tmp
            dst
        );
        let w = param!(log $other, $name);
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
            $h.run_arg(concat!("algo_",$name),$dim,&[BufArg(sd[(j+m)%2+begi],"src"),BufArg(sd[(j+m+1)%2],"dst"),Param("i",$Ep(i as _)),Param("dir",U8(dir as u8)),Param("w",U8(w as u8))]);
            while (1<<i) < x {
                i += 1; j += 1;
                $h.run_arg(concat!("algo_",$name),$dim,&[BufArg(sd[(j+m)%2],"src"),BufArg(sd[(j+m+1)%2],"dst"),Param("i",$Ep(i as _))]);
            }
            $h.run_arg("cdivides",D1((l*w as usize*$Eb(Default::default()).len()) as _),&[BufArg(sd[(j+m+1)%2],"src"),Param("c",$Ebp(x as _)),BufArg(&sd[(j+m+1)%2],"dst")]);
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

    uint id = w*(x+x_size*(y+y_size*z));
    uint ida = w*(xx+x_size*(yy+y_size*zz));
    uint idb = w*(xp+x_size*(yp+y_size*zp));
    for(int _iw_ = 0; _iw_<w; _iw_++) {
        ",$src,"
        id++; ida++; idb++;
    }")
    };
    (args $CEb:ident|$CEp:ident|$CEp_:ident) => {
        vec![KCBuffer("src",$CEb),KCBuffer("dst",$CEb),KCParam("i",$CEp),KCParam("dir",CU8),KCParam("w",CU8)]
    };
}

macro_rules! algo_gen {
    ($algo_macro:ident $name:literal, $CEb:ident|$Eb:ident $CEp:ident|$Ep:ident $CEp_:ident|$Ep_:ident, $src:literal) => {
        algo_gen!($algo_macro $name, $CEb|$Eb|nop $CEp|$Ep $CEp_|$Ep_, $src);
    };
    ($algo_macro:ident $name:literal, $CEb:ident|$Eb:ident|$Ebp:ident $CEp:ident|$Ep:ident $CEp_:ident|$Ep_:ident, $src:literal) => {
        #[allow(unused)]
        Algorithm {
            name: $name,
            callback: Rc::new(|h: &mut Handler, dim: Dim, dirs: &[DimDir], bufs: &[&str], option: AlgorithmParam| {
                $algo_macro!(doing $name, $Eb|$Ebp $Ep|$Ep_, h, dim, dirs, bufs, option);
                Ok(None)
            }),
            needed: $algo_macro!(nedeed
                NewKernel(Kernel {
                    name: concat!("algo_",$name),
                    args: $algo_macro!(args $CEb|$CEp|$CEp_),
                    src: $algo_macro!(src $src),
                    needed: $algo_macro!(kern_needed),
                })
            )
        }
    };

}

macro_rules! random {
    (philox4x32_10) => {
        "    uint key[2] = {0,x};
    const uint l = 4;
    const uint M = 0xD2511F53;
    const uint M2 = 0xCD9E8D57;
    for(int i = 0;i<10;i++){
        uint hi0 = mul_hi(M,src[x*l+0]);
        uint lo0 = M * src[x*l+0];
        uint hi1 = mul_hi(M2,src[x*l+2]);
        uint lo1 = M2 * src[x*l+2];
        src[x*l+0] = hi1^key[0]^src[x*l+1];
        src[x*l+1] = lo1;
        src[x*l+2] = hi0^key[1]^src[x*l+3];
        src[x*l+3] = lo0;
        key[0] += 0x9E3779B9;
        key[1] += 0xBB67AE85;
    }
"

    };
    (philox2x64_10) => { 
        "    ulong key = x;
    const uint l = 2;
    const ulong M = 0xD2B74407B1CE6E93;
    for(int i = 0;i<10;i++){
        ulong hi = mul_hi(M,src[x*l+0]);
        ulong lo = M * src[x*l+0];
        src[x*l+0] = hi^key^src[x*l+1];
        src[x*l+1] = lo;
        key += 0x9E3779B97F4A7C15;
    }
"
    };
    (philox4x64_10) => {
        "    ulong key[2] = {0,x};
    const uint l = 4;
    const ulong M = 0xD2B74407B1CE6E93;
    const ulong M2 = 0xCA5A826395121157;
    for(int i = 0;i<10;i++){
        ulong hi0 = mul_hi(M,src[x*l+0]);
        ulong lo0 = M * src[x*l+0];
        ulong hi1 = mul_hi(M2,src[x*l+2]);
        ulong lo1 = M2 * src[x*l+2];
        src[x*l+0] = hi1^key[0]^src[x*l+1];
        src[x*l+1] = lo1;
        src[x*l+2] = hi0^key[1]^src[x*l+3];
        src[x*l+3] = lo0;
        key[0] += 0x9E3779B97F4A7C15;
        key[1] += 0xBB67AE8584CAA73B;
    }
"
    };
    ($name:ident, $more:literal) => {
        concat!(random!($name),$more)
    };
}

macro_rules! random_kernels {
    (u64 $name:ident,$type:ident) => {
        vec![NewKernel(Kernel {
            name: stringify!($name),
            args: vec![KCBuffer("src",$type)],
            src: random!($name),
            needed: vec![],
        }),
        NewKernel(Kernel {
            name: concat!(stringify!($name),"_unit"),
            args: vec![KCBuffer("src",$type),KCBuffer("dst",CF64)],
            src: random!($name,"    for(uint i = 0;i<l;i++)
    dst[x*l+i] = (double)(src[x*l+i]>>11)/(1l << 53);"
            ),
            needed: vec![],
        }),
        NewKernel(Kernel {
            name: concat!(stringify!($name),"_normal"),
            args: vec![KCBuffer("src",$type),KCBuffer("dst",CF64)],
            src: random!($name,"    for(uint i = 0;i<l;i+=2) {
    double u1 = (double)(src[x*l+i]>>11)/(1l << 53);
    double u2 = (double)(src[x*l+i+1]>>11)/(1l << 53);
    dst[x*l+i] = sqrt(-2*log(u1))*cos(2*M_PI*u2);
    dst[x*l+i+1] = sqrt(-2*log(u1))*sin(2*M_PI*u2);
}"
            ),
            needed: vec![],
        })]
    };
    (u32 $name:ident,$type:ident) => {
        vec![NewKernel(Kernel {
            name: stringify!($name),
            args: vec![KCBuffer("src",$type)],
            src: random!($name),
            needed: vec![],
        }),
        NewKernel(Kernel {
            name: concat!(stringify!($name),"_unit"),
            args: vec![KCBuffer("src",$type),KCBuffer("dst",CF64)],
            src: random!($name,"    const uint l2 = l/2;
    for(uint i = 0;i<l2;i++)
        dst[x*l2+i] = (double)(((((ulong)src[x*l+i*2])<<32)+src[x*l+i*2+1])>>11)/(1l << 53);"
            ),
            needed: vec![],
        }),
        NewKernel(Kernel {
            name: concat!(stringify!($name),"_normal"),
            args: vec![KCBuffer("src",$type),KCBuffer("dst",CF64)],
            src: random!($name,"    const uint l2 = l/2;
    for(uint i = 0;i<l2;i+=2) {
        double u1 = (double)(((((ulong)src[x*l+i])<<32)+src[x*l+i+1])>>11)/(1l << 53);
        double u2 = (double)(((((ulong)src[x*l+i+2])<<32)+src[x*l+i+3])>>11)/(1l << 53);
        dst[x*l2+i] = sqrt(-2*log(u1))*cos(2*M_PI*u2);
        dst[x*l2+i+1] = sqrt(-2*log(u1))*sin(2*M_PI*u2);
    }"
            ),
            needed: vec![],
        })]
    };
}

#[derive(Debug,Clone,Copy)]
pub enum RandomType {
    Normal,
    Uniform,
    Integer,
}

macro_rules! gen_random {
    ($size:ident $name:ident,$type:ident,$n:literal) => {
        Algorithm {
            name: stringify!($name),
            callback: callback_gen!(h,dim,_dirs,bufs,param, {
                let rtype = if let Ref(p) = param {
                    *p.downcast_ref().expect("The optional paramter for random algorithms must be \"RandomType\"")
                } else {
                    RandomType::Integer
                };

                let dim = if let D1(d) = dim {
                    D1(d/$n)
                } else {
                    panic!("Dim of random algorithms must be D1.")
                };

                match rtype {
                    RandomType::Normal => {
                        if bufs.len() != 2 { panic!("Normal random algorithms takes two buffers (src and dst).") }
                        h.run_arg(concat!(stringify!($name),"_normal"),dim,&[
                            BufArg(bufs[0],"src"),BufArg(bufs[1],"dst")
                        ])?
                    },
                    RandomType::Uniform => {
                        if bufs.len() != 2 { panic!("Uniform random algorithms takes two buffers (src and dst).") }
                        h.run_arg(concat!(stringify!($name),"_unit"),dim,&[
                            BufArg(bufs[0],"src"),BufArg(bufs[1],"dst")
                        ])?
                    },
                    RandomType::Integer => {
                        if bufs.len() != 1 { panic!("Integer random algorithms takes two buffers (src and dst).") }
                        h.run_arg(concat!(stringify!($name)),dim,&[
                            BufArg(bufs[0],"src")
                        ])?
                    },
                }

                Ok(None)
            }),
            needed: random_kernels!($size $name,$type),
        }

    };
}

#[derive(Debug,Clone)]
pub struct MomentsParam {
    pub num: u32,
    pub vect_dim: u32,
    pub packed: bool,
}

pub fn algorithms() -> HashMap<&'static str,Algorithm<'static>> {
    vec![
        // sum each elements.
        algo_gen!(logreduce "sum",CF64|F64 CU32|U32 CU32_4|U32_4,"dst[id] = src[id]+src[idp];"),
        // find min value.
        algo_gen!(logreduce "min",CF64|F64 CU32|U32 CU32_4|U32_4,"dst[id] = (src[id]<src[idp])?src[id]:src[idp];"),
        // find max value.
        algo_gen!(logreduce "max",CF64|F64 CU32|U32 CU32_4|U32_4,"dst[id] = (src[id]>src[idp])?src[id]:src[idp];"),
        // Compute correlation.
        algo_gen!(center "correlation",CF64|F64 CU32|U32 CU32_4|U32_4,"dst[id] = src[id]*src[idp];"),
        // Compute the FFT
        algo_gen!(log "FFT",CF64_2|F64_2|F64 CU32|U32 CU32_4|U32_4,"dst[id] = src[ida] + c_times(src[idb],c_exp(-2*M_PI*u/(1<<i)));"),
        // Compute moments. With D1 apply on whole buffer, with D2 apply on all y sub-buffers of
        // size x (where x and y are the first and second dimensions).
        Algorithm {
            name: "moments",
            callback: callback_gen!(h,dim,dirs,bufs,param, {
                bufs!(bufs, "moments", 4,
                    src
                    tmp
                    sum
                    dst
                );
                let MomentsParam{ num, vect_dim: w, packed } = if let Ref(p) = param {
                    p.downcast_ref::<MomentsParam>().expect("Optional parameter of \"moment\" algorithm must be MomentsParam.").clone()
                } else {
                    MomentsParam{ num: 4, vect_dim: 1, packed: true }
                };
                if num < 1 { panic!("There must be at least one moment calculated in \"moments\" algorithm."); }

                let (l,size) = match dim {
                    D1(x) => (x,[x as u32,1,1]),
                    D2(x,y) => (x*y,[x as u32,y as u32,1]),
                    D3(x,y,z) => (x*y*z,[x as u32,y as u32,z as u32])
                };

                let sumsize = dirs.iter().fold(size.clone(), |mut a,dir| { a[*dir as usize] = 1; a });
                let sumlen = sumsize[0]*sumsize[1]*sumsize[2];
                let mut sizedst = [if packed { num } else { 1 },sumsize[0],sumsize[1],0];
                let mut ap = ReduceParam{ vect_dim: w, dst_size: Some(Packing::new(sizedst, packed)), window: None };
                h.run_algorithm("sum",dim,dirs,&[src,sum,dst],Ref(&ap))?;
                if num >= 1 {
                    h.run_arg("times",D1(l*w as usize),&[BufArg(&src,"a"),BufArg(&src,"b"),BufArg(&tmp,"dst")])?;
                    sizedst[3] = 1;
                    ap.dst_size = Some(Packing::new(sizedst, packed));
                    h.run_algorithm("sum",dim,dirs,&[tmp,sum,dst],Ref(&ap))?;
                    h.set_arg("times",&[BufArg(&tmp,"a")])?;
                }
                for i in 2..num {
                    h.run("times",D1(l*w as usize))?;
                    sizedst[3] = i as u32;
                    ap.dst_size = Some(Packing::new(sizedst, packed));
                    h.run_algorithm("sum",dim,dirs,&[tmp,sum,dst],Ref(&(ap)))?;
                }
                h.run_arg("ctimes",D1((num*w*sumlen) as usize),&[BufArg(&dst,"src"),Param("c",F64(sumlen as f64/l as f64)),BufArg(&dst,"dst")])?;

                Ok(None)
            }),
            needed: vec![
                KernelName("times"),
                KernelName("ctimes"),
                AlgorithmName("sum"),
            ]
        },
        gen_random!(u64 philox2x64_10,CU64,2),
        gen_random!(u64 philox4x64_10,CU64,4),
        gen_random!(u32 philox4x32_10,CU32,2),//WARNING because floating point results are f64 this 4x32 is considered to take element as if they were u64 (so two u32 count for one)
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
pub fn moments_to_cumulants(moments: &[f64], w: usize) -> Vec<f64> {
    let len = moments.len()/w;
    let mut cumulants = vec![0.0; len*w];
    for i in 0..w {
        for n in 0..len {
            let mut m = 0.0;
            for k in 0..n {
                m += C(n,k) as f64*cumulants[i+w*k]*moments[i+w*(n-k-1)];
            }
            cumulants[i+w*n] = moments[i+w*n] - m;
        }
    }

    cumulants
}
