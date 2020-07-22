use crate::descriptors::{KernelConstructor::{self,*},SKernelConstructor};
use crate::descriptors::ConstructorTypes::*;
use std::collections::HashMap;
use crate::functions::{Needed::{self,*},SNeeded};
use serde::{Serialize,Deserialize};

#[derive(Clone,Debug)]
pub struct Kernel<'a> { //TODO use one SC for each &'a str
    pub name: &'a str,
    pub args: Vec<KernelConstructor<'a>>,
    pub src: &'a str,
    pub needed: Vec<Needed<'a>>,
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct SKernel {
    pub name: String,
    pub args: Vec<SKernelConstructor>,
    pub src: String,
    pub needed: Vec<SNeeded>,
}

impl<'a> From<&Kernel<'a>> for SKernel {
    fn from(f: &Kernel<'a>) -> Self {
        SKernel {
            name: f.name.into(),
            args: f.args.iter().map(|i| i.into()).collect(),
            src: f.src.into(),
            needed: f.needed.iter().map(|i| i.into()).collect(),
        }
    }
}


pub fn kernels() -> HashMap<&'static str,Kernel<'static>> {
    vec![
        Kernel {
            name: "plus",
            args: vec![KCBuffer("a",CF64),KCBuffer("b",CF64),KCBuffer("dst",CF64)],
            src: "    dst[x] = a[x]+b[x];",
            needed: vec![],
        },
        Kernel {
            name: "minus",
            args: vec![KCBuffer("a",CF64),KCBuffer("b",CF64),KCBuffer("dst",CF64)],
            src: "    dst[x] = a[x]-b[x];",
            needed: vec![],
        },
        Kernel {
            name: "times",
            args: vec![KCBuffer("a",CF64),KCBuffer("b",CF64),KCBuffer("dst",CF64)],
            src: "    dst[x] = a[x]*b[x];",
            needed: vec![],
        },
        Kernel {
            name: "divides",
            args: vec![KCBuffer("a",CF64),KCBuffer("b",CF64),KCBuffer("dst",CF64)],
            src: "    dst[x] = a[x]/b[x];",
            needed: vec![],
        },
        Kernel {
            name: "cplus",
            args: vec![KCBuffer("src",CF64),KCParam("c",CF64),KCBuffer("dst",CF64)],
            src: "    dst[x] = src[x]+c;",
            needed: vec![],
        },
        Kernel {
            name: "cminus",
            args: vec![KCBuffer("src",CF64),KCParam("c",CF64),KCBuffer("dst",CF64)],
            src: "    dst[x] = src[x]-c;",
            needed: vec![],
        },
        Kernel {
            name: "ctimes",
            args: vec![KCBuffer("src",CF64),KCParam("c",CF64),KCBuffer("dst",CF64)],
            src: "    dst[x] = src[x]*c;",
            needed: vec![],
        },
        Kernel {
            name: "cdivides",
            args: vec![KCBuffer("src",CF64),KCParam("c",CF64),KCBuffer("dst",CF64)],
            src: "    dst[x] = src[x]/c;",
            needed: vec![],
        },
        Kernel {
            name: "vcplus",
            args: vec![KCBuffer("src",CF64),KCBuffer("dst",CF64),KCBuffer("c",CF64),KCParam("size",CU32_4),KCParam("vect_dim",CU32)],
            src: "    for(uint _w = 0; _w<vect_dim; _w++) dst[vect_dim*(x+x_size*(y+y_size*z))+_w] = src[vect_dim*(x+x_size*(y+y_size*z))+_w]+c[_w*size.w/vect_dim + size.w*(x*size.x/x_size+size.x*(y*size.y/y_size+size.y*(z*size.z/z_size)))];",
            needed: vec![],
        },
        Kernel {
            name: "vcminus",
            args: vec![KCBuffer("src",CF64),KCBuffer("dst",CF64),KCBuffer("c",CF64),KCParam("size",CU32_4),KCParam("vect_dim",CU32)],
            src: "    for(uint _w = 0; _w<vect_dim; _w++) dst[vect_dim*(x+x_size*(y+y_size*z))+_w] = src[vect_dim*(x+x_size*(y+y_size*z))+_w]-c[_w*size.w/vect_dim + size.w*(x*size.x/x_size+size.x*(y*size.y/y_size+size.y*(z*size.z/z_size)))];",
            needed: vec![],
        },
        Kernel {
            name: "vctimes",
            args: vec![KCBuffer("src",CF64),KCBuffer("dst",CF64),KCBuffer("c",CF64),KCParam("size",CU32_4),KCParam("vect_dim",CU32)],
            src: "    for(uint _w = 0; _w<vect_dim; _w++) dst[vect_dim*(x+x_size*(y+y_size*z))+_w] = src[vect_dim*(x+x_size*(y+y_size*z))+_w]*c[_w*size.w/vect_dim + size.w*(x*size.x/x_size+size.x*(y*size.y/y_size+size.y*(z*size.z/z_size)))];",
            needed: vec![],
        },
        Kernel {
            name: "vcdivides",
            args: vec![KCBuffer("src",CF64),KCBuffer("dst",CF64),KCBuffer("c",CF64),KCParam("size",CU32_4),KCParam("vect_dim",CU32)],
            src: "    for(uint _w = 0; _w<vect_dim; _w++) dst[vect_dim*(x+x_size*(y+y_size*z))+_w] = src[vect_dim*(x+x_size*(y+y_size*z))+_w]/c[_w*size.w/vect_dim + size.w*(x*size.x/x_size+size.x*(y*size.y/y_size+size.y*(z*size.z/z_size)))];",
            needed: vec![],
        },
        Kernel {
            name: "move",
            args: vec![KCBuffer("src",CF64),KCBuffer("dst",CF64),KCParam("size",CU32_4),KCParam("offset",CU32),KCParam("vect_dim",CU32)],
            src: "    for(uint _w = 0; _w<vect_dim; _w++) dst[vect_dim*(x+x_size*(y+y_size*z) + offset)+_w] = src[vect_dim*(x+size.x*(y+size.y*z))+_w];",
            needed: vec![],
        },
        Kernel {
            name: "omove",
            args: vec![KCBuffer("src",CF64),KCBuffer("dst",CF64),KCParam("size",CU32_2),KCParam("offsets",CU32_4),KCParam("vect_dim",CU32)],
            src: "    for(uint _w = 0; _w<vect_dim; _w++) dst[vect_dim*(x+x_size*(y+y_size*z))+_w] = src[vect_dim*(x+offsets.x)+size.x*((y+offsets.y)+size.y*(z+offsets.z))+_w];",
            needed: vec![],
        },
        Kernel {
            name: "smove",
            args: vec![KCBuffer("src",CF64),KCBuffer("dst",CF64),KCParam("size",CU32_4),KCParam("offset",CU32),KCParam("vect_dim",CU32)],
            src: "    for(uint _w = 0; _w<vect_dim; _w++) dst[vect_dim*(x+size.x*(y+size.y*z) + offset)+_w] = src[vect_dim*x+x_size*(y+y_size*z)+_w];",
            needed: vec![],
        },
        Kernel {
            name: "dmove",
            args: vec![KCBuffer("src",CF64),KCBuffer("dst",CF64),KCParam("size",CU32_4),KCParam("dst_size",CU32_4),KCParam("vect_dim",CU32)],
            src: "    for(uint _w = 0; _w<vect_dim; _w++) dst[vect_dim*(dst_size.x*(x+dst_size.y*(y+dst_size.z*z)) + dst_size.w)+_w] = src[vect_dim*x+size.x*(y+size.y*z)+_w];",
            needed: vec![],
        },
        Kernel {
            name: "rdmove",
            args: vec![KCBuffer("src",CF64),KCBuffer("dst",CF64),KCParam("size",CU32_4),KCParam("dst_size",CU32_4),KCParam("vect_dim",CU32)],
            src: "    for(uint _w = 0; _w<vect_dim; _w++) dst[vect_dim*(x+dst_size.y*(y+dst_size.z*(z+dst_size.x*dst_size.w)))+_w] = src[vect_dim*x+size.x*(y+size.y*z)+_w];",
            needed: vec![],
        },
        Kernel {
            name: "complex_from_real",
            args: vec![KCBuffer("src",CF64),KCBuffer("dst",CF64_2)],
            src: "    dst[x] = (double2)(src[x],0);",
            needed: vec![],
        },
        Kernel {
            name: "complex_from_image",
            args: vec![KCBuffer("src",CF64),KCBuffer("dst",CF64_2)],
            src: "    dst[x] = (double2)(0,src[x]);",
            needed: vec![],
        },
        Kernel {
            name: "real_from_complex",
            args: vec![KCBuffer("src",CF64_2),KCBuffer("dst",CF64)],
            src: "    dst[x] = src[x].x;",
            needed: vec![],
        },
        Kernel {
            name: "image_from_complex",
            args: vec![KCBuffer("src",CF64_2),KCBuffer("dst",CF64)],
            src: "    dst[x] = src[x].y;",
            needed: vec![],
        },
        Kernel {
            name: "kc_sqrmod",
            args: vec![KCBuffer("src",CF64_2),KCBuffer("dst",CF64)],
            src: "    dst[x] = c_sqrmod(src[x]);",
            needed: vec![FuncName("c_sqrmod".into())],
        },
        Kernel {
            name: "kc_mod",
            args: vec![KCBuffer("src",CF64_2),KCBuffer("dst",CF64)],
            src: "    dst[x] = c_mod(src[x]);",
            needed: vec![FuncName("c_mod".into())],
        },
        Kernel {
            name: "kc_times",
            args: vec![KCBuffer("a",CF64_2),KCBuffer("b",CF64_2),KCBuffer("dst",CF64_2)],
            src: "    dst[x] = c_times(a[x],b[x]);",
            needed: vec![FuncName("c_times".into())],
        },
        Kernel {
            name: "kc_times_conj",
            args: vec![KCBuffer("a",CF64_2),KCBuffer("b",CF64_2),KCBuffer("dst",CF64_2)],
            src: "    dst[x] = c_times_conj(a[x],b[x]);",
            needed: vec![FuncName("c_times_conj".into())],
        },
        Kernel {
            name: "kc_divides",
            args: vec![KCBuffer("a",CF64_2),KCBuffer("b",CF64_2),KCBuffer("dst",CF64_2)],
            src: "    dst[x] = c_divides(a[x],b[x]);",
            needed: vec![FuncName("c_divides".into())],
        },
        Kernel {
            name: "moments_to_cumulants",
            args: vec![KCBuffer("moments",CF64),KCBuffer("cumulants",CF64),KCParam("vect_dim",CU32),KCParam("num",CU32)],
            src: "
    uint C[] = {1,1,1,1,2,1,1,3,3,1,1,4,6,4,1,1,5,10,10,5,1,1,6,15,20,15,6,1,1,7,21,35,35,21,7,1,1,8,28,56,70,56,28,8,1,1,9,36,84,126,126,84,36,9,1,1,10,45,120,210,252,210,120,45,10,1,1,11,55,165,330,462,462,330,165,55,11,1,1,12,66,220,495,792,924,792,495,220,66,12,1,1,13,78,286,715,1287,1716,1716,1287,715,286,78,13,1,1,14,91,364,1001,2002,3003,3432,3003,2002,1001,364,91,14,1,1,15,105,455,1365,3003,5005,6435,6435,5005,3003,1365,455,105,15,1,1,16,120,560,1820,4368,8008,11440,12870,11440,8008,4368,1820,560,120,16,1,1,17,136,680,2380,6188,12376,19448,24310,24310,19448,12376,6188,2380,680,136,17,1,1,18,153,816,3060,8568,18564,31824,43758,48620,43758,31824,18564,8568,3060,816,153,18,1,1,19,171,969,3876,11628,27132,50388,75582,92378,92378,75582,50388,27132,11628,3876,969,171,19,1,1,20,190,1140,4845,15504,38760,77520,125970,167960,184756,167960,125970,77520,38760,15504,4845,1140,190,20,1,1,21,210,1330,5985,20349,54264,116280,203490,293930,352716,352716,293930,203490,116280,54264,20349,5985,1330,210,21,1,1,22,231,1540,7315,26334,74613,170544,319770,497420,646646,705432,646646,497420,319770,170544,74613,26334,7315,1540,231,22,1,1,23,253,1771,8855,33649,100947,245157,490314,817190,1144066,1352078,1352078,1144066,817190,490314,245157,100947,33649,8855,1771,253,23,1,1,24,276,2024,10626,42504,134596,346104,735471,1307504,1961256,2496144,2704156,2496144,1961256,1307504,735471,346104,134596,42504,10626,2024,276,24,1};
    uint w = vect_dim;
    uint size = w*num;
    for(uint i = 0; i<vect_dim; i++) {
        for(uint n = 0; n<num; n++) {
            uint nn = n*(n+1)/2;
            double m = 0.0;
            for(uint k = 0; k<n; k++) {
                m += C[nn + k]*cumulants[i+w*k + x*size]*moments[i+w*(n-k-1) + x*size];
            }
            cumulants[i+w*n + x*size] = moments[i+w*n + x*size] - m;
        }
    }
"
            ,
            needed: vec![]
        },
        ].into_iter().map(|k| (k.name,k)).collect()
}
