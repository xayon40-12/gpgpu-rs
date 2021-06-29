use crate::descriptors::ConstructorTypes::*;
use crate::descriptors::{
    KernelConstructor::{self, *},
    SKernelConstructor,
};
use crate::functions::{
    Needed::{self, *},
    SNeeded,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Kernel<'a> {
    //TODO use one SC for each &'a str
    pub name: &'a str,
    pub args: Vec<KernelConstructor<'a>>,
    pub src: &'a str,
    pub needed: Vec<Needed<'a>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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

pub fn kernels() -> HashMap<&'static str, Kernel<'static>> {
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

#[derive(Debug, Clone)]
pub struct Radial {
    pub pos: f64,
    pub vals: Vec<f64>,
}

pub fn radial(
    a: &Vec<f64>,
    vec_dim: usize,
    dim: &[usize; 3],
    weight: &[f64; 3],
    mean: bool,
) -> Vec<Vec<Radial>> {
    weight
        .iter()
        .for_each(|w| assert!(*w >= 0.0, "Each weight must be positive."));
    dim.iter()
        .for_each(|d| assert!(*d >= 1, "Each dim must be at least 1."));
    assert!(weight[0] > 0.0, "first weight must be non zero");
    assert!(
        (weight[1] == 0.0 && weight[2] == 0.0) || weight[1] != 0.0,
        "Only the last non empty dimension can have a zero weight to mean the result is chunked."
    );

    let dm = [dim[0] / 2, dim[1] / 2, dim[2] / 2];
    let dist = |p: &[usize], i: usize| {
        ((p[i] as f64 - dm[i] as f64) * weight[i] / dim[i] as f64).powf(2.0)
    };
    let pos = |i: usize| {
        let x = i % dim[0];
        let y = (i / dim[0]) % dim[1];
        let z = (i / dim[0] / dim[1]) % dim[2];
        let p = [x, y, z];
        let d = (0..3).fold(0.0, |acc, i| acc + dist(&p, i)).sqrt();
        d
    };
    // vec with distance to center
    let n = dim
        .iter()
        .zip(weight.iter())
        .fold(1, |acc, (a, b)| if *b == 0.0 { 1 } else { *a } * acc);
    let mut res: Vec<Vec<Radial>> = a
        .chunks(vec_dim)
        .enumerate()
        .map(|(i, v)| Radial {
            pos: pos(i),
            vals: (*v).to_vec(),
        })
        .collect::<Vec<_>>()
        .chunks(n)
        .map(|i| i.to_vec())
        .collect();
    // sort by distance
    res.iter_mut()
        .for_each(|i| i.sort_by(|a, b| a.pos.partial_cmp(&b.pos).unwrap()));

    // compact by same distance
    let mut j = 0;
    let mut counts = vec![1];
    for i in 1..n {
        if res[0][i].pos == res[0][j].pos {
            counts[j] += 1;
            res.iter_mut().for_each(|res| {
                for vd in 0..vec_dim {
                    res[j].vals[vd] += res[i].vals[vd]
                }
            });
        } else {
            j += 1;
            counts.push(1);
            res.iter_mut().for_each(|res| res[j] = res[i].clone());
        }
    }

    // truncate so that the vector have a lenght corresponding to only the compacted elements
    res.iter_mut().for_each(|res| res.truncate(j + 1));
    // obtaining the mean
    if mean {
        res.iter_mut().for_each(|res| {
            for i in 0..=j {
                for vd in 0..vec_dim {
                    res[i].vals[vd] /= counts[i] as f64
                }
            }
        });
    }

    res
}

#[test]
fn radial_test() {
    let s = [100, 100, 100];
    let p = [10.0, 10.0, 0.0];
    let s2 = s.iter().map(|i| *i as i32 / 2).collect::<Vec<_>>();
    let s2 = &s2;
    let dx = s
        .iter()
        .zip(p.iter())
        .fold(0.0, |acc, (s, p)| acc + (*p / *s as f64).powf(2.0))
        .sqrt();
    let f = |x: f64| f64::exp(-x / 2.0) / x;
    let a = (0..s[0] as i32)
        .flat_map(move |z| {
            (0..s[1] as i32).flat_map(move |y| {
                (0..s[2] as i32).map(move |x| {
                    let u = (0..3)
                        .fold(0.0, |acc, i| {
                            acc + ((([x, y, z][i] - s2[i]) as f64 * p[i] / s[i] as f64).powf(2.0))
                        })
                        .sqrt();
                    f(u + dx)
                })
            })
        })
        .collect::<Vec<_>>();
    let res = radial(&a, 1, &s, &p, true);

    let cmp = |a: f64, b: f64, e: usize| (a - b) / a < 10.0f64.powf(-(e as f64));
    assert!(res.len() == s[2], "There should be {} chunks", s[2]);
    let j = res[0].len();
    res.iter()
        .for_each(|r| assert!(r.len() == j, "Each chunk should have {}", j));
    for i in &res {
        for r in i {
            let v = r.vals[0];
            let e = f(r.pos + dx);
            assert!(
                cmp(e, v, 13),
                "\ndiff: 10^{}\nv: {}\ne :{}\npos: {}\n",
                ((e - v) / v).log10(),
                v,
                e,
                r.pos
            );
        }
    }
}
