use crate::descriptors::KernelDescriptor::{self,*};
use std::collections::HashMap;

#[derive(Clone)]
pub struct Kernel<S: Into<String>+Clone> {
    pub name: S,
    pub args: Vec<KernelDescriptor<S>>,
    pub src: S
}

pub fn convert<S: Into<String>+Clone>(k: &Kernel<S>) -> Kernel<String> {
    Kernel {
        name: k.name.clone().into(),
        args: k.args.iter().map(|a| match a {
            KernelDescriptor::Buffer(n) => KernelDescriptor::Buffer(n.clone().into()),
            KernelDescriptor::BufArg(n,m) => KernelDescriptor::BufArg(n.clone().into(),m.clone().into()),
            KernelDescriptor::Param(n,v) => KernelDescriptor::Param(n.clone().into(),*v)
        }).collect(),
        src: k.src.clone().into()
    }
}

pub fn kernels() -> HashMap<String,Kernel<&'static str>> {
    vec![
        Kernel {
            name: "plus",
            args: vec![Buffer("a"),Buffer("b"),Buffer("dst")],
            src: "dst[x] = a[x]+b[x];"
        },
        Kernel {
            name: "minus",
            args: vec![Buffer("a"),Buffer("b"),Buffer("dst")],
            src: "dst[x] = a[x]-b[x];"
        },
        Kernel {
            name: "times",
            args: vec![Buffer("a"),Buffer("b"),Buffer("dst")],
            src: "dst[x] = a[x]*b[x];"
        },
        Kernel {
            name: "divided",
            args: vec![Buffer("a"),Buffer("b"),Buffer("dst")],
            src: "dst[x] = a[x]/b[x];"
        },
    ].into_iter().map(|k| (k.name.to_string(),k)).collect()
}
