use crate::descriptors::KernelDescriptor::{self,*};
use std::collections::HashMap;

#[derive(Clone)]
pub struct Kernel {
    pub name: &'static str,
    pub args: Vec<KernelDescriptor>,
    pub src: &'static str
}

pub fn kernels() -> HashMap<&'static str,Kernel> {
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
    ].into_iter().map(|k| (k.name,k)).collect()
}
