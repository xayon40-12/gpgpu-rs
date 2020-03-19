use crate::descriptors::KernelDescriptor::{self,*};
use std::collections::HashMap;

#[derive(Clone)]
pub struct Kernel<'a> {
    pub name: &'a str,
    pub args: Vec<KernelDescriptor>,
    pub src: &'a str
}

pub fn kernels<'a>() -> HashMap<&'static str,Kernel<'a>> {
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
            name: "divide",
            args: vec![Buffer("a"),Buffer("b"),Buffer("dst")],
            src: "dst[x] = a[x]/b[x];"
        },
        Kernel {
            name: "cplus",
            args: vec![Buffer("a"),Param("c",0.0),Buffer("dst")],
            src: "dst[x] = a[x]+c;"
        },
        Kernel {
            name: "cminus",
            args: vec![Buffer("a"),Param("c",0.0),Buffer("dst")],
            src: "dst[x] = a[x]-c;"
        },
        Kernel {
            name: "ctimes",
            args: vec![Buffer("a"),Param("c",0.0),Buffer("dst")],
            src: "dst[x] = a[x]*c;"
        },
        Kernel {
            name: "cdivide",
            args: vec![Buffer("a"),Param("c",0.0),Buffer("dst")],
            src: "dst[x] = a[x]/c;"
        },
        Kernel {
            name: "philox2x64_10",
            args: vec![Buffer("src"),Buffer("dst")],
            src: "
                unsigned long key = x;
                const unsigned int l = 2;
                const unsigned long long M = 0xD2B74407B1CE6E93;
                unsigned long counter[2] = {src[x*l],src[x*l+1]};
                for(int i = 0;i<10;i++){
                    unsigned long long prod = M * counter[0];
                    unsigned long hi = (prod >> 64);
                    unsigned long lo = prod;
                    counter[0] = hi^key^counter[1];
                    counter[1] = lo;
                    key += 0x9E3779B97F4A7C15;
                }
                dst[x*l]   = (double)(counter[0]>>11)/(1l << 53);
                dst[x*l+1] = (double)(counter[1]>>11)/(1l << 53);
                src[x*l]   = counter[0];
                src[x*l+1] = counter[1];
            "
        },
        Kernel {
            name: "philox4x64_10",
            args: vec![Buffer("src"),Buffer("dst")],
            src: "
                unsigned long key[2] = {0,x};
                const unsigned int l = 4;
                const unsigned long long M = 0xD2B74407B1CE6E93;
                const unsigned long long M2 = 0xCA5A826395121157;
                unsigned long counter[4] = {src[x*l],src[x*l+1],src[x*l+2],src[x*l+3]};
                for(int i = 0;i<10;i++){
                    unsigned long long prod = M * counter[0];
                    unsigned long hi0 = (prod >> 64);
                    unsigned long lo0 = prod;
                    prod = M2 * counter[2];
                    unsigned long hi1 = (prod >> 64);
                    unsigned long lo1 = prod;
                    counter[0] = hi1^key[1]^counter[3];
                    counter[2] = hi0^key[0]^counter[1];
                    counter[1] = lo0;
                    counter[3] = lo1;
                    key[0] += 0x9E3779B97F4A7C15;
                    key[1] += 0xBB67AE8584CAA73B;
                }
                dst[x*l]   = (double)(counter[0]>>11)/(1l << 53);
                dst[x*l+1] = (double)(counter[1]>>11)/(1l << 53);
                dst[x*l+2] = (double)(counter[2]>>11)/(1l << 53);
                dst[x*l+3] = (double)(counter[3]>>11)/(1l << 53);
                src[x*l]   = counter[0];
                src[x*l+1] = counter[1];
                src[x*l+2] = counter[2];
                src[x*l+3] = counter[3];
            "
        },
        Kernel {
            name: "philox4x32_10",
            args: vec![Buffer("src"),Buffer("dst")],
            src: "
                unsigned int key[2] = {x>>32,x};
                const unsigned int l = 2;
                const unsigned long M = 0xD2511F53;
                const unsigned long M2 = 0xCD9E8D57;
                const unsigned long tmp0 = src[x*l], tmp1 = src[x*l+1];
                unsigned int counter[4] = {tmp0>>32,tmp0,tmp1>>32,tmp1};
                for(int i = 0;i<10;i++){
                    unsigned long prod = M * counter[0];
                    unsigned int hi0 = (prod >> 32);
                    unsigned int lo0 = prod;
                    prod = M2 * counter[2];
                    unsigned int hi1 = (prod >> 32);
                    unsigned int lo1 = prod;
                    counter[0] = hi1^key[1]^counter[3];
                    counter[2] = hi0^key[0]^counter[1];
                    counter[1] = lo0;
                    counter[3] = lo1;
                    key[0] += 0x9E3779B9;
                    key[1] += 0xBB67AE85;
                }
                unsigned long r1 = (((unsigned long)counter[0])<<32)+counter[1];
                unsigned long r2 = (((unsigned long)counter[2])<<32)+counter[3];
                dst[x*l]   = (double)(r1>>11)/(1l << 53);
                dst[x*l+1] = (double)(r2>>11)/(1l << 53);
                src[x*l]   = r1;
                src[x*l+1] = r2;
            "//TODO make this kernel take 4 uint as src and not 2 double (or ulong)
        },
    ].into_iter().map(|k| (k.name,k)).collect()
}
