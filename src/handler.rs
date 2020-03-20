use ocl::{ProQue,Buffer,Kernel};
use std::collections::{HashMap,BTreeMap};

pub mod handler_builder;
pub use handler_builder::HandlerBuilder;

use crate::dim::Dim;
use crate::descriptors::KernelDescriptor;
use crate::algorithms::Callback;

#[allow(dead_code)]
pub struct Handler {
    pq: ProQue,
    kernels: HashMap<String,(Kernel,BTreeMap<String,u32>)>,
    algorithms: HashMap<String,Callback>,
    buffers: HashMap<String,Buffer<f64>>,
}

impl Handler {
    pub fn builder<'a>() -> ocl::Result<HandlerBuilder<'a>> {
        HandlerBuilder::new()
    }

    pub fn get(&self, name: &str) -> crate::Result<Vec<f64>> {
        let buf = self.buffers.get(name).expect(&format!("Buffer \"{}\" not found",name));
        let len = buf.len();
        let mut vec = Vec::with_capacity(len);
        unsafe { vec.set_len(len); }
        buf.read(&mut vec).enq()?;
        Ok(vec)
    }

    pub fn get_first(&self, name: &str) -> crate::Result<f64> {
        let buf = self.buffers.get(name).expect(&format!("Buffer \"{}\" not found",name));
        let mut val = vec![0f64];
        buf.read(&mut val).enq()?;
        Ok(val[0])
    }

    pub fn run(&mut self, name: &str, dim: Dim, desc: Vec<KernelDescriptor>) -> ocl::Result<()> {
        let kernel = &self.kernels.get(name).expect(&format!("Kernel \"{}\" not found",name));
        for d in desc {
            match d {
                KernelDescriptor::Param(n,v) =>
                    kernel.0.set_arg(kernel.1[n],v),
                KernelDescriptor::Buffer(n) =>
                    kernel.0.set_arg(kernel.1[n],self.buffers.get(n).expect(&format!("Buffer \"{}\" not found",n))),
                KernelDescriptor::BufArg(n,m) =>
                    kernel.0.set_arg(kernel.1[m],self.buffers.get(n).expect(&format!("Buffer \"{}\" not found",n)))
            }?;
        }

        unsafe {
            kernel.0.cmd().global_work_size(dim).enq()
        }
    }
    
    pub fn run_algorithm(&mut self, name: &str, dim: Dim, desc: Vec<KernelDescriptor>) -> crate::Result<()> {
        (self.algorithms.get(name).expect(&format!("Algorithm \"{}\" not found",name)).clone())(self,dim,desc)
    }
}
