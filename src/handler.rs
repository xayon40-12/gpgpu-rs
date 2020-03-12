use ocl::{ProQue,Buffer,Kernel};
use std::collections::HashMap; 

pub mod handler_builder;
pub use handler_builder::HandlerBuilder;

use crate::dim::Dim;
use crate::descriptors::KernelDescriptor;
use crate::algorithms::Callback;


pub struct Handler {
    pq: ProQue,
    kernels: HashMap<&'static str,Kernel>,
    algorithms: HashMap<&'static str,Callback>,
    buffers: HashMap<&'static str,Buffer<f64>>,
}

impl Handler {
    pub fn builder() -> ocl::Result<HandlerBuilder> {
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

    pub fn run(&mut self, name: &str, dim: Dim, desc: Vec<KernelDescriptor>) -> ocl::Result<()> {
        let kernel = &self.kernels[name];
        for d in desc {
            match d {
                KernelDescriptor::Param(n,v) =>
                    kernel.set_arg(n,v),
                KernelDescriptor::Buffer(n) =>
                    kernel.set_arg(n,self.buffers.get(n).expect(&format!("Buffer \"{}\" not found",n))),
                KernelDescriptor::BufArg(n,m) =>
                    kernel.set_arg(m,self.buffers.get(n).expect(&format!("Buffer \"{}\" not found",n)))
            }?;
        }

        unsafe {
            kernel.cmd().global_work_size(dim).enq()
        }
    }
    
    pub fn run_algorithm(&mut self, name: &str, dim: Dim, desc: Vec<KernelDescriptor>) -> crate::Result<()> {
        (self.algorithms[name].clone())(self,dim,desc)
    }
}
