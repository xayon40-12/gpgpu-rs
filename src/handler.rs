use ocl::{ProQue,Buffer,Kernel};
use std::collections::HashMap; 

pub mod handler_builder;
pub use handler_builder::HandlerBuilder;

use crate::dim::Dim;
use crate::descriptors::KernelDescriptor;
use crate::algorithms::Algorithm;


pub struct Handler {
    pq: ProQue,
    kernels: HashMap<String,Kernel>,
    algorithms: HashMap<String,Algorithm<String>>,
    buffers: HashMap<String,Buffer<f64>>,
}

impl Handler {
    pub fn builder() -> ocl::Result<HandlerBuilder> {
        HandlerBuilder::new()
    }

    pub fn get<S: Into<String>+Clone>(&self, name: S) -> crate::Result<Vec<f64>> {
        let buf = &self.buffers[&name.into()];
        let len = buf.len();
        let mut vec = Vec::with_capacity(len);
        unsafe { vec.set_len(len); }
        buf.read(&mut vec).enq()?;
        Ok(vec)
    }

    pub fn run<S: Into<String>+Clone>(&mut self, name: S, dim: Dim, desc: Vec<KernelDescriptor<S>>) -> ocl::Result<()> {
        unsafe { 
            let kernel = &self.kernels[&name.into()];
            for d in desc {
                match d {
                    KernelDescriptor::Param(n,v) =>
                        kernel.set_arg(n.into(),v),
                    KernelDescriptor::Buffer(n) => {
                        let n = n.into();
                        kernel.set_arg(n.clone(),&self.buffers[&n])
                    },
                    KernelDescriptor::BufArg(n,m) =>
                        kernel.set_arg(m.into(),&self.buffers[&n.into()])
                }?;
            }

            kernel.cmd().global_work_size(dim).enq()
        }
    }
    
    pub fn run_algorithm<S: Into<String>+Clone>(&mut self, name: S) -> crate::Result<()> {
        (self.algorithms[&name.into()].callback.clone())(self)
    }

    pub fn kernel_dim<S: Into<String>+Clone>(&self, kernel_name: S) -> Dim {
        match self.kernels[&kernel_name.into()].wg_info(self.pq.device(),
        ocl::enums::KernelWorkGroupInfo::GlobalWorkSize).unwrap() {
            ocl::enums::KernelWorkGroupInfoResult::GlobalWorkSize(size) => size.into(),
            _ => panic!("Dimension not available.")
        }
    }

}
