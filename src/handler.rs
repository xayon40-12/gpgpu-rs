use ocl::{ProQue,Buffer,Kernel};
use std::collections::HashMap; 

pub mod handler_builder;
pub use handler_builder::HandlerBuilder;

use crate::dim::Dim;


pub struct Handler {
    _pq: ProQue,
    kernels: HashMap<String,Kernel>,
    buffers: HashMap<String,Buffer<f64>>,
}

impl Handler {
    pub fn builder<S: Into<String>+Clone>(src: S) -> ocl::Result<HandlerBuilder<S>> {
        HandlerBuilder::new(src)
    }

    pub fn get<S: Into<String>+Clone>(&self, name: S) -> crate::Result<Vec<f64>> {
        let buf = &self.buffers[&name.into()];
        let len = buf.len();
        let mut vec = Vec::with_capacity(len);
        unsafe { vec.set_len(len); }
        buf.read(&mut vec).enq()?;
        Ok(vec)
    }

    pub fn run<S: Into<String>+Clone>(&mut self, name: S, dim: Dim) -> ocl::Result<()> {
        unsafe { self.kernels[&name.into()].cmd().global_work_size(dim).enq() }
    }
}
