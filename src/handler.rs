use ocl::{ProQue,Buffer,Kernel};
use std::collections::HashMap; 

pub mod handler_builder;
pub use handler_builder::HandlerBuilder;

struct Param {
    id: usize,
    val: i64
}

pub struct Handler {
    pq: ProQue,
    kernel: Kernel,
    buffers: HashMap<String,Buffer<i64>>,
    params: Vec<Param>,
    params_buffer: Option<Buffer<i64>>
}

impl Handler {
    pub fn builder(src: &str) -> HandlerBuilder {
        HandlerBuilder::new(src)
    }

    pub fn run(&mut self, dim: &[u64]) -> ocl::Result<()> {
        unsafe { self.kernel.enq() }
    }
}
