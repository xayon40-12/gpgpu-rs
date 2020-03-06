use ocl::{ProQue,Buffer,Kernel};
use std::collections::HashMap; 

pub struct HandlerBuilder {
    pq: Option<ProQue>,
    src: String,
    kernel: Option<Kernel>,
    buffers: HashMap<String,Buffer<i64>>,
    params: Vec<super::Param>,
    params_buffer: Option<Buffer<i64>>
}

impl HandlerBuilder {
    pub fn new(src: &str) -> HandlerBuilder {
        HandlerBuilder {
            pq: None,
            src: src.to_string(),
            kernel: None,
            buffers: HashMap::new(),
            params: Vec::new(),
            params_buffer: None
        }
    }

    pub fn build(self) -> ocl::Result<super::Handler> {
        let src = self.src.clone();
        //TODO complet src with all the outside of the program with "main" as entry point (declaration, structs, ...)

        let pq = ProQue::builder()
            .src(src)
            .build()?;

        let buffers = self.buffers;
        let mut kernel = pq.kernel_builder("main");
        for (n,b) in &buffers {
            kernel.arg_named(n,b);
        }
        let kernel = kernel.build()?;

        Ok(super::Handler {
            pq,
            kernel,
            buffers: buffers,
            params: self.params,
            params_buffer: self.params_buffer
        })
    }


}
