use ocl::ProQue;
use std::collections::HashMap;
use crate::descriptors::*;


pub struct HandlerBuilder<S: Into<String>+Clone> {
    src: String,
    kernels: Vec<(String,Vec<KernelDescriptor<S>>)>,
    buffers: Vec<(String,BufferDescriptor)>
}

impl<S: Into<String>+Clone> HandlerBuilder<S> {
    pub fn new(src: S) -> ocl::Result<HandlerBuilder<S>> {
        Ok(HandlerBuilder {
            src: src.into(),
            kernels: Vec::new(),
            buffers: Vec::new()
        })
    }

    pub fn add_buffer(mut self, name: S, desc: BufferDescriptor) -> Self {
        self.buffers.push((name.into(),desc));
        self
    }

    pub fn add_buffers(self, buffers: Vec<(S,BufferDescriptor)>) -> Self {
        let mut hand = self;
        for (name,desc) in buffers {
            hand = hand.add_buffer(name,desc);
        }
        hand
    }

    pub fn add_kernel(mut self, name: S, desc: Vec<KernelDescriptor<S>>) -> Self {
        self.kernels.push((name.into(),desc));
        self
    }

    pub fn build(self) -> ocl::Result<super::Handler> {
        let mut src = String::new();
        //TODO complet src with all the outside of the program with "main" as entry point (declaration, structs, ...)
        
        src += "\n__kernel void main(";
        if let Some(desc) = self.kernels.iter()
                                        .find(|(n,_)| n == &"main".to_string())
                                        .and_then(|(_,d)| Some(d)) {
            for d in desc {
                match d {
                    KernelDescriptor::Param(n,_) => { src += "double "; src += &n.clone().into(); },
                    KernelDescriptor::Buffer(n) => { src += "__global double *"; src += &n.clone().into(); }
                    KernelDescriptor::BufDst(_) => { src += "__global double *dst"; },
                    KernelDescriptor::BufSrc(_) => { src += "__global double *src"; }
                };
                src += ",";
            }
        } else {
            panic!("No \"main\" kernel description done. Consider using add_kernel(\"main\", ...).");
        }
        src.pop(); // remove last unnescessary ","
        src += ") {\n";
        src += "    long x = get_global_id(0); long x_size = get_global_size(0);\n";
        src += "    long y = get_global_id(0); long y_size = get_global_size(0);\n";
        src += "    long z = get_global_id(0); long z_size = get_global_size(0);\n";
        src += &self.src;
        src += "\n}\n";

        let pq = ProQue::builder()
            .src(src)
            .dims(1) //TODO should not be needed
            .build()?;

        let mut buffers = HashMap::new();
        for (name,desc) in self.buffers {
            match desc {
                BufferDescriptor::Len(len,val) =>
                    buffers.insert(name, pq.buffer_builder()
                                           .len(len)
                                           .fill_val(val)
                                           .build()?),
                BufferDescriptor::Data(data) =>
                    buffers.insert(name, pq.buffer_builder()
                                           .len(data.len())
                                           .copy_host_slice(&data)
                                           .build()?)
            };
        }

        let mut kernels = HashMap::new();
        for (name,desc) in self.kernels {
            let mut kernel = pq.kernel_builder(&name);
            for desc in desc {
                match desc {
                    KernelDescriptor::Param(n,v) => 
                        kernel.arg_named(n.into(),v),
                    KernelDescriptor::Buffer(n) => {
                        let n = n.into();
                        kernel.arg_named(n.clone(),&buffers[&n])
                    },
                    KernelDescriptor::BufDst(n) => 
                        kernel.arg_named("dst".clone(),&buffers[&n.into()]),
                    KernelDescriptor::BufSrc(n) => 
                        kernel.arg_named("src".clone(),&buffers[&n.into()])
                };
            }
            kernels.insert(name,kernel.build()?);
        }

        Ok(super::Handler {
            _pq: pq,
            kernels,
            buffers
        })
    }


}
