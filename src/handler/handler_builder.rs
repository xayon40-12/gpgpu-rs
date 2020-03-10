use ocl::ProQue;
use std::collections::HashMap;
use crate::descriptors::*;
use crate::kernels::{Kernel,self};

pub struct HandlerBuilder {
    available_kernels: HashMap<String,Kernel<&'static str>>,
    kernels: Vec<(Kernel<String>,Option<String>)>,
    buffers: Vec<(String,BufferDescriptor)>
}

impl HandlerBuilder {
    pub fn new() -> ocl::Result<HandlerBuilder> {
        Ok(HandlerBuilder {
            available_kernels: kernels::kernels(),
            kernels: Vec::new(),
            buffers: Vec::new()
        })
    }

    pub fn add_buffer<S: Into<String>+Clone>(mut self, name: S, desc: BufferDescriptor) -> Self {
        self.buffers.push((name.into(),desc));
        self
    }

    pub fn add_buffers<S: Into<String>+Clone>(self, buffers: Vec<(S,BufferDescriptor)>) -> Self {
        let mut hand = self;
        for (name,desc) in buffers {
            hand = hand.add_buffer(name,desc);
        }
        hand
    }

    pub fn create_kernel<S: Into<String>+Clone>(mut self, kernel: Kernel<S>) -> Self {
        self.kernels.push((kernels::convert(&kernel),None));
        self
    }

    pub fn load_kernel<S: Into<String>+Clone>(mut self, name: S) -> Self {
        self.kernels.push((kernels::convert(&self.available_kernels[&name.clone().into()]),Some(name.into())));
        self
    }

    pub fn load_kernel_named<S: Into<String>+Clone>(mut self, name: S, as_name: S) -> Self {
        self.kernels.push((kernels::convert(&self.available_kernels[&name.clone().into()]),Some(as_name.into())));
        self
    }

    pub fn build(self) -> ocl::Result<super::Handler> {
        let mut prog = String::new();
        for (Kernel {name,src,args},_) in &self.kernels {
            prog += &format!("\n__kernel void {}(\n",name.clone());
            for a in args {
                match a {
                    KernelDescriptor::Param(n,_) => 
                        prog += &format!("double {},", n.clone()),
                    KernelDescriptor::Buffer(n) | KernelDescriptor::BufArg(_,n) => 
                        prog += &format!("__global double *{},", n.clone())
                };
            }
            prog.pop(); // remove last unnescessary ","
            prog += ") {\n";
            prog += "    long x = get_global_id(0); long x_size = get_global_size(0);\n";
            prog += "    long y = get_global_id(0); long y_size = get_global_size(0);\n";
            prog += "    long z = get_global_id(0); long z_size = get_global_size(0);\n";
            prog += &src.clone();
            prog += "\n}\n";
        }

        let pq = ProQue::builder()
            .src(prog)
            .dims(1) //TODO should not be needed
            .build()?;

        let mut buffers = HashMap::new();
        for (name,desc) in self.buffers {
            match desc {
                BufferDescriptor::Len(val,len) =>
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
        for (Kernel {name,src: _,args},loadedname) in self.kernels {
            let mut kernel = pq.kernel_builder(&name.clone());
            for a in args {
                match a {
                    KernelDescriptor::Param(n,v) =>
                        kernel.arg_named(n,v),
                    KernelDescriptor::Buffer(n) => {
                        if loadedname.is_some() {
                            kernel.arg_named(n.clone(),None::<&ocl::Buffer<f64>>)
                        } else {
                            kernel.arg_named(n.clone(),&buffers[&n])
                        }
                    },
                    KernelDescriptor::BufArg(n,m) =>
                        if loadedname.is_some() {
                            kernel.arg_named(n,None::<&ocl::Buffer<f64>>)
                        } else {
                            kernel.arg_named(m,&buffers[&n])
                        }
                };
            }
            let name = loadedname.unwrap_or(name.into());
            kernels.insert(name,kernel.build()?);
        }

        Ok(super::Handler {
            _pq: pq,
            kernels,
            buffers
        })
    }


}
