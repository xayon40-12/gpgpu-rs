use ocl::ProQue;
use std::collections::{HashMap,HashSet};
use crate::descriptors::*;
use crate::kernels::{Kernel,self};
use crate::algorithms::{self,Algorithm};

pub struct HandlerBuilder {
    available_kernels: HashMap<&'static str,Kernel>,
    available_algorithms: HashMap<&'static str,Algorithm>,
    kernels: Vec<(Kernel,Option<&'static str>)>,
    algorithms: Vec<(Algorithm,Option<&'static str>)>,
    buffers: Vec<(&'static str,BufferDescriptor)>
}

impl HandlerBuilder {
    pub fn new() -> ocl::Result<HandlerBuilder> {
        Ok(HandlerBuilder {
            available_kernels: kernels::kernels(),
            available_algorithms: algorithms::algorithms(),
            kernels: Vec::new(),
            algorithms: Vec::new(),
            buffers: Vec::new()
        })
    }

    pub fn add_buffer(mut self, name: &'static str, desc: BufferDescriptor) -> Self {
        self.buffers.push((name,desc));
        self
    }

    pub fn add_buffers(self, buffers: Vec<(&'static str,BufferDescriptor)>) -> Self {
        let mut hand = self;
        for (name,desc) in buffers {
            hand = hand.add_buffer(name,desc);
        }
        hand
    }

    pub fn create_kernel(mut self, kernel: Kernel) -> Self {
        self.kernels.push((kernel,None));
        self
    }

    pub fn load_kernel(mut self, name: &'static str) -> Self {
        self.kernels.push((self.available_kernels.get(name).expect(&format!("kernel \"{}\" not found",name)).clone(),Some(name)));
        self
    }

    pub fn load_kernel_named(mut self, name: &'static str, as_name: &'static str) -> Self {
        self.kernels.push((self.available_kernels.get(name).expect(&format!("kernel \"{}\" not found",name)).clone(),Some(as_name)));
        self
    }

    pub fn create_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithms.push((algorithm,None));
        self
    }

    pub fn load_algorithm(mut self, name: &'static str) -> Self {
        self.algorithms.push((self.available_algorithms.get(name).expect(&format!("kernel \"{}\" not found",name)).clone(),Some(name)));
        self
    }

    pub fn load_algorithm_named(mut self, name: &'static str, as_name: &'static str) -> Self {
        self.algorithms.push((self.available_algorithms.get(name).expect(&format!("kernel \"{}\" not found",name)).clone(),Some(as_name)));
        self
    }

    pub fn build(mut self) -> ocl::Result<super::Handler> {
        let mut algorithms = HashMap::new();
        for (Algorithm { name,kernels,callback },loadedname) in self.algorithms {
            for k in kernels {
                let name = k.name;
                self.kernels.push((k,Some(name)));
            }
            let name = loadedname.unwrap_or(name);
            if algorithms.insert(name,callback).is_some() { panic!("Cannot add two algorithms with the same name \"{}\"",name) }
        }

        let mut prog = String::new();
        let mut kern_names = HashSet::new();
        for (Kernel {name,src,args},_) in &self.kernels {
            if !kern_names.insert(name) { panic!("Cannot add two kernels with the same name \"{}\"",name) }
            prog += &format!("\n__kernel void {}(\n",name);
            for a in args {
                match a {
                    KernelDescriptor::Param(n,_) => 
                        prog += &format!("double {},", n),
                    KernelDescriptor::Buffer(n) | KernelDescriptor::BufArg(_,n) => 
                        prog += &format!("__global double *{},", n)
                };
            }
            prog.pop(); // remove last unnescessary ","
            prog += ") {\n";
            prog += "    long x = get_global_id(0); long x_size = get_global_size(0);\n";
            prog += "    long y = get_global_id(0); long y_size = get_global_size(0);\n";
            prog += "    long z = get_global_id(0); long z_size = get_global_size(0);\n";
            prog += src;
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
                    if buffers.insert(name, pq.buffer_builder()
                                           .len(len)
                                           .fill_val(val)
                                           .build()?).is_some() { 
                        panic!("Cannot add two buffers with the same name \"{}\"",name) },
                BufferDescriptor::Data(data) =>
                    if buffers.insert(name, pq.buffer_builder()
                                           .len(data.len())
                                           .copy_host_slice(&data)
                                           .build()?).is_some() {
                        panic!("Cannot add two buffers with the same name \"{}\"",name) }
            };
        }

        let mut kernels = HashMap::new();
        for (Kernel {name,src: _,args},loadedname) in self.kernels {
            let mut kernel = pq.kernel_builder(name);
            for a in args {
                match a {
                    KernelDescriptor::Param(n,v) =>
                        kernel.arg_named(n,v),
                    KernelDescriptor::Buffer(n) => {
                        if loadedname.is_some() {
                            kernel.arg_named(n,None::<&ocl::Buffer<f64>>)
                        } else {
                            kernel.arg_named(n,&buffers[n])
                        }
                    },
                    KernelDescriptor::BufArg(n,m) =>
                        if loadedname.is_some() {
                            kernel.arg_named(n,None::<&ocl::Buffer<f64>>)
                        } else {
                            kernel.arg_named(m,&buffers[n])
                        }
                };
            }
            let name = loadedname.unwrap_or(name);
            if kernels.insert(name,kernel.build()?).is_some() { panic!("Cannot add two kernels with the same name \"{}\"",name) }
        }

        Ok(super::Handler {
            pq,
            kernels,
            algorithms,
            buffers
        })
    }


}
