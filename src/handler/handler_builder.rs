use ocl::ProQue;
use std::collections::{HashMap,HashSet,BTreeMap};
use crate::descriptors::*;
use crate::kernels::{Kernel,self};
use crate::algorithms::{self,Algorithm,Needed};

pub struct HandlerBuilder<'a> {
    available_kernels: HashMap<&'static str,Kernel<'a>>,
    available_algorithms: HashMap<&'static str,Algorithm<'a>>,
    kernels: Vec<(Kernel<'a>,BTreeMap<String,u32>,Option<String>)>,
    algorithms: Vec<(Algorithm<'a>,Option<String>)>,
    buffers: Vec<(String,BufferConstructor)>
}

impl<'a> HandlerBuilder<'a> {
    pub fn new() -> ocl::Result<HandlerBuilder<'a>> {
        Ok(HandlerBuilder {
            available_kernels: kernels::kernels(),
            available_algorithms: algorithms::algorithms(),
            kernels: Vec::new(),
            algorithms: Vec::new(),
            buffers: Vec::new()
        })
    }

    pub fn add_buffer(mut self, name: &str, desc: BufferConstructor) -> Self {
        self.buffers.push((name.to_string(),desc));
        self
    }

    pub fn add_buffers(self, buffers: Vec<(&str,BufferConstructor)>) -> Self {
        let mut hand = self;
        for (name,desc) in buffers {
            hand = hand.add_buffer(name,desc);
        }
        hand
    }

    pub fn create_kernel(mut self, kernel: Kernel<'a>) -> Self {
        self.kernels.push((kernel,BTreeMap::new(),None));
        self
    }

    pub fn load_kernel(mut self, name: &str) -> Self {
        self.kernels.push((self.available_kernels.get(name).expect(&format!("kernel \"{}\" not found",name)).clone(),BTreeMap::new(),None));
        self
    }

    pub fn load_kernel_named(mut self, name: &str, as_name: &str) -> Self {
        self.kernels.push((self.available_kernels.get(name).expect(&format!("kernel \"{}\" not found",name)).clone(),BTreeMap::new(),Some(as_name.to_string())));
        self
    }

    pub fn create_algorithm(mut self, algorithm: Algorithm<'a>) -> Self {
        self.algorithms.push((algorithm,None));
        self
    }

    pub fn load_algorithm(mut self, name: &str) -> Self {
        self.algorithms.push((self.available_algorithms.get(name).expect(&format!("kernel \"{}\" not found",name)).clone(),None));
        self
    }

    pub fn load_algorithm_named(mut self, name: &str, as_name: &str) -> Self {
        self.algorithms.push((self.available_algorithms.get(name).expect(&format!("kernel \"{}\" not found",name)).clone(),Some(as_name.to_string())));
        self
    }

    pub fn build(mut self) -> ocl::Result<super::Handler> {
        let mut algorithms = HashMap::new();
        for (Algorithm { name,callback,needed },loadedname) in self.algorithms {
            for n in needed {
                match n {
                    Needed::NewKernel(k) => {
                        let name = k.name;
                        self.kernels.push((k,BTreeMap::new(),Some(name.to_string())));
                    },
                    _ => panic!("Other Needed variantes not handled.")
                }
            }
            let name = loadedname.unwrap_or(name.to_string());
            if algorithms.insert(name.clone(),callback).is_some() { panic!("Cannot add two algorithms with the same name \"{}\"",name) }
        }

        let mut prog = String::new();
        let mut kern_names = HashSet::new();
        for (Kernel {name,src,args},_,_) in &self.kernels {
            if !kern_names.insert(name) { panic!("Cannot add two kernels with the same name \"{}\"",name) }
            prog += &format!("\n__kernel void {}(\n",name);
            for a in args {
                match a {
                    KernelConstructor::Param(n,t) => 
                        prog += &format!("{} {},", t.type_name_ocl(), n),
                    KernelConstructor::Buffer(n,t) => 
                        prog += &format!("__global {} *{},", t.type_name_ocl(), n)//TODO use buffer typename
                };
            }
            prog.pop(); // remove last unnescessary ","
            prog += ") {\n";
            prog += "    long x = get_global_id(0); long x_size = get_global_size(0);\n";
            prog += "    long y = get_global_id(1); long y_size = get_global_size(1);\n";
            prog += "    long z = get_global_id(2); long z_size = get_global_size(2);\n";
            prog += src;
            prog += "\n}\n";
        }

        let pq = ProQue::builder()
            .src(prog)
            .dims(1) //TODO should not be needed
            .build()?;

        let mut buffers = HashMap::new();
        for (name,desc) in self.buffers {
            let existing = match &desc {
                BufferConstructor::Len(val,len) => buffers.insert(name.clone(),
                    iner_each_gen!(val,Type BufType,val,
                        pq.buffer_builder()
                        .len(*len)
                        .fill_val(*val)
                        .build()?)),
                BufferConstructor::Data(data) => buffers.insert(name.clone(),
                    iner_each_gen!(data,VecType BufType,data,
                        pq.buffer_builder()
                       .len(data.len())
                       .copy_host_slice(data)
                       .build()?))
            };
            if existing.is_some() {
                panic!("Cannot add two buffers with the same name \"{}\"",name)
            }
        }

        let mut kernels = HashMap::new();
        for (Kernel {name,src: _,args},mut map,loadedname) in self.kernels {
            let mut kernel = pq.kernel_builder(name);
            let mut id = 0;
            for a in args {
                match a {
                    KernelConstructor::Param(n,v) => {
                        map.insert(n.to_string(),id); id += 1;
                        each_default!(param,v,kernel)
                    },
                    KernelConstructor::Buffer(n,b) => {
                        map.insert(n.to_string(),id); id += 1;
                        each_default!(buffer,b,kernel)
                    },
                };
            }
            let name = loadedname.unwrap_or(name.to_string());
            if kernels.insert(name.clone(),(kernel.build()?,map)).is_some() { panic!("Cannot add two kernels with the same name \"{}\"",name) }
        }

        Ok(super::Handler {
            pq,
            kernels,
            algorithms,
            buffers
        })
    }


}
