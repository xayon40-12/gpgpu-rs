use ocl::ProQue;
use std::collections::{HashMap,BTreeMap};
use crate::descriptors::*;
use crate::kernels::{self,Kernel};
use crate::algorithms::{self,Algorithm,Needed,Callback};
use crate::functions::{self,Function,Needed as FNeeded};
use crate::data_file::{DataFile,Format};

pub struct HandlerBuilder<'a> {
    available_kernels: HashMap<&'static str,Kernel<'a>>,
    available_algorithms: HashMap<&'static str,Algorithm<'a>>,
    available_functions: HashMap<String,Function<'a>>,
    kernels: HashMap<&'a str,(Kernel<'a>,&'a str)>,
    algorithms: HashMap<&'a str,(Callback,&'a str)>,
    functions: HashMap<String,(Function<'a>,String)>,
    buffers: Vec<(String,BufferConstructor)>,
    data: HashMap<&'a str, DataFile>,
}

impl<'a> HandlerBuilder<'a> {
    pub fn new() -> ocl::Result<HandlerBuilder<'a>> {
        Ok(HandlerBuilder {
            available_kernels: kernels::kernels(),
            available_algorithms: algorithms::algorithms(),
            available_functions: functions::functions(),
            kernels: HashMap::new(),
            algorithms: HashMap::new(),
            functions: HashMap::new(),
            buffers: Vec::new(),
            data: HashMap::new(),
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

    pub fn create_function(self, function: Function<'a>) -> Self {
        let name = function.name.clone().into();
        self.add_function(function, Some(name),None)
    }

    pub fn load_function(self, name: &str) -> Self {
        let function = self.available_functions.get(name).expect(&format!("function \"{}\" not found",name)).clone();
        self.add_function(function,None,None)
    }

    pub fn load_function_named(self, name: &str, as_name: &'a str) -> Self {
        let function = self.available_functions.get(name).expect(&format!("function \"{}\" not found",name)).clone();
        self.add_function(function,Some(as_name.into()),None)
    }
    
    fn add_function(mut self, mut function: Function<'a>, as_name: Option<String>, from: Option<String>) -> Self{
        let name = function.name.clone();
        let needed = function.needed;
        function.needed = vec![];

        if let Some(as_name) = as_name {
            if let Some((_,from)) = self.functions.get(&as_name) {
                panic!("Cannot add two functions with the same name \"{}\", already added by {}.",as_name,from);
            } else {
                self.functions.insert(as_name.into(),(function,"User".into()));
            }
        } else if let Some((_,from)) = self.functions.get(&name) {
            if from == &"User" {
                panic!("Cannot add two functions with the same name \"{}\", already added by User.",name);
            } else {
                return self;
            }
        } else {
            self.functions.insert(name.clone(),(function,from.unwrap_or("".into())));//TODO verify if empty string here causes problem
        }
        for n in needed {
            self = match n {
                FNeeded::FuncName(name) => self.load_function(&name),
                FNeeded::CreateFunc(func) => self.add_function(func,None,Some(format!("function \"{}\"",name))),
            }
        }


        self
    }

    pub fn create_kernel(self, kernel: Kernel<'a>) -> Self {
        let name = kernel.name;
        self.add_kernel(kernel, Some(name),None)
    }

    pub fn load_kernel(self, name: &str) -> Self {
        let kernel = self.available_kernels.get(name).expect(&format!("kernel \"{}\" not found",name)).clone();
        self.add_kernel(kernel,None,None)
    }

    pub fn load_kernel_named(self, name: &str, as_name: &'a str) -> Self {
        let kernel = self.available_kernels.get(name).expect(&format!("kernel \"{}\" not found",name)).clone();
        self.add_kernel(kernel,Some(as_name),None)
    }
    
    fn add_kernel(mut self, mut kernel: Kernel<'a>, as_name: Option<&'a str>, from_alg: Option<&'a str>) -> Self{
        let name = kernel.name;
        let needed = kernel.needed;
        kernel.needed = vec![];
        if let Some(as_name) = as_name {
            if let Some((_,from)) = self.kernels.get(as_name) {
                panic!("Cannot add two kernels with the same name \"{}\", already added by algorithm \"{}\".",as_name,from);
            } else {
                self.kernels.insert(as_name,(kernel,"User"));
            }
        } else if let Some((_,from)) = self.kernels.get(name) {
            if from == &"User" {
                panic!("Cannot add two kernels with the same name \"{}\", already added by User.",name);
            } else {
                return self;
            }
        } else {
            self.kernels.insert(name,(kernel,from_alg.unwrap_or("")));//TODO verify if empty string here causes problem
        }
        for n in needed {
            self = match n {
                FNeeded::FuncName(name) => self.load_function(&name),
                FNeeded::CreateFunc(func) => self.add_function(func,None,Some(format!("kernel \"{}\"",name))),
            }
        }

        self
    }

    pub fn create_algorithm(self, algorithm: Algorithm<'a>) -> Self {
        let name = algorithm.name;
        self.add_algorithm(algorithm, Some(name), None)
    }

    pub fn load_algorithm(self, name: &str) -> Self {
        let alg = self.available_algorithms.get(name).expect(&format!("algorithm \"{}\" not found",name)).clone();
        self.add_algorithm(alg, None, None)
    }

    pub fn load_algorithm_named(self, name: &str, as_name: &'a str) -> Self {
        assert_ne!(name, as_name, "Names must be different for method \"load_algorithm_named\", given name \"{}\"", name);
        let alg = self.available_algorithms.get(name).expect(&format!("algorithm \"{}\" not found",name)).clone();
        self.add_algorithm(alg, Some(as_name), None)
    }
    
    fn add_algorithm(mut self, alg: Algorithm<'a>, as_name: Option<&'a str>, from_alg: Option<&'a str>) -> Self{
        let Algorithm { name,callback,needed } = alg;
        if let Some(as_name) = as_name {
            if let Some((_,from)) = self.algorithms.get(as_name) {
                panic!("Cannot add two algorithms with the same name \"{}\", already added by algorithm \"{}\".",as_name,from);
            } else {
                self.algorithms.insert(as_name,(callback,"User"));
            }
        } else if let Some((_,from)) = self.algorithms.get(name) {
            if from == &"User" {
                panic!("Cannot add two algorithms with the same name \"{}\", already added by User.",name);
            } else {
                return self;
            }
        } else {
            self.algorithms.insert(name,(callback,from_alg.unwrap_or("")));//TODO verify if empty string here causes problem
        }
        for n in needed {
            self = match n {
                Needed::NewKernel(k) => self.create_kernel(k),
                Needed::KernelName(n) => {
                    let kernel = self.available_kernels.get(n).expect(&format!("kernel \"{}\" not found",n)).clone();
                    self.add_kernel(kernel,None,Some(as_name.unwrap_or(name)))
                },
                Needed::AlgorithmName(n) => {
                    let alg = self.available_algorithms.get(n).expect(&format!("algorithm \"{}\" not found",n)).clone();
                    self.add_algorithm(alg, None, Some(as_name.unwrap_or(name)))
                },
            }
        }

        self
    }

    pub fn build(self) -> ocl::Result<super::Handler> {
        let mut prog = String::new();

        for (name,(Function {src,args,ret_type,..},..)) in &self.functions {
            prog += &if let Some(ret) = ret_type {
                format!("\ninline {} {}(\n",ret.type_name_ocl(),name)
            } else {
                format!("\nvoid {}(\n",name)
            };
            for a in args {
                match a {
                    FunctionConstructor::Param(n,t) => 
                        prog += &format!("{} {},", t.type_name_ocl(), n),
                    FunctionConstructor::Ptr(n,t) =>
                        prog += &format!("{} *{},", t.type_name_ocl(), n),
                    FunctionConstructor::GlobalPtr(n,t) =>
                        prog += &format!("__global {} *{},", t.type_name_ocl(), n),
                    FunctionConstructor::ConstPtr(n,t) =>
                        prog += &format!("__constant {} *{},", t.type_name_ocl(), n)
                };
            }
            prog.pop(); // remove last unnescessary ","
            prog += ") {\n";
            prog += src;
            prog += "\n}\n";
        }

        for (name,(Kernel {src,args,..},..)) in &self.kernels {
            prog += &format!("\n__kernel void {}(\n",name);
            for a in args {
                match a {
                    KernelConstructor::Param(n,t) => 
                        prog += &format!("{} {},", t.type_name_ocl(), n),
                    KernelConstructor::Buffer(n,t) =>
                        prog += &format!("__global {} *{},", t.type_name_ocl(), n),
                    KernelConstructor::ConstBuffer(n,t) =>
                        prog += &format!("__constant {} *{},", t.type_name_ocl(), n)
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
        for (name,(Kernel {args,..},..)) in self.kernels {
            let mut map = BTreeMap::new();
            let mut kernel = pq.kernel_builder(name);
            let mut id = 0;
            for a in args {
                match a {
                    KernelConstructor::Param(n,v) => {
                        map.insert(n.to_string(),id); id += 1;
                        each_default!(param,v,kernel)
                    },
                    KernelConstructor::Buffer(n,b) | KernelConstructor::ConstBuffer(n,b) => {
                        map.insert(n.to_string(),id); id += 1;
                        each_default!(buffer,b,kernel)
                    }
                };
            }
            kernels.insert(name.to_string(),(kernel.build()?,map));
        }

        let algorithms = self.algorithms.into_iter().map(|(s,(c,_))| (s.to_string(),c)).collect();

        let data = self.data.into_iter().map(|(s,d)| (s.to_string(),d)).collect();

        Ok(super::Handler {
            pq,
            kernels,
            algorithms,
            buffers,
            data
        })
    }

    pub fn load_data(mut self, name: &'a str, data: Format<'a>) -> Self {
        let data = DataFile::parse(data);
        self = self.create_function(data.to_function(name));
        self.data.insert(name,data);
        self
    }


}
