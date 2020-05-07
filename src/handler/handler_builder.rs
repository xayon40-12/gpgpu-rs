use ocl::ProQue;
use std::collections::{HashMap,BTreeMap};
use crate::descriptors::*;
use crate::kernels::{self,Kernel,SKernel};
use crate::algorithms::{self,Algorithm,SAlgorithm,SNeeded,Callback};
use crate::functions::{self,Function,SFunction,SNeeded as FSNeeded};
use crate::data_file::{DataFile,Format};

pub struct HandlerBuilder {
    available_kernels: HashMap<&'static str,Kernel<'static>>,
    available_algorithms: HashMap<&'static str,Algorithm<'static>>,
    available_functions: HashMap<&'static str,Function<'static>>,
    kernels: HashMap<String,(SKernel,String)>,
    algorithms: HashMap<String,(Callback,String)>,
    functions: HashMap<String,(SFunction,String)>,
    buffers: Vec<(String,BufferConstructor)>,
    data: HashMap<String, DataFile>,
}

impl HandlerBuilder {
    pub fn new() -> ocl::Result<HandlerBuilder> {
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

    pub fn create_function(self, function: impl Into<SFunction>) -> Self {
        let function = function.into();
        let name = function.name.clone().into();
        self.add_function(function, Some(name),None)
    }

    pub fn load_function(self, name: &str) -> Self {
        let function = self.available_functions.get(name).expect(&format!("function \"{}\" not found",name)).clone();
        self.add_function(&function,None,None)
    }

    pub fn load_function_named(self, name: &str, as_name: &str) -> Self {
        let function = self.available_functions.get(name).expect(&format!("function \"{}\" not found",name)).clone();
        self.add_function(&function,Some(as_name.into()),None)
    }
    
    fn add_function(mut self, function: impl Into<SFunction>, as_name: Option<String>, from: Option<String>) -> Self{
        let mut function = function.into();
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
            if from == "User" {
                panic!("Cannot add two functions with the same name \"{}\", already added by User.",name);
            } else {
                return self;
            }
        } else {
            self.functions.insert(name.clone(),(function,from.unwrap_or("".into())));//TODO verify if empty string here causes problem
        }
        for n in needed {
            self = match n {
                FSNeeded::FuncName(name) => self.load_function(&name),
                FSNeeded::CreateFunc(func) => self.add_function(func,None,Some(format!("function \"{}\"",name))),
            }
        }


        self
    }

    pub fn create_kernel(self, kernel: impl Into<SKernel>) -> Self {
        let kernel = kernel.into();
        let name = kernel.name.clone();
        self.add_kernel(kernel, Some(name),None)
    }

    pub fn load_kernel(self, name: &str) -> Self {
        let kernel = self.available_kernels.get(name).expect(&format!("kernel \"{}\" not found",name)).clone();
        self.add_kernel(&kernel,None,None)
    }

    pub fn load_kernel_named(self, name: &str, as_name: &str) -> Self {
        let kernel = self.available_kernels.get(name).expect(&format!("kernel \"{}\" not found",name)).clone();
        self.add_kernel(&kernel,Some(as_name.into()),None)
    }
    
    fn add_kernel(mut self, kernel: impl Into<SKernel>, as_name: Option<String>, from_alg: Option<String>) -> Self{
        let mut kernel = kernel.into();
        let name = kernel.name.clone();
        let needed = kernel.needed;
        kernel.needed = vec![];
        if let Some(as_name) = as_name {
            if let Some((_,from)) = self.kernels.get(&as_name) {
                panic!("Cannot add two kernels with the same name \"{}\", already added by algorithm \"{}\".",as_name,from);
            } else {
                self.kernels.insert(as_name,(kernel,"User".into()));
            }
        } else if let Some((_,from)) = self.kernels.get(&name) {
            if from == "User" {
                panic!("Cannot add two kernels with the same name \"{}\", already added by User.",name);
            } else {
                return self;
            }
        } else {
            self.kernels.insert(name.clone(),(kernel,from_alg.unwrap_or("".into())));//TODO verify if empty string here causes problem
        }
        for n in needed {
            self = match n {
                FSNeeded::FuncName(name) => self.load_function(&name),
                FSNeeded::CreateFunc(func) => self.add_function(func,None,Some(format!("kernel \"{}\"",name))),
            }
        }

        self
    }

    pub fn create_algorithm(self, algorithm: impl Into<SAlgorithm>) -> Self {
        let algorithm = algorithm.into();
        let name = algorithm.name.clone();
        self.add_algorithm(algorithm, Some(name), None)
    }

    pub fn load_algorithm(self, name: &str) -> Self {
        let alg = self.available_algorithms.get(name).expect(&format!("algorithm \"{}\" not found",name)).clone();
        self.add_algorithm(&alg, None, None)
    }
    
    pub fn load_all_algorithms(mut self) -> Self {
        for alg in self.available_algorithms.values().map(|a| a.clone()).collect::<Vec<_>>() {
            self = self.add_algorithm(&alg, None, None);
        }

        self
    }

    pub fn load_algorithm_named(self, name: &str, as_name: &str) -> Self {
        assert_ne!(name, as_name, "Names must be different for method \"load_algorithm_named\", given name \"{}\"", name);
        let alg = self.available_algorithms.get(name).expect(&format!("algorithm \"{}\" not found",name)).clone();
        self.add_algorithm(&alg, Some(as_name.into()), None)
    }
    
    fn add_algorithm(mut self, alg: impl Into<SAlgorithm>, as_name: Option<String>, from_alg: Option<String>) -> Self{
        let alg = alg.into();
        let SAlgorithm { name,callback,needed } = alg;
        if let Some(as_name) = as_name.clone() {
            if let Some((_,from)) = self.algorithms.get(&as_name) {
                panic!("Cannot add two algorithms with the same name \"{}\", already added by algorithm \"{}\".",as_name,from);
            } else {
                self.algorithms.insert(as_name,(callback,"User".into()));
            }
        } else if let Some((_,from)) = self.algorithms.get(&name) {
            if from == "User" {
                panic!("Cannot add two algorithms with the same name \"{}\", already added by User.",name);
            } else {
                return self;
            }
        } else {
            self.algorithms.insert(name.clone(),(callback,from_alg.unwrap_or("".into())));//TODO verify if empty string here causes problem
        }
        for n in needed {
            self = match n {
                SNeeded::NewKernel(k) => self.create_kernel(k),
                SNeeded::KernelName(n) => {
                    let s: &str = &n;
                    let kernel = self.available_kernels.get(s).expect(&format!("kernel \"{}\" not found",n)).clone();
                    self.add_kernel(&kernel,None,Some(as_name.clone().unwrap_or(name.clone())))
                },
                SNeeded::AlgorithmName(n) => {
                    let s: &str = &n;
                    let alg = self.available_algorithms.get(s).expect(&format!("algorithm \"{}\" not found",n)).clone();
                    self.add_algorithm(&alg, None, Some(as_name.clone().unwrap_or(name.clone())))
                },
            }
        }

        self
    }

    pub fn source_code(&self) -> String {
        let mut prog = String::new();

        for (name,(SFunction {src,args,ret_type,..},..)) in &self.functions {
            prog += &if let Some(ret) = ret_type {
                format!("\ninline {} {}(",ret.type_name_ocl(),name)
            } else {
                format!("\nvoid {}(",name)
            };
            for a in args {
                match a {
                    SFunctionConstructor::FCParam(n,t) => 
                        prog += &format!("{} {},", t.type_name_ocl(), n),
                    SFunctionConstructor::FCPtr(n,t) =>
                        prog += &format!("{} *{},", t.type_name_ocl(), n),
                    SFunctionConstructor::FCGlobalPtr(n,t) =>
                        prog += &format!("__global {} *{},", t.type_name_ocl(), n),
                    SFunctionConstructor::FCConstPtr(n,t) =>
                        prog += &format!("__constant {} *{},", t.type_name_ocl(), n)
                };
            }
            prog.pop(); // remove last unnescessary ","
            prog += ") {\n";
            prog += src;
            prog += "\n}\n";
        }

        for (name,(SKernel {src,args,..},..)) in &self.kernels {
            prog += &format!("\n__kernel void {}(",name);
            for a in args {
                match a {
                    SKernelConstructor::KCParam(n,t) => 
                        prog += &format!("{} {},", t.type_name_ocl(), n),
                    SKernelConstructor::KCBuffer(n,t) =>
                        prog += &format!("__global {} *{},", t.type_name_ocl(), n),
                    SKernelConstructor::KCConstBuffer(n,t) =>
                        prog += &format!("__constant {} *{},", t.type_name_ocl(), n)
                };
            }
            prog.pop(); // remove last unnescessary ","
            prog += ") {\n";
            prog += "    uint x = get_global_id(0); uint x_size = get_global_size(0);\n";
            prog += "    uint y = get_global_id(1); uint y_size = get_global_size(1);\n";
            prog += "    uint z = get_global_id(2); uint z_size = get_global_size(2);\n";
            prog += src;
            prog += "\n}\n";
        }

        prog
    }

    pub fn build(self) -> ocl::Result<super::Handler> {
        let prog = self.source_code();

        let pq = ProQue::builder()
            .src(prog)
            .dims(1) //TODO should not be needed
            .build()?;

        let mut buffers = HashMap::new();
        for (name,desc) in self.buffers {
            let existing = match &desc {
                BufferConstructor::Len(val,len) => buffers.insert(name.clone(),
                    val.gen_buffer(&pq, *len)?),
                BufferConstructor::Data(data) => buffers.insert(name.clone(),
                    data.gen_buffer(&pq)?),
            };
            if existing.is_some() {
                panic!("Cannot add two buffers with the same name \"{}\"",name)
            }
        }

        let mut kernels = HashMap::new();
        for (name,(SKernel {args,..},..)) in self.kernels {
            let mut map = BTreeMap::new();
            let mut kernel = pq.kernel_builder(name.clone());
            let mut id = 0;
            for a in args {
                match a {
                    SKernelConstructor::KCParam(n,v) => {
                        map.insert(n.to_string(),id); id += 1;
                        v.default_param(&mut kernel);
                    },
                    SKernelConstructor::KCBuffer(n,b) | SKernelConstructor::KCConstBuffer(n,b) => {
                        map.insert(n.to_string(),id); id += 1;
                        b.default_buffer(&mut kernel);
                    }
                };
            }
            kernels.insert(name,(kernel.build()?,map));
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

    pub fn load_data(mut self, name: &str, data: Format, interpolated: bool, huge_file_buf_name: Option<&str>) -> Self {
        let data = DataFile::parse(data);
        self = self.create_function(if interpolated { data.to_function_interpolated(name, huge_file_buf_name.is_some()) } else { data.to_function(name, huge_file_buf_name.is_some()) });
        if let Some(bufname) = huge_file_buf_name {
            self = self.add_buffer(bufname, BufferConstructor::Data(data.data.clone().into()));
        }
        self.data.insert(name.into(),data);
        self
    }


}
