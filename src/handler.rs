use ocl::{ProQue,Buffer,Kernel};
use std::collections::{HashMap,BTreeMap};

pub mod handler_builder;
pub use handler_builder::HandlerBuilder;

use crate::dim::Dim;
use crate::descriptors::{KernelArg,BufType,Type,VecType};
use crate::algorithms::Callback;
use crate::data_file::DataFile;

use std::any::{type_name,Any};

#[allow(dead_code)]
pub struct Handler {
    pq: ProQue,
    kernels: HashMap<String,(Kernel,BTreeMap<String,u32>)>,
    algorithms: HashMap<String,Callback>,
    buffers: HashMap<String,BufType>,
    data: HashMap<String,DataFile>,
}

impl Handler {
    pub fn builder<'a>() -> ocl::Result<HandlerBuilder<'a>> {
        HandlerBuilder::new()
    }

    fn get_buffer<T: ocl::OclPrm>(&self, name: &str) -> &Buffer<T> {
        let buf = self.buffers
            .get(name)
            .expect(&format!("Buffer \"{}\" not found",name));
        buf.iner_any()
            .downcast_ref::<Buffer<T>>()
            .expect(&format!("Wrong type for buffer \"{}\", expected {}, found {}",name,type_name::<T>(),buf.type_name()))
    }
    
    fn set_kernel_arg_buf(&self, name: &str, kernel: &(Kernel,BTreeMap<String,u32>), n: &str, m: &str) -> crate::Result<()>{
        let buf = self.buffers
            .get(n)
            .expect(&format!("Buffer \"{}\" not found",n));
        iner_each!(buf,BufType,buf,kernel.0.set_arg(*kernel.1.get(m).expect(&format!("Param \"{}\" not present in kernel \"{}\"",m,name)),buf))
    }

    pub fn get<T: ocl::OclPrm>(&self, name: &str) -> crate::Result<Vec<T>> {
        let buf = self.get_buffer(name);
        let len = buf.len();
        let mut vec = Vec::with_capacity(len);
        unsafe { vec.set_len(len); }
        buf.read(&mut vec).enq()?;
        Ok(vec)
    }

    pub fn get_first<T: ocl::OclPrm>(&self, name: &str) -> crate::Result<T> {
        let buf = self.get_buffer(name);
        let mut val = vec![Default::default()];
        buf.read(&mut val).enq()?;
        Ok(val[0])
    }

    fn _set_arg(&self, name: &str, desc: &[KernelArg], kernel: &(Kernel,BTreeMap<String,u32>)) -> crate::Result<()> {
        for d in desc {
            match d {
                KernelArg::Param(n,v) =>
                    iner_each!(v,Type,v,kernel.0.set_arg(*kernel.1.get(*n).expect(&format!("Param \"{}\" not present in kernel \"{}\"",n,name)),v)),
                KernelArg::Buffer(n) =>
                    self.set_kernel_arg_buf(name,kernel,n,n),
                KernelArg::BufArg(n,m) =>
                    self.set_kernel_arg_buf(name,kernel,n,m),
            }?;
        }
        Ok(())
    }

    pub fn set_arg(&mut self, name: &str, desc: &[KernelArg]) -> crate::Result<()> {
        let kernel = &self.kernels.get(name).expect(&format!("Kernel \"{}\" not found",name));
        self._set_arg(name,desc,kernel)
    }


    pub fn run(&mut self, name: &str, dim: Dim) -> crate::Result<()> {
        unsafe {
            self.kernels.get(name)
                .expect(&format!("Kernel \"{}\" not found",name))
                .0.cmd()
                .global_work_size(dim)
                .enq()
        }
    }

    pub fn run_arg(&mut self, name: &str, dim: Dim, desc: &[KernelArg]) -> ocl::Result<()> {
        let kernel = &self.kernels.get(name).expect(&format!("Kernel \"{}\" not found",name));
        self._set_arg(name,desc,kernel)?;

        unsafe {
            kernel.0.cmd().global_work_size(dim).enq()
        }
    }
    
    pub fn run_algorithm(&mut self, name: &str, dim: Dim, bufs: &[&str], other_args: Option<&dyn Any>) -> crate::Result<Option<Vec<VecType>>> {
        (self.algorithms.get(name).expect(&format!("Algorithm \"{}\" not found",name)).clone())(self,dim,bufs,other_args)
    }
}
