use ocl::{ProQue,Kernel};
use std::collections::{HashMap,BTreeMap};

pub mod handler_builder;
pub use handler_builder::HandlerBuilder;

use crate::dim::{Dim,DimDir};
use crate::descriptors::{KernelArg,BufferType,VecType,Type};
use crate::algorithms::Callback;
use crate::data_file::DataFile;

use std::any::{type_name,Any};

#[allow(dead_code)]
pub struct Handler {
    pq: ProQue,
    kernels: HashMap<String,(Kernel,BTreeMap<String,u32>)>,
    algorithms: HashMap<String,Callback>,
    buffers: HashMap<String,Box<dyn BufferType>>,
    data: HashMap<String,DataFile>,
}

impl Handler {
    pub fn builder() -> ocl::Result<HandlerBuilder> {
        HandlerBuilder::new()
    }
    
    fn set_kernel_arg_buf(&self, name: &str, kernel: &(Kernel,BTreeMap<String,u32>), n: &str, m: &str) -> crate::Result<()>{
        let buf = self.buffers
            .get(n)
            .expect(&format!("Buffer \"{}\" not found",n));
        buf.set_arg(&kernel,name,m)
    }

    pub fn get<T: 'static+VecType>(&self, name: &str) -> crate::Result<T> {
        let buf = self.buffers
            .get(name)
            .expect(&format!("Buffer \"{}\" not found",name));
        let tname = buf.type_name();
        Ok(*buf.get()?
            .downcast::<T>()
            .expect(&format!("Wrong type for buffer \"{}\", expected {}, found {}",name,type_name::<T>(),tname)))
    }

    pub fn get_first<T: 'static+Type>(&self, name: &str) -> crate::Result<T> {
        let buf = self.buffers
            .get(name)
            .expect(&format!("Buffer \"{}\" not found",name));
        let tname = buf.type_name();
        Ok(*buf.get_first()?
            .downcast::<T>()
            .expect(&format!("Wrong type for buffer \"{}\", expected {}, found {}",name,type_name::<T>(),tname)))
    }

    fn _set_arg(&self, name: &str, desc: &[KernelArg], kernel: &(Kernel,BTreeMap<String,u32>)) -> crate::Result<()> {
        for d in desc {
            match d {
                KernelArg::Param(n,v) =>
                    v.set_arg(kernel,name,n),
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
    
    pub fn run_algorithm(&mut self, name: &str, dim: Dim, dimdir: &[DimDir], bufs: &[&str], other_args: Option<&dyn Any>) -> crate::Result<()> {
        (self.algorithms.get(name).expect(&format!("Algorithm \"{}\" not found",name)).clone())(self,dim,dimdir,bufs,other_args)
    }

    pub fn copy(&mut self, src: &str, dst: &str) -> crate::Result<()> {//TODO add verbosity to error for copy (names of the buffers)
        self.buffers.get(src).expect(&format!("Buffer \"{}\" not found",src))
            .copy(self.buffers.get(dst).expect(&format!("Buffer \"{}\" not found",dst)).as_ref())
    }
}
