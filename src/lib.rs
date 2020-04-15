#[macro_use] pub mod descriptors;

pub mod handler;
pub use handler::Handler;

pub use descriptors::{KernelArg,BufferConstructor};

pub mod dim;
pub use dim::{Dim,DimDir};

pub mod kernels;
pub mod algorithms;
pub mod functions;
pub mod philox;

pub mod data_file;

pub use ocl::prm::*;

pub type Result<T> = ocl::Result<T>;
