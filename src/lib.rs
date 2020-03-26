#[macro_use] pub mod descriptors;

pub mod handler;
pub use handler::Handler;

pub use descriptors::{KernelArg,BufferConstructor};

pub mod dim;
pub use dim::Dim;

pub mod kernels;
pub mod algorithms;
pub mod philox;

pub mod file;

pub type Result<T> = ocl::Result<T>;
