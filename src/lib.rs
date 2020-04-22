pub mod handler;
pub use handler::Handler;

pub mod dim;
pub use dim::{Dim,DimDir};

pub mod kernels;
pub mod algorithms;
pub mod functions;
pub mod descriptors;
pub mod philox;
pub mod integrators;

pub mod data_file;

pub type Result<T> = ocl::Result<T>;
