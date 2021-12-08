pub mod handler;
pub use handler::Handler;

pub mod dim;
pub use dim::{Dim, DimDir};

pub mod algorithms;
pub mod descriptors;
pub mod functions;
pub mod integrators;
pub mod kernels;
pub mod pde_parser;
pub mod philox;

pub mod data_file;

pub type Result<T> = ocl::Result<T>;
