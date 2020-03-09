pub mod handler;
pub use handler::Handler;

pub mod descriptors;
pub use descriptors::{KernelDescriptor,BufferDescriptor};

pub mod dim;
pub use dim::Dim;

pub type Result<T> = ocl::Result<T>;
