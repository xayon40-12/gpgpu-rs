#[derive(Clone)]
pub enum KernelDescriptor<S: Into<String>+Clone> {
    Param(S,f64),
    Buffer(S),
    BufArg(S,S)
}

pub enum BufferDescriptor {
    Len(f64,usize), // Len(repeated value, len)
    Data(Vec<f64>)
}
