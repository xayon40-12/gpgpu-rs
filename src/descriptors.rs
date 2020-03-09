pub enum KernelDescriptor<S: Into<String>+Clone> {
    Param(S,f64),
    Buffer(S)
}

pub enum BufferDescriptor {
    Len(usize),
    Data(Vec<f64>)
}
