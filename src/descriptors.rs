pub enum KernelDescriptor<S: Into<String>+Clone> {
    Param(S,f64),
    Buffer(S),
    BufDst(S),
    BufSrc(S)
}

pub enum BufferDescriptor {
    Len(usize,f64), // Len(len, repeated value)
    Data(Vec<f64>)
}
