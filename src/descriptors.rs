#[derive(Clone)]
pub enum KernelDescriptor {
    Param(&'static str,f64),
    Buffer(&'static str),
    BufArg(&'static str,&'static str) // BufArg(mem buf, kernel buf)
}

pub enum BufferDescriptor {
    Len(f64,usize), // Len(repeated value, len)
    Data(Vec<f64>)
}
