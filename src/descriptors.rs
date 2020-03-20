#[derive(Clone)]
pub enum KernelDescriptor<'a> {
    Param(&'a str,f64),
    Buffer(&'a str),
    BufArg(&'a str,&'a str) // BufArg(mem buf, kernel buf)
}

pub enum BufferDescriptor {
    Len(f64,usize), // Len(repeated value, len)
    Data(Vec<f64>)
}
