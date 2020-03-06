use ocl::{ProQue,Buffer,Kernel};
use std::collections::HashMap; 

struct Param {
    id: usize,
    val: i64
}

pub struct Handler {
    pq: ProQue,
    kernel: Option<Kernel>,
    entry_point: String,
    buffers: HashMap<String,Buffer<i64>>,
    params: Vec<Param>,
    params_buffer: Option<Buffer<i64>>
}

impl Handler {
    pub fn new(src: &str) -> ocl::Result<Handler> {
        let pro_que = ProQue::builder()
            .src(src)
            .dims(1 << 20)
            .build()?;

        let buffer = pro_que.create_buffer::<f32>()?;

        let kernel = pro_que.kernel_builder("add")
            .arg(&buffer)
            .arg(10.0f32)
            .build()?;


        let mut vec = vec![0.0f32; buffer.len()];
        buffer.read(&mut vec).enq()?;

        println!("The value at index [{}] is now '{}'!", 200007, vec[200007]);

        Ok(Handler {})
    }

    pub fn run(&mut self, dim: &[u64]) -> ocl::Result<()> {
        if self.kernel.is_none() {
            self.kernel = Some(self.pq.kernel_builder(self.entry_point));
        }
        unsafe { self.kernel.enq()?; }
        Ok(())
    }

    pub fn get() {

    }

}
