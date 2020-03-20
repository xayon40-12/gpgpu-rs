#[macro_use] pub mod descriptors;

pub mod handler;
pub use handler::Handler;

pub use descriptors::{KernelDescriptor,BufferDescriptor};

pub mod dim;
pub use dim::Dim;

pub mod kernels;
pub mod algorithms;
pub mod philox;

pub type Result<T> = ocl::Result<T>;


#[cfg(test)]
mod test {
    use crate::Handler;
    use crate::descriptors::{BufferDescriptor::*,KernelDescriptor::*};
    use crate::kernels::Kernel;
    use crate::Dim;
    use crate::descriptors::Type::*;

    #[test]
    fn simple_main() -> crate::Result<()> {
        let src = "u[x] = p+x_size*1000+x*100;".to_string();

        let num = 8;
        let param_name = String::from("p");
        let mut gpu = Handler::builder()?
            .add_buffer("u", Len(0.0, num))
            .create_kernel(Kernel {
                name: "_main",
                src: &src,
                args: vec![Buffer("u"),Param(&param_name,F32(0.0))]
            })
            .build()?;

        for i in 0..10 {
            gpu.run("_main",Dim::D1(num),vec![Param("p",F32( i as f32))])?;
            assert_eq!(gpu.get::<f64>("u")?, (0..num).map(|j| i as f64 + num as f64*1000.0 + j as f64*100.0).collect::<Vec<_>>());
        }

        Ok(())
    }

    #[test]
    fn plus() -> crate::Result<()> {
        let num = 8;
        let mut gpu = Handler::builder()?
            .add_buffer("a", Data((0..num).map(|i| i as f64).collect()))
            .add_buffer("b", Len(2.0, num))
            .add_buffer("dst", Len(0.0, num))
            .load_kernel("plus")
            .build()?;

        gpu.run("plus",Dim::D1(num),vec![Buffer("a"),Buffer("b"),Buffer("dst")])?;
        assert_eq!(gpu.get::<f64>("dst")?, (0..num).map(|i| i as f64 + 2.0).collect::<Vec<_>>());

        Ok(())
    }

    #[test]
    fn times() -> crate::Result<()> {
        let num = 8;
        let mut gpu = Handler::builder()?
            .add_buffer("a", Data((0..num).map(|i| i as f64).collect()))
            .add_buffer("b", Len(2.0, num))
            .add_buffer("dst", Len(0.0, num))
            .load_kernel("times")
            .build()?;

        gpu.run("times",Dim::D1(num),vec![Buffer("a"),Buffer("b"),Buffer("dst")])?;
        assert_eq!(gpu.get::<f64>("dst")?, (0..num).map(|i| i as f64 * 2.0).collect::<Vec<_>>());

        Ok(())
    }

    #[test]
    fn sum() -> crate::Result<()> {
        let num = 8;
        let mut gpu = Handler::builder()?
            .add_buffer("src", Data((0..num).map(|i| i as f64).collect()))
            .add_buffer("dst", Len(0.0, num))
            .load_algorithm("sum")
            .build()?;

        gpu.run_algorithm("sum",Dim::D1(num),vec![Buffer("src"),Buffer("dst")])?;
        assert_eq!(gpu.get::<f64>("dst")?[0], (0..num).map(|i| i as f64).fold(0.0,|i,a| i+a));

        Ok(())
    }

}
