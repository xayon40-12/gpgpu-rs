use gpgpu::integrators::{*,PDETokens::*,DiffDir::*};
use gpgpu::dim::DimDir::*;

fn main() -> gpgpu::Result<()> {
    let u = Indx(IndexingTypes::new_vector(2,"u","bound"));
    let v = Vect(vec![1.into(),2.into()]);
    let f = Forward(vec![X,Y]);
    println!("{}\n", Add(&Mul(&Symb("a"),&Const(1.into())),&Sub(&Div(&Symb("b"),&Symb("c")),&Symb("d"))).to_ocl());
    println!("{}\n", &Mul(&v,&Diff(&Diff(&u,&f),&f)).to_ocl());
    println!("{}\n", &Mul(&Diff(&Diff(&u,&f),&f),&v).to_ocl());
    println!("{}\n", &Mul(&u,&v).to_ocl());
    


    Ok(())
}
