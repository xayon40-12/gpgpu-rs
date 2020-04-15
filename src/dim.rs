use ocl::SpatialDims;
use serde::{Deserialize,Serialize};

#[derive(Deserialize,Serialize,Debug,Clone,Copy)]
pub enum Dim {
    D1(usize),
    D2(usize,usize),
    D3(usize,usize,usize)
}

#[derive(Deserialize,Serialize,Debug,Clone,Copy)]
pub enum DimDir {
    X,
    Y,
    Z
}

impl From<Dim> for SpatialDims {
    fn from(dim: Dim) -> SpatialDims {
        match dim {
            Dim::D1(x) => SpatialDims::One(x),
            Dim::D2(x,y) => SpatialDims::Two(x,y),
            Dim::D3(x,y,z) => SpatialDims::Three(x,y,z)
        }
    }
}

impl From<[usize; 3]> for Dim {
    fn from(dim: [usize; 3]) -> Dim {
        if dim[2] > 1 {
            Dim::D3(dim[0],dim[1],dim[2])
        } else if dim [1] > 1 {
            Dim::D2(dim[0],dim[1])
        } else {
            Dim::D1(dim[0])
        }
    }
}

impl From<Dim> for [usize; 3] {
    fn from(dim: Dim) -> [usize; 3] {
        match dim {
            Dim::D1(x) => [x,1,1],
            Dim::D2(x,y) => [x,y,1],
            Dim::D3(x,y,z) => [x,y,z]
        }
    }
}

impl From<[u32; 3]> for Dim { fn from(dim: [u32; 3]) -> Dim { [dim[0] as usize,dim[1] as usize,dim[2] as usize].into() } }
impl From<[u64; 3]> for Dim { fn from(dim: [u64; 3]) -> Dim { [dim[0] as usize,dim[1] as usize,dim[2] as usize].into() } }
