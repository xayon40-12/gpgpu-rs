use ocl::SpatialDims;

pub enum Dim {
    D1(usize),
    D2(usize,usize),
    D3(usize,usize,usize)
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
