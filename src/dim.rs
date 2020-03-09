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
