use ocl::SpatialDims;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone, Copy)]
pub enum Dim {
    D1(usize),
    D2(usize, usize),
    D3(usize, usize, usize),
}
impl Dim {
    pub fn len(&self) -> usize {
        match self {
            Self::D1(..) => 1,
            Self::D2(..) => 2,
            Self::D3(..) => 3,
        }
    }
    pub fn all_dirs(&self) -> Vec<DimDir> {
        use DimDir::*;
        match self {
            Self::D1(..) => vec![X],
            Self::D2(..) => vec![X, Y],
            Self::D3(..) => vec![X, Y, Z],
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
pub enum DimDir {
    X,
    Y,
    Z,
}

impl From<DimDir> for usize {
    fn from(i: DimDir) -> usize {
        (&i).into()
    }
}

impl From<&DimDir> for usize {
    fn from(i: &DimDir) -> usize {
        match i {
            DimDir::X => 0,
            DimDir::Y => 1,
            DimDir::Z => 2,
        }
    }
}

impl From<usize> for DimDir {
    fn from(i: usize) -> DimDir {
        match i {
            0 => DimDir::X,
            1 => DimDir::Y,
            2 => DimDir::Z,
            _ => panic!(
                "Wrong value to create DimDir, usize accepted {{0,1,2}}, given {}.",
                i
            ),
        }
    }
}

impl From<Dim> for SpatialDims {
    fn from(dim: Dim) -> SpatialDims {
        match dim {
            Dim::D1(x) => SpatialDims::One(x),
            Dim::D2(x, y) => SpatialDims::Two(x, y),
            Dim::D3(x, y, z) => SpatialDims::Three(x, y, z),
        }
    }
}

impl From<[usize; 3]> for Dim {
    fn from(dim: [usize; 3]) -> Dim {
        if dim[2] > 1 {
            Dim::D3(dim[0], dim[1], dim[2])
        } else if dim[1] > 1 {
            Dim::D2(dim[0], dim[1])
        } else {
            Dim::D1(dim[0])
        }
    }
}

impl From<Dim> for [usize; 3] {
    fn from(dim: Dim) -> [usize; 3] {
        match dim {
            Dim::D1(x) => [x, 1, 1],
            Dim::D2(x, y) => [x, y, 1],
            Dim::D3(x, y, z) => [x, y, z],
        }
    }
}

impl From<[u32; 3]> for Dim {
    fn from(dim: [u32; 3]) -> Dim {
        [dim[0] as usize, dim[1] as usize, dim[2] as usize].into()
    }
}
impl From<[u64; 3]> for Dim {
    fn from(dim: [u64; 3]) -> Dim {
        [dim[0] as usize, dim[1] as usize, dim[2] as usize].into()
    }
}
