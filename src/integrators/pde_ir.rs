pub use crate::dim::DimDir;
pub mod kt_scheme;
pub mod lexer_compositor;
use serde::{Deserialize, Serialize};
//use decorum::R64;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DiffDir {
    Forward(Vec<DimDir>),
    Backward(Vec<DimDir>),
}
use DiffDir::*;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Indexable {
    coord: [i32; 4],
    dim: usize,
    global_dim: usize,
    vec_dim: usize,
    var_name: String,
    boundary: String,
}

fn coords_str(c: &[i32; 4], dim: usize, vec_dim: usize) -> String {
    let coo = |i| {
        format!(
            "{}{}{}",
            ['x', 'y', 'z'][i],
            if c[i] <= 0 { "" } else { "+" },
            if c[i] == 0 {
                "".to_string()
            } else {
                format!("{}", c[i])
            }
        )
    };
    let res = (1..dim).fold(coo(0), |a, i| format!("{},{}", a, coo(i)));
    format!("{},{},{}", res, c[3], vec_dim)
}
impl Indexable {
    pub fn apply_idx(mut self, idx: &[i32; 4]) -> Self {
        for i in 0..4 {
            self.coord[i] += idx[i];
        }
        self
    }
    pub fn new_scalar(dim: usize, global_dim: usize, var_name: &str, boundary: &str) -> SPDETokens {
        if dim > 3 {
            panic!("Dimension of Indexable must be 1, 2 or 3.")
        }
        SPDETokens::Indx(Indexable {
            coord: [0; 4],
            dim,
            global_dim,
            vec_dim: 1,
            var_name: var_name.into(),
            boundary: boundary.into(),
        })
    }
    pub fn new_vector(
        var_dim: usize,
        global_dim: usize,
        vec_dim: usize,
        var_name: &str,
        boundary: &str,
    ) -> SPDETokens {
        Indexable::new_slice(
            var_dim,
            global_dim,
            vec_dim,
            &[0..vec_dim],
            var_name,
            boundary,
        )
    }
    pub fn new_slice(
        var_dim: usize,
        global_dim: usize,
        vec_dim: usize,
        slices: &[std::ops::Range<usize>],
        var_name: &str,
        boundary: &str,
    ) -> SPDETokens {
        if var_dim > 3 {
            panic!("Dimension of Indexable must be 1, 2 or 3.");
        }
        for slice in slices {
            if slice.end > vec_dim {
                if slice.len() == 1 {
                    panic!(
                        "Index ({}) out of bounds for var '{}' of vectarial dim {}.",
                        slice.start, var_name, vec_dim
                    );
                } else {
                    panic!(
                        "Slice ({:?}) out of bounds for var '{}' of vectarial dim {}.",
                        slice, var_name, vec_dim
                    );
                }
            }
        }
        SPDETokens::Vect(
            slices
                .iter()
                .flat_map(|slice| {
                    slice.clone().map(|i| {
                        let mut coord = [0; 4];
                        coord[3] = i as i32;
                        SPDETokens::Indx(Indexable {
                            coord,
                            dim: var_dim,
                            global_dim,
                            vec_dim,
                            var_name: var_name.into(),
                            boundary: boundary.into(),
                        })
                    })
                })
                .collect(),
            true,
        )
    }
    pub fn to_string(&self) -> String {
        format!(
            "{}({},{})",
            self.boundary,
            coords_str(&self.coord, self.global_dim, self.vec_dim),
            self.var_name
        )
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum SPDETokens {
    Add(Box<SPDETokens>, Vec<SPDETokens>, bool),
    Sub(Box<SPDETokens>, Vec<SPDETokens>, bool),
    Mul(Box<SPDETokens>, Vec<SPDETokens>, bool),
    Div(Box<SPDETokens>, Vec<SPDETokens>, bool),
    Pow(Box<SPDETokens>, Box<SPDETokens>, bool),
    Func(String, Vec<SPDETokens>, bool),
    Symb(String), // considered to be a scalar
    Const(f64),
    Vect(Vec<SPDETokens>, bool),
    Indx(Indexable),
}

impl SPDETokens {
    fn is_scalar(&self) -> bool {
        use SPDETokens::*;
        match self.clone().convert() {
            // the result under asume that everything as been converted so the .convert() is needed to guaranty the asumption
            Add(a, _, _) => a.is_scalar(),
            Sub(a, _, _) => a.is_scalar(),
            Mul(a, _, _) => a.is_scalar(),
            Div(a, _, _) => a.is_scalar(),
            Pow(..) => true,
            Func(..) => true,
            Symb(..) => true,
            Const(..) => true,
            Vect(..) => false,
            Indx(..) => true,
        }
    }

    fn dim(&self) -> Option<usize> {
        use SPDETokens::*;
        macro_rules! andthen {
            ($a:ident $b:ident) => { $a.and_then(|$a| $b.and_then(|$b| if $b == $a { Some($a) } else { None })) };
            (dim $a:ident dim  $b:ident) => { $a.dim().and_then(|$a| $b.dim().and_then(|$b| if $b == $a { Some($a) } else { None })) };
            (dim $a:ident  $b:ident) => { $a.dim().and_then(|$a| $b.and_then(|$b| if $b == $a { Some($a) } else { None })) };
            ($a:ident dim $b:ident) => { $a.and_then(|$a| $b.dim().and_then(|$b| if $b == $a { Some($a) } else { None })) };
            ($a:ident vec $v:ident) => {
                $v.iter().fold($a.dim(), |acc,i| {
                    let i = i.dim();
                    andthen!(acc i)
                })
            };
        }
        match self {
            Add(a, t, _) => andthen!(a vec t),
            Sub(a, t, _) => andthen!(a vec t),
            Mul(a, t, _) => andthen!(a vec t),
            Div(a, t, _) => andthen!(a vec t),
            Pow(a, b, _) => andthen!(dim a dim b),
            Func(_, v, _) => {
                let mut v = v.clone();
                let s = v.pop().expect("There must be at least one expr in a Func.");
                andthen!(s vec v)
            }
            Symb(_) => Some(1),
            Const(_) => Some(1),
            Vect(v, _) => Some(v.len()),
            Indx(..) => Some(1),
        }
    }
    fn _to_ocl(self) -> String {
        use SPDETokens::*;
        macro_rules! foldop {
            ($a:ident, $b:ident, $ext:literal, $op:tt) => {
                format!(
                    $ext,
                    $b.into_iter()
                        .map(|i| i._to_ocl())
                        .fold($a._to_ocl(), |u, v| format!(
                            "{} {} {}",
                            u,
                            stringify!($op),
                            v
                        ))
                )
            };
            (par $a:ident $op:tt $b:ident) => {
                foldop!($a, $b, "({})", $op)
            };
            ($a:ident $op:tt $b:ident) => {
                foldop!($a, $b, "{}", $op)
            };
        }
        match self {
            Add(a, b, _) => foldop!(par a + b),
            Sub(a, b, _) => foldop!(par a - b),
            Mul(a, b, _) => foldop!(a * b),
            Div(a, b, _) => foldop!(a / b),
            Pow(a, b, _) => format!("pow({},{})", a._to_ocl(), b._to_ocl()),
            Func(n, a, _) => format!(
                "{}({})",
                n,
                a.into_iter()
                    .map(|i| i._to_ocl())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Symb(a) => a,
            Const(a) => format!("{:e}", a),
            Indx(a) => a.to_string(),
            s @ _ => panic!("Not expected during SPDEToken::to_ocl: {:?}", s),
        }
    }
    pub fn to_ocl(&self) -> Vec<String> {
        use SPDETokens::*;
        match self.clone() {
            Vect(v, _) => v.into_iter().map(|i| i.convert()._to_ocl()).collect(),
            s @ _ => vec![s.convert()._to_ocl()],
        }
    }

    fn is_converted(&self) -> bool {
        use SPDETokens::*;
        match self {
            Add(_, _, b) => *b,
            Sub(_, _, b) => *b,
            Mul(_, _, b) => *b,
            Div(_, _, b) => *b,
            Pow(_, _, b) => *b,
            Func(_, _, b) => *b,
            Vect(_, b) => *b,
            _ => true,
        }
    }

    fn convert(self) -> Self {
        use ir_helper::*;
        use SPDETokens::*;
        macro_rules! conv{
            ($a:ident $op:tt $b:ident) => { $b.iter().fold(*$a,|u,v| u $op v) };
        }
        if self.is_converted() {
            self
        } else {
            match self {
                Add(a,b,_) => conv!(a + b),
                Sub(a,b,_) => conv!(a - b),
                Mul(a,b,_) => conv!(a * b),
                Div(a,b,_) => conv!(a / b),
                Pow(a,b,_) => a ^ b,
                Func(n,a,_) => func(&n,a),
                Vect(_,_) => panic!("Cannot convert SPDETokens::Vector, it should have been multiplied by another vector."),
                _ => self,
            }
        }
    }

    fn apply_indexable<F: Fn(Indexable) -> SPDETokens + Copy>(self, f: F) -> Self {
        use SPDETokens::*;
        macro_rules! apply {
            ($a:ident) => {
                Box::new($a.apply_indexable(f))
            };
            (vec $a:ident) => {
                $a.into_iter()
                    .map(|i| i.apply_indexable(f))
                    .collect::<Vec<_>>()
            };
            ($a:ident $op:ident|$c:ident $b:ident) => {
                $op(apply!($a),apply!(vec $b),$c)
            };
        }
        match self {
            Add(a, b, c) => apply!(a Add|c b),
            Sub(a, b, c) => apply!(a Sub|c b),
            Mul(a, b, c) => apply!(a Mul|c b),
            Div(a, b, c) => apply!(a Div|c b),
            Pow(a, b, c) => Pow(apply!(a), apply!(b), c),
            Func(n, a, c) => Func(n, apply!(vec a), c),
            Indx(a) => f(a),
            Vect(v, c) => Vect(apply!(vec v), c),
            _ => self,
        }
        .into()
    }

    fn apply_idx(self, idx: &[i32; 4]) -> Self {
        self.apply_indexable(|s| SPDETokens::Indx(s.apply_idx(idx)))
    }

    // WARNING: diff are considered to be multiplied by the inverse of dx: (f(x+1)-f(x))*ivdx
    // where ivdx = 1.0/dx
    fn apply_diff(self, dir: DiffDir) -> Self {
        use SPDETokens::*;

        let (coordinc, mut dirs) = match dir.clone() {
            Forward(dirs) => (1, dirs),
            Backward(dirs) => (0, dirs),
        };
        let div = |a: SPDETokens, d| {
            let mut idx = [0; 4];
            idx[d as usize] = coordinc;
            let l = a.clone().apply_idx(&idx);
            idx[d as usize] -= 1;
            let r = a.apply_idx(&idx);
            (l - r) * Symb(["ivdx", "ivdy", "ivdz"][d as usize].into())
        };
        let _dir = [DimDir::X, DimDir::Y, DimDir::Z];
        match self.convert() {
            Vect(v, _) => {
                if dirs.len() == 0 {
                    if v.len() > 3 {
                        panic!("Vect has dimension higher than 3, no divergence possible.");
                    }
                    dirs = v
                        .iter()
                        .enumerate()
                        .map(|(i, _)| _dir[i])
                        .collect::<Vec<_>>();
                }
                if v.len() != dirs.len() {
                    panic!("Could not apply diff on Vect as the dimension of the Vect and the diff array are different.")
                }
                let mut vals = v.into_iter().enumerate().map(|(i, t)| div(t, dirs[i]));
                let first = vals
                    .next()
                    .expect("There must be at least one element in Vect");
                vals.fold(first, |a, i| a + i)
            }
            a @ _ => {
                if dirs.len() == 0 {
                    match a.dim() {
                        Some(d) => {
                            if d > 3 { panic!("Diff cannot be applied to an expression of dim>3 (as the dim of a pde variable should not be greater than 3).") }
                            dirs = (0..d).map(|i| _dir[i]).collect::<Vec<_>>()
                        },
                        None => panic!("Diff must be applyed to an expression containing only sub-expression of the same dim (dim of a Const is 1)."),
                    }
                }
                let mut res = dirs
                    .into_iter()
                    .map(|d| div(a.clone(), d))
                    .collect::<Vec<_>>();
                if res.len() == 1 {
                    res.pop().unwrap()
                } else {
                    ir_helper::vect(res)
                }
            }
        }
    }
}

pub mod ir_helper {
    pub use super::*;
    pub use DiffDir::*;
    pub use SPDETokens::*;

    #[derive(Debug, Clone)]
    ///PDE descriptor
    pub struct DPDE {
        pub var_name: String,
        pub boundary: String,
        pub var_dim: usize,
        pub vec_dim: usize,
    }

    pub fn func<'a, T: Into<SPDETokens>>(n: &'a str, a: Vec<T>) -> SPDETokens {
        Func(
            n.to_string(),
            a.into_iter()
                .map(|i| i.into().convert())
                .collect::<Vec<_>>(),
            true,
        )
    }

    pub fn symb<'a>(n: &'a str) -> SPDETokens {
        Symb(n.to_string())
    }

    pub fn diff<T: Into<SPDETokens>, U: Into<DiffDir>>(a: T, d: U) -> SPDETokens {
        a.into().apply_diff(d.into())
    }

    pub fn kt<T: Into<SPDETokens>>(u: T, fu: T, eigs: Vec<T>, dirs: Vec<DimDir>) -> SPDETokens {
        use kt_scheme::*;
        let u = u.into().convert();
        let err = "first parameter of pde_id::ir_helper::kt function must be an SPDETokens::Indx.";
        let dim = match &u {
            SPDETokens::Indx(u) => u,
            SPDETokens::Vect(v, _) => match &v[0] {
                // FIXME all parts of a vector might not have the same dimension
                SPDETokens::Indx(u) => u,
                _ => panic!("{}", err),
            },
            _ => panic!("{}", err),
        }
        .dim;
        let fu = fu.into();
        let eigs = eigs.into_iter().map(|i| i.into()).collect::<Vec<_>>();
        let dirs = if dirs.len() == 0 {
            (0..dim).map(|i| i.into()).collect::<Vec<_>>()
        } else {
            dirs
        };
        let mut res = dirs
            .into_iter()
            .map(|i| kt(&u, &fu, &eigs, i as usize))
            .collect::<Vec<_>>();
        if res.len() == 1 {
            res.pop().unwrap()
        } else {
            Vect(res, true)
        }
    }

    pub fn vect(v: Vec<SPDETokens>) -> SPDETokens {
        Vect(v.into_iter().map(|i| i.convert()).collect(), true)
    }

    #[macro_export]
    macro_rules! vect {
        ($($val:expr),+) => {
            vect(vec![$($val),+])
        };
    }
}

impl<'a> From<&'a DiffDir> for DiffDir {
    fn from(a: &'a DiffDir) -> Self {
        a.clone()
    }
}

impl<'a> From<&'a SPDETokens> for SPDETokens {
    fn from(a: &'a SPDETokens) -> Self {
        a.clone()
    }
}

impl<'a> From<&'a str> for SPDETokens {
    fn from(a: &'a str) -> Self {
        Self::Symb(a.into())
    }
}

impl From<String> for SPDETokens {
    fn from(a: String) -> Self {
        Self::Symb(a)
    }
}

impl From<f64> for SPDETokens {
    fn from(a: f64) -> Self {
        Self::Const(a)
    }
}

use std::ops::*;
macro_rules! impl_cs {
    ($op:ident,$f:ident) => {
        impl<'a> $op<Box<SPDETokens>> for Box<SPDETokens> {
            type Output = SPDETokens;
            fn $f(self, other: Box<SPDETokens>) -> SPDETokens {
                SPDETokens::$f(*self, *other)
            }
        }
        impl<'a> $op<SPDETokens> for Box<SPDETokens> {
            type Output = SPDETokens;
            fn $f(self, other: SPDETokens) -> SPDETokens {
                SPDETokens::$f(*self, other)
            }
        }
        impl<'a> $op<Box<SPDETokens>> for SPDETokens {
            type Output = SPDETokens;
            fn $f(self, other: Box<SPDETokens>) -> SPDETokens {
                SPDETokens::$f(self, *other)
            }
        }
        impl<'a> $op<&'a SPDETokens> for SPDETokens {
            type Output = SPDETokens;
            fn $f(self, other: &'a SPDETokens) -> SPDETokens {
                SPDETokens::$f(self, other.clone())
            }
        }
        impl<'a> $op<SPDETokens> for &'a SPDETokens {
            type Output = SPDETokens;
            fn $f(self, other: SPDETokens) -> SPDETokens {
                SPDETokens::$f(self.clone(), other)
            }
        }
        impl<'a> $op<&'a SPDETokens> for &'a SPDETokens {
            type Output = SPDETokens;
            fn $f(self, other: &'a SPDETokens) -> SPDETokens {
                SPDETokens::$f(self.clone(), other.clone())
            }
        }
        impl $op<SPDETokens> for f64 {
            type Output = SPDETokens;
            fn $f(self, other: SPDETokens) -> SPDETokens {
                use SPDETokens::*;
                SPDETokens::$f(Const(self), other)
            }
        }
        impl $op<f64> for SPDETokens {
            type Output = SPDETokens;
            fn $f(self, other: f64) -> SPDETokens {
                use SPDETokens::*;
                SPDETokens::$f(Const(other), self)
            }
        }
        impl<'a> $op<SPDETokens> for &'a str {
            type Output = SPDETokens;
            fn $f(self, other: SPDETokens) -> SPDETokens {
                use SPDETokens::*;
                SPDETokens::$f(Symb(self.into()), other)
            }
        }
        impl<'a> $op<&'a str> for SPDETokens {
            type Output = SPDETokens;
            fn $f(self, other: &'a str) -> SPDETokens {
                use SPDETokens::*;
                SPDETokens::$f(Symb(other.into()), self)
            }
        }
        impl $op<SPDETokens> for String {
            type Output = SPDETokens;
            fn $f(self, other: SPDETokens) -> SPDETokens {
                use SPDETokens::*;
                SPDETokens::$f(Symb(self), other)
            }
        }
        impl $op<String> for SPDETokens {
            type Output = SPDETokens;
            fn $f(self, other: String) -> SPDETokens {
                use SPDETokens::*;
                SPDETokens::$f(Symb(other), self)
            }
        }
    };
}

fn similar(a: &SPDETokens, b: &SPDETokens) -> bool {
    a.is_scalar() == b.is_scalar()
}

//
macro_rules! compact{
    ($a:expr, $b:expr, $v:expr, $e:ident|$op:tt) => {{
        use SPDETokens::Const;
        let a = $a; let b = $b;
        if let Const(a) = a {
            if let Const(b) = b {
                let c = Const(a $op b);
                if $v.len() > 0 {
                    $e(Box::new(c),$v,true)
                } else {
                    c
                }
            } else {
                $v.push(b);
                $e(Box::new(Const(a)),$v,true)
            }
        } else {
            if let Const(b) = b {
                $v.push(a);
                $e(Box::new(Const(b)),$v,true)
            } else {
                $v.push(b);
                $e(Box::new(a),$v,true)
            }
        }
    }};
}

macro_rules! opconcat{
    ($e:ident|$op:tt $a:ident $b:ident) => {{
        let a = $a.convert();
        let b = $b.convert();
        if let $e(a,mut v,_) = a {
            if let $e(b,mut w,_) = b {
                v.append(&mut w);
                compact!(*a, *b, v, $e|$op)
            } else {
                compact!(*a, b, v, $e|$op)
            }
        } else {
            let mut v: Vec<SPDETokens> = vec![];
            compact!(a, b, v, $e|$op)
        }
    }};
    ($e:ident|$op:tt $inv:ident $a:ident $b:ident) => {{
        let a = $a.convert();
        let b = $b.convert();
        if let $e(a,mut v,_) = a {
            if let $e(b,w,_) = b {
                let ab = (*a,*b);
                if let (Const(a),Const(b)) = ab {
                    $e(Box::new($inv(Box::new(Const(a $op b)),w,true)),v,true)
                } else { //TODO further optimize (if a is an Add,Sub,... that start by a Const and same for b)
                    let (a,b) = ab;
                    v.push(b);
                    $e(Box::new($inv(Box::new(a),w,true)),v,true)
                }
            } else {
                compact!(*a, b, v, $e|$op)
            }
        } else {
            let mut v: Vec<SPDETokens> = vec![];
            compact!(a, b, v, $e|$op)
        }
    }};
}
impl Add for SPDETokens {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        use SPDETokens::*;
        let a = self.convert();
        let b = other.convert();
        if let Vect(a, _) = a {
            if let Vect(b, _) = b {
                if a.len() != b.len() {
                    panic!("Vect must have the same size to be added")
                } else {
                    Vect(
                        a.into_iter()
                            .zip(b.into_iter())
                            .map(|(a, b)| a + b)
                            .collect(),
                        true,
                    )
                }
            } else {
                panic!("Vect must be added to another Vect.")
            }
        } else if let Vect(..) = b {
            panic!("Vect must be added to another Vect.")
        } else {
            if !similar(&a, &b) {
                panic!("Cannot add Vect to scalar.");
            }
            opconcat!(Add|+ a b)
        }
    }
}
impl_cs! {Add,add}

impl Sub for SPDETokens {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        use SPDETokens::*;
        let a = self.convert();
        let b = other.convert();
        if let Vect(a, _) = a {
            if let Vect(b, _) = b {
                if a.len() != b.len() {
                    panic!("Vect must have the same size to be added")
                } else {
                    Vect(
                        a.into_iter()
                            .zip(b.into_iter())
                            .map(|(a, b)| a - b)
                            .collect(),
                        true,
                    )
                }
            } else {
                panic!("Vect must be added to another Vect.")
            }
        } else if let Vect(..) = b {
            panic!("Vect must be added to another Vect.")
        } else {
            if !similar(&a, &b) {
                panic!("Cannot substract Vect to scalar.");
            }
            opconcat!(Sub|- Add a b)
        }
    }
}
impl_cs! {Sub,sub}

impl Mul for SPDETokens {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        macro_rules! dist {
            (left $e:ident, $a:ident $b:ident $c:ident $d:ident) => {
                $e(
                    Box::new($a.clone() * $c),
                    $d.iter()
                        .map(|d| $a.clone() * d.clone())
                        .collect::<Vec<_>>(),
                    true,
                )
            };
            (right $e:ident, $a:ident $b:ident $c:ident $d:ident) => {
                $e(
                    Box::new($c * $b.clone()),
                    $d.iter()
                        .map(|d| d.clone() * $b.clone())
                        .collect::<Vec<_>>(),
                    true,
                )
            };
        }
        use SPDETokens::*;
        let a = Box::new(self.convert());
        let b = Box::new(other.convert());
        match *a {
            Add(c, d, _) => dist!(right Add, a b c d),
            Sub(c, d, _) => dist!(right Sub, a b c d),
            _ => match *b {
                Add(c, d, _) => dist!(left Add, a b c d),
                Sub(c, d, _) => dist!(left Sub, a b c d),
                _ => match *a {
                    Vect(a, _) => match *b {
                        Vect(b, _) => {
                            if a.len() != b.len() {
                                panic!("Vect must have the same len in SPDEToken::Mul")
                            } else {
                                let mut tmp = a.into_iter().enumerate().map(|(i, v)| {
                                    Mul(Box::new(v.convert()), vec![b[i].clone().convert()], true)
                                });
                                let first = tmp.next().unwrap();
                                tmp.fold(first, |a, i| a + i)
                            }
                        }
                        _ => {
                            let a = Vect(a, true);
                            opconcat!(Mul|* a b)
                        }
                    },
                    _ => opconcat!(Mul|* a b),
                },
            },
        }
    }
}
impl_cs! {Mul,mul}

impl BitXor for SPDETokens {
    type Output = Self;
    fn bitxor(self, other: Self) -> Self {
        use SPDETokens::*;
        let dead = || panic!("A vector cannot be un exponant.");
        let o = other.convert();
        let s = self.convert();
        if let Vect(..) = o {
            dead();
        }
        if let Vect(..) = s {
            if !o.is_scalar() {
                dead()
            } else if let Const(b) = o {
                if b.trunc() == b {
                    (0..b as u64 - 1).fold(s.clone(), |acc, _| acc * s.clone())
                } else {
                    panic!("Vect must be exponanciated to a natural number exponant.");
                }
            } else {
                panic!("Vect must be exponanciated to a natural number exponant.");
            }
        } else {
            let ab = (s, o);
            if let (Const(a), Const(b)) = ab {
                a.powf(b).into()
            } else {
                let (a, b) = ab;
                Pow(Box::new(a), Box::new(b), true)
            }
        }
    }
}
impl_cs! {BitXor,bitxor}

impl Div for SPDETokens {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        use SPDETokens::*;
        if !other.is_scalar() {
            panic!("Connat devide by a Vect.");
        }
        opconcat!(Div|/ Mul self other)
    }
}
impl_cs! {Div,div}
