pub use crate::dim::DimDir;
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
    vector: bool,
    var_name: String,
    boundary: String,
}

fn coords_str(c: &[i32; 4], dim: usize, vector: bool) -> String {
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
    if vector {
        format!("{},{}", res, c[3])
    } else {
        res
    }
}
impl Indexable {
    pub fn apply_idx(mut self, idx: &[i32; 4]) -> Self {
        for i in 0..4 {
            self.coord[i] += idx[i];
        }
        self
    }
    pub fn new_scalar<'a>(dim: usize, var_name: &'a str, boundary: &'a str) -> SPDETokens {
        if dim > 3 {
            panic!("Dimension of Indexable must be 1, 2 or 3.")
        }
        SPDETokens::Indx(Indexable {
            coord: [0; 4],
            dim,
            vector: false,
            var_name: var_name.into(),
            boundary: boundary.into(),
        })
    }
    pub fn new_vector<'a>(
        var_dim: usize,
        vec_dim: usize,
        var_name: &'a str,
        boundary: &'a str,
    ) -> SPDETokens {
        Indexable::new_slice(var_dim, vec_dim, 0..vec_dim, var_name, boundary)
    }
    pub fn new_slice<'a>(
        var_dim: usize,
        vec_dim: usize,
        slice: std::ops::Range<usize>,
        var_name: &'a str,
        boundary: &'a str,
    ) -> SPDETokens {
        if var_dim > 3 {
            panic!("Dimension of Indexable must be 1, 2 or 3.");
        }
        if slice.end > vec_dim {
            panic!(
                "Slice ({:?}) out of bounds for var '{}' of vectarial dim {}.",
                slice, var_name, vec_dim
            );
        }
        SPDETokens::Vect(
            slice
                .map(|i| {
                    let mut coord = [0; 4];
                    coord[3] = i as i32;
                    SPDETokens::Indx(Indexable {
                        coord,
                        dim: var_dim,
                        vector: true,
                        var_name: var_name.into(),
                        boundary: boundary.into(),
                    })
                })
                .collect(),
        )
    }
    pub fn to_string(&self) -> String {
        format!(
            "{}({},{})",
            self.boundary,
            coords_str(&self.coord, self.dim, self.vector),
            self.var_name
        )
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum SPDETokens {
    Add(Box<SPDETokens>, Vec<SPDETokens>),
    Sub(Box<SPDETokens>, Vec<SPDETokens>),
    Mul(Box<SPDETokens>, Vec<SPDETokens>),
    Div(Box<SPDETokens>, Vec<SPDETokens>),
    Pow(Box<SPDETokens>, Box<SPDETokens>),
    Func(String, Vec<SPDETokens>),
    Symb(String), // considered to be a scalar
    Const(f64),
    Vect(Vec<SPDETokens>),
    Indx(Indexable),
}

impl SPDETokens {
    fn is_scalar(&self) -> bool {
        use SPDETokens::*;
        match self.clone().convert() {
            // the result under asume that everything as been converted se the .convert() is needed to guaranty the asumption
            Add(a, _) => a.is_scalar(),
            Sub(a, _) => a.is_scalar(),
            Mul(a, _) => a.is_scalar(),
            Div(a, _) => a.is_scalar(),
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
            Add(a, t) => andthen!(a vec t),
            Sub(a, t) => andthen!(a vec t),
            Mul(a, t) => andthen!(a vec t),
            Div(a, t) => andthen!(a vec t),
            Pow(a, b) => andthen!(dim a dim b),
            Func(_, v) => {
                let mut v = v.clone();
                let s = v.pop().expect("There must be at least one expr in a Func.");
                andthen!(s vec v)
            }
            Symb(_) => Some(1),
            Const(_) => Some(1),
            Vect(v) => Some(v.len()),
            Indx(indx) => Some(indx.dim),
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
            Add(a, b) => foldop!(par a + b),
            Sub(a, b) => foldop!(par a - b),
            Mul(a, b) => foldop!(a * b),
            Div(a, b) => foldop!(a / b),
            Pow(a, b) => format!("pow({},{})", a._to_ocl(), b._to_ocl()),
            Func(n, a) => format!(
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
            Vect(v) => v.into_iter().map(|i| i.convert()._to_ocl()).collect(),
            s @ _ => vec![s.convert()._to_ocl()],
        }
    }

    fn convert(self) -> Self {
        use ir_helper::*;
        use SPDETokens::*;
        macro_rules! conv{
            ($a:ident $op:tt $b:ident) => { $b.iter().fold(*$a,|u,v| u $op v) };
        }
        match self {
            Add(a,b) => conv!(a + b),
            Sub(a,b) => conv!(a - b),
            Mul(a,b) => conv!(a * b),
            Div(a,b) => conv!(a / b),
            Pow(a,b) => a ^ b,
            Func(n,a) => func(&n,a),
            Vect(_) => panic!("Cannot convert SPDETokens::Vector, it should have been multiplied by another vector."),
            _ => self,
        }
    }

    fn apply_idx(self, idx: &[i32; 4]) -> Self {
        use SPDETokens::*;
        macro_rules! applyidx{
            ($a:ident $op:tt $b:ident) => {
                $b.into_iter().fold($a.apply_idx(idx),|u,v| u $op v.apply_idx(idx))
            };
        }
        match self {
            Add(a, b) => applyidx!(a + b),
            Sub(a, b) => applyidx!(a - b),
            Mul(a, b) => applyidx!(a * b),
            Div(a, b) => applyidx!(a / b),
            Func(n, a) => Func(
                n,
                a.into_iter().map(|i| i.apply_idx(idx)).collect::<Vec<_>>(),
            ),
            Indx(a) => Indx(a.apply_idx(idx)),
            _ => self,
        }
        .into()
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
        match self {
            Vect(v) => {
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
                    Vect(res)
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
        )
    }

    pub fn symb<'a>(n: &'a str) -> SPDETokens {
        Symb(n.to_string())
    }

    pub fn diff<T: Into<SPDETokens>, U: Into<DiffDir>>(a: T, d: U) -> SPDETokens {
        a.into().apply_diff(d.into())
    }

    #[macro_export]
    macro_rules! vect {
        ($($val:expr),+) => {
            Vect(vec![$($val),+])
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
macro_rules! compact_or{
    ($a:expr, $b:expr, $e:ident|$op:tt, $or:expr) => {{
        use SPDETokens::Const;
        let a = $a; let b = $b;
        if let Const(a) = a {
            if let Const(b) = b {
                Const(a $op b)
            } else {
                $e(Box::new(Const(a)),$or(b))
            }
        } else {
            if let Const(b) = b {
                $e(Box::new(Const(b)),$or(a))
            } else {
                $e(Box::new(a),$or(b))
            }
        }
    }};
}

macro_rules! opconcat{
    ($e:ident|$op:tt $a:ident $b:ident) => {{
        let a = $a.convert();
        let b = $b.convert();
        if let $e(a,mut v) = a {
            if let $e(b,mut w) = b {
                v.append(&mut w);
                compact_or!(*a, *b, $e|$op, |b| {v.push(b); v})
            } else {
                compact_or!(*a, b, $e|$op, |b| {v.push(b); v})
            }
        } else {
            compact_or!(a, b, $e|$op, |b|{vec![b]})
        }
    }};
    ($e:ident|$op:tt $inv:ident $a:ident $b:ident) => {{
        let a = $a.convert();
        let b = $b.convert();
        if let $e(a,mut v) = a {
            if let $e(b,w) = b {
                let ab = (*a,*b);
                if let (Const(a),Const(b)) = ab {
                    $e(Box::new($inv(Box::new(Const(a $op b)),w)),v)
                } else { //TODO further optimize (if a is an Add,Sub,... that start by a Const and same for b)
                    let (a,b) = ab;
                    v.push(b);
                    $e(Box::new($inv(Box::new(a),w)),v)
                }
            } else {
                compact_or!(*a, b, $e|$op, |b| { v.push(b); v})
            }
        } else {
            compact_or!(a, b, $e|$op, |b|{vec![b]})
        }
    }};
}
impl Add for SPDETokens {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        use SPDETokens::*;
        let a = self;
        let b = other;
        if let Vect(a) = a {
            if let Vect(b) = b {
                if a.len() != b.len() {
                    panic!("Vect must have the same size to be added")
                } else {
                    Vect(
                        a.into_iter()
                            .zip(b.into_iter())
                            .map(|(a, b)| a + b)
                            .collect(),
                    )
                }
            } else {
                panic!("Vect must be added to another Vect.")
            }
        } else if let Vect(_) = b {
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
        let a = self;
        let b = other;
        if let Vect(a) = a {
            if let Vect(b) = b {
                if a.len() != b.len() {
                    panic!("Vect must have the same size to be added")
                } else {
                    Vect(
                        a.into_iter()
                            .zip(b.into_iter())
                            .map(|(a, b)| a - b)
                            .collect(),
                    )
                }
            } else {
                panic!("Vect must be added to another Vect.")
            }
        } else if let Vect(_) = b {
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
                    Box::new(Mul($a.clone(), vec![*$c]).convert()),
                    $d.iter()
                        .map(|d| Mul($a.clone(), vec![d.clone()]).convert())
                        .collect::<Vec<_>>(),
                )
            };
            (right $e:ident, $a:ident $b:ident $c:ident $d:ident) => {
                $e(
                    Box::new(Mul($c, vec![*$b.clone()]).convert()),
                    $d.iter()
                        .map(|d| Mul(Box::new(d.clone()), vec![*$b.clone()]).convert())
                        .collect::<Vec<_>>(),
                )
            };
        }
        use SPDETokens::*;
        let a = Box::new(self);
        let b = Box::new(other);
        match *a {
            Add(c, d) => dist!(right Add, a b c d),
            Sub(c, d) => dist!(right Sub, a b c d),
            _ => match *b {
                Add(c, d) => dist!(left Add, a b c d),
                Sub(c, d) => dist!(left Sub, a b c d),
                _ => match *a {
                    Vect(a) => match *b {
                        Vect(b) => {
                            if a.len() != b.len() {
                                panic!("Vect must have the same len in SPDEToken::Mul")
                            } else {
                                let mut tmp = a.into_iter().enumerate().map(|(i, v)| {
                                    Mul(Box::new(v.convert()), vec![b[i].clone().convert()])
                                });
                                let first = tmp.next().unwrap();
                                tmp.fold(first, |a, i| Add(Box::new(a), vec![i]))
                            }
                        }
                        _ => {
                            let a = Vect(a);
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
        if let Vect(_) = other {
            dead();
        }
        if let Vect(_) = self {
            if !other.is_scalar() {
                dead()
            } else if let Const(b) = other {
                if b.trunc() == b {
                    (0..b as u64 - 1).fold(self.clone(), |acc, _| acc * self.clone())
                } else {
                    panic!("Vect must be exponanciated to a natural number exponant.");
                }
            } else {
                panic!("Vect must be exponanciated to a natural number exponant.");
            }
        } else {
            let ab = (self, other);
            if let (Const(a), Const(b)) = ab {
                a.powf(b).into()
            } else {
                let (a, b) = ab;
                Pow(Box::new(a), Box::new(b))
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
