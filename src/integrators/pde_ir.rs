use crate::dim::DimDir;
use serde::{Serialize,Deserialize};
//use decorum::R64;

#[derive(Debug,Clone,PartialEq,Eq,PartialOrd,Ord,Serialize,Deserialize)]
pub enum DiffDir {
    Forward(Vec<DimDir>),
    Backward(Vec<DimDir>),
}
use DiffDir::*;

#[derive(Debug,Clone,PartialEq,Eq,PartialOrd,Ord,Serialize,Deserialize)]
pub struct Indexable {
    coord: [i32;4],
    dim: usize,
    vector: bool,
    var_name: String,
    boundary: String,
}

fn coords_str(c: &[i32;4], dim: usize, vector: bool) -> String {
    let coo = |i| format!("{}{}{}",
        ['x','y','z'][i],
        if c[i]<=0 { "" } else { "+" },
        if c[i]==0 { "".to_string() } else { format!("{}",c[i]) 
    });
    let res = (1..dim).fold(coo(0), |a,i| format!("{},{}",a,coo(i)));
    if vector {
        format!("{},{}", res, c[3])
    } else {
        res
    }
}
impl Indexable {
    pub fn apply_idx(mut self, idx: &[i32;4]) -> Self{
        for i in 0..4 {
            self.coord[i] += idx[i];
        }
        self
    }
    pub fn new_scalar<'a>(dim: usize, var_name: &'a str, boundary: &'a str) -> SPDETokens {
        SPDETokens::Indx(Indexable { coord: [0;4], dim, vector: false, var_name: var_name.into(), boundary: boundary.into() })
    }
    pub fn new_vector<'a>(dim: usize, var_name: &'a str, boundary: &'a str) -> SPDETokens {
        if dim > 3 { panic!("Dimension of IndexingTypes::Vector must be 1, 2 or 3.") }
        SPDETokens::Vect((0..dim).map(|i| {
            let mut coord = [0;4];
            coord[3] = i as i32;
            SPDETokens::Indx(Indexable { coord, dim, vector: true, var_name: var_name.into(), boundary: boundary.into() })
        }).collect())
    }
    pub fn to_string(&self) -> String {
        format!("{}({},{})",self.boundary,coords_str(&self.coord,self.dim,self.vector),self.var_name)
    }
}

#[derive(Debug,Clone,PartialEq,PartialOrd,Serialize,Deserialize)]
pub enum SPDETokens {
    Add(Box<SPDETokens>,Vec<SPDETokens>),
    Sub(Box<SPDETokens>,Vec<SPDETokens>),
    Mul(Box<SPDETokens>,Vec<SPDETokens>),
    Div(Box<SPDETokens>,Vec<SPDETokens>),
    Diff(Box<SPDETokens>,DiffDir),
    Func(String,Box<SPDETokens>),
    Symb(String),
    Const(f64),
    Vect(Vec<SPDETokens>),
    Indx(Indexable),
}

impl SPDETokens {
    fn _to_ocl(self) -> String {
        use SPDETokens::*;
        macro_rules! foldop {
            ($a:ident, $b:ident, $ext:literal, $op:tt) =>
                { format!($ext, $b.into_iter().map(|i| i._to_ocl()).fold($a._to_ocl(),|u,v|format!("{} {} {}",u,stringify!($op),v))) };
            (par $a:ident $op:tt $b:ident) => { foldop!($a, $b, "({})", $op) };
            ($a:ident $op:tt $b:ident) => { foldop!($a, $b, "{}", $op) };
        }
        match self {
            Add(a,b) => foldop!(par a + b),
            Sub(a,b) => foldop!(par a - b),
            Mul(a,b) => foldop!(a * b),
            Div(a,b) => foldop!(a / b),
            Func(n,a) => format!("{}({})",n,a._to_ocl()),
            Symb(a) => a,
            Const(a) => format!("{:e}",a),
            Indx(a) => a.to_string(),
            s @ _ => panic!("Not expected during SPDEToken::to_ocl: {:?}", s),
        }
    }
    pub fn to_ocl(self) -> Vec<String> {
        use SPDETokens::*;
        match self {
            Diff(..) => self.convert().to_ocl(),
            Vect(v) => v.into_iter().map(|i| i.convert()._to_ocl()).collect(),
            s @ _ => vec![s.convert()._to_ocl()],
        }
    }

    fn convert(self) -> Self {
        use SPDETokens::*;
        use ir_helper::*;
        macro_rules! conv{
            ($a:ident $op:tt $b:ident) => { $b.iter().fold(*$a,|u,v| u $op v) };
        }
        match self {
            Add(a,b) => conv!(a + b),
            Sub(a,b) => conv!(a - b),
            Mul(a,b) => conv!(a * b),
            Div(a,b) => conv!(a / b),
            Func(n,a) => func(&n,*a),
            Diff(a,d) => diff(*a,d),
            Vect(_) => panic!("Cannot convert SPDETokens::Vector, it should have been multiplied by an other vector."),
            _ => self,
        }
    }

    fn apply_idx(self, idx: &[i32;4]) -> Self {
        use SPDETokens::*;
        macro_rules! applyidx{
            ($a:ident $op:tt $b:ident) => {
                $b.into_iter().fold($a.apply_idx(idx),|u,v| u $op v.apply_idx(idx))
            };
        }
        match self {
            Add(a,b) => applyidx!(a + b),
            Sub(a,b) => applyidx!(a - b),
            Mul(a,b) => applyidx!(a * b),
            Div(a,b) => applyidx!(a / b),
            Diff(a,d) => a.apply_diff(d).apply_idx(idx),
            Func(n,a) => Func(n,Box::new(a.apply_idx(idx))),
            Indx(a) => Indx(a.apply_idx(idx)),
            _ => self
        }.into()
    }

    // WARNING: diff are considered to be multiplied by the inverse of dx: (f(x+1)-f(x))*ivdx
    // where ivdx = 1.0/dx
    fn apply_diff(self, dir: DiffDir) -> Self {
        use SPDETokens::*;

        let (coordinc,dirs) = match dir.clone() {
            Forward(dirs) => (1,dirs),
            Backward(dirs) => (0,dirs)
        };
        let div = |a: SPDETokens,d| {
            let mut idx = [0;4];
            idx[d as usize] = coordinc;
            let l = a.clone().apply_idx(&idx);
            idx[d as usize] -= 1;
            let r = a.apply_idx(&idx);
            (l-r)*Symb(["ivdx","ivdy","ivdz"][d as usize].into())
        };
        match self {
            Diff(a,d) => a.apply_diff(d).apply_diff(dir),
            Vect(v) => {
                if v.len() != dirs.len() { panic!("Could not apply diff on Vect as the dimension of the Vect and the diff array are different.") }
                let mut vals = v.into_iter().enumerate().map(|(i,t)| {
                    div(t,dirs[i])
                });
                let first = vals.next().expect("There must be at least one element in Vect");
                vals.fold(first, |a,i| a+i)
            },
            a @ _ => {
                Vect(dirs.into_iter().map(|d| div(a.clone(),d)).collect())
            },
        }
    }
}

pub mod ir_helper {
    pub use super::*;
    pub use DiffDir::*;
    pub use SPDETokens::*;

    pub fn func<'a,T: Into<SPDETokens>>(n: &'a str, a: T) -> SPDETokens {
        Func(n.to_string(),Box::new(a.into().convert()))
    }

    pub fn symb<'a>(n: &'a str) -> SPDETokens {
        Symb(n.to_string())
    }

    pub fn diff<T: Into<SPDETokens>,U: Into<DiffDir>>(a: T, d: U) -> SPDETokens {
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
        impl<'a> $op<&'a SPDETokens> for SPDETokens {
            type Output = SPDETokens;
            fn $f(self, other: &'a SPDETokens) -> SPDETokens {
                SPDETokens::$f(self,other.clone())
            }
        }
        impl<'a> $op<SPDETokens> for &'a SPDETokens {
            type Output = SPDETokens;
            fn $f(self, other: SPDETokens) -> SPDETokens {
                SPDETokens::$f(self.clone(),other)
            }
        }
        impl<'a> $op<&'a SPDETokens> for &'a SPDETokens {
            type Output = SPDETokens;
            fn $f(self, other: &'a SPDETokens) -> SPDETokens {
                SPDETokens::$f(self.clone(),other.clone())
            }
        }
        impl $op<SPDETokens> for f64 {
            type Output = SPDETokens;
            fn $f(self, other: SPDETokens) -> SPDETokens {
                use SPDETokens::*;
                SPDETokens::$f(Const(self),other)
            }
        }
        impl $op<f64> for SPDETokens {
            type Output = SPDETokens;
            fn $f(self, other: f64) -> SPDETokens {
                use SPDETokens::*;
                SPDETokens::$f(Const(other),self)
            }
        }
        impl<'a> $op<SPDETokens> for &'a str {
            type Output = SPDETokens;
            fn $f(self, other: SPDETokens) -> SPDETokens {
                use SPDETokens::*;
                SPDETokens::$f(Symb(self.into()),other)
            }
        }
        impl<'a> $op<&'a str> for SPDETokens {
            type Output = SPDETokens;
            fn $f(self, other: &'a str) -> SPDETokens {
                use SPDETokens::*;
                SPDETokens::$f(Symb(other.into()),self)
            }
        }
        impl $op<SPDETokens> for String {
            type Output = SPDETokens;
            fn $f(self, other: SPDETokens) -> SPDETokens {
                use SPDETokens::*;
                SPDETokens::$f(Symb(self),other)
            }
        }
        impl $op<String> for SPDETokens {
            type Output = SPDETokens;
            fn $f(self, other: String) -> SPDETokens {
                use SPDETokens::*;
                SPDETokens::$f(Symb(other),self)
            }
        }
    };
}

macro_rules! opconcat{
    ($e:ident $a:ident $b:ident) => {{
        let a = $a.convert();
        let b = $b.convert();
        if let $e(a,mut v) = a {
            if let $e(b,mut w) = b {
                v.push(*b);
                v.append(&mut w);
            } else {
                v.push(b);
            }
            $e(a,v)
        } else {
            $e(Box::new(a),vec![b])
        }
    }};
    ($e:ident $inv:ident $a:ident $b:ident) => {{
        let a = $a.convert();
        let b = $b.convert();
        if let $e(a,mut v) = a {
            if let $e(b,w) = b {
                v.push(*b);
                $e(Box::new($inv(a,w)),v)
            } else {
                v.push(b);
                $e(a,v)
            }
        } else {
            $e(Box::new(a),vec![b])
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
                    Vect(a.into_iter().zip(b.into_iter()).map(|(a,b)| a + b).collect())
                }
            } else {
                panic!("Vect must be added to another Vect.")
            }
        } else if let Vect(_) = b {
            panic!("Vect must be added to another Vect.")
        } else {
            opconcat!(Add a b)
        }
    }
}
impl_cs!{Add,add}

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
                    Vect(a.into_iter().zip(b.into_iter()).map(|(a,b)| a - b).collect())
                }
            } else {
                panic!("Vect must be added to another Vect.")
            }
        } else if let Vect(_) = b {
            panic!("Vect must be added to another Vect.")
        } else {
            opconcat!(Sub Add a b)
        }
    }
}
impl_cs!{Sub,sub}


impl Mul for SPDETokens {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        macro_rules! dist{
            (left $e:ident, $a:ident $b:ident $c:ident $d:ident) => {
                $e(Box::new(Mul($a.clone(),vec![*$c]).convert()),$d.iter().map(|d| Mul($a.clone(),vec![d.clone()]).convert()).collect::<Vec<_>>())
            };
            (right $e:ident, $a:ident $b:ident $c:ident $d:ident) => {
                $e(Box::new(Mul($c,vec![*$b.clone()]).convert()),$d.iter().map(|d| Mul(Box::new(d.clone()),vec![*$b.clone()]).convert()).collect::<Vec<_>>())
            };
        }
        use SPDETokens::*;
        let a = Box::new(self);
        let b = Box::new(other);
        match *a {
            Add(c,d) => dist!(right Add, a b c d),
            Sub(c,d) => dist!(right Sub, a b c d),
            Diff(c,d) => c.apply_diff(d) * *b,
            _ => match *b {
                Add(c,d) => dist!(left Add, a b c d),
                Sub(c,d) => dist!(left Sub, a b c d),
                Diff(c,d) => *a * c.apply_diff(d),
                _ => match *a {
                    Vect(a) => match *b {
                        Vect(b) => if a.len() != b.len() { 
                            panic!("Vect must have the same len in SPDEToken::Mul") 
                        } else { 
                            let mut tmp = a.into_iter().enumerate().map(|(i,v)| Mul(Box::new(v.convert()),vec![b[i].clone().convert()]));
                            let first = tmp.next().unwrap();
                            tmp.fold(first,|a,i| Add(Box::new(a),vec![i]))
                        },
                        _ => panic!("Vect must be multiplied with Vect or Indx(Vector(..)). given {:?}.", b)
                    },
                    _ => opconcat!(Mul a b),
                },
            },
        }
    }
}
impl_cs!{Mul,mul}

impl Div for SPDETokens {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        use SPDETokens::*;
        opconcat!(Div Mul self other)
    }
}
impl_cs!{Div,div}
