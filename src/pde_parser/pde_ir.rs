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
pub enum Num {
    Real(f64),
    Complex(f64, f64),
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum SPDETokens {
    Add(Box<SPDETokens>, Vec<SPDETokens>, bool),
    Sub(Box<SPDETokens>, Box<SPDETokens>, bool),
    Mul(Box<SPDETokens>, Vec<SPDETokens>, bool),
    Div(Box<SPDETokens>, Box<SPDETokens>, bool),
    Pow(Box<SPDETokens>, Box<SPDETokens>, bool),
    Func(String, Vec<SPDETokens>, bool),
    Symb(String), // considered to be a scalar
    Const(f64),   //TODO use Num to handle complex number
    Vect(Vec<SPDETokens>, bool),
    Indx(Indexable),
}

fn vdim(v: &Vec<SPDETokens>) -> usize {
    v.iter().map(|i| i.dim()).fold(0, usize::max)
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

    fn indexables(&self) -> Vec<Indexable> {
        use SPDETokens::*;
        let all = |a: Option<SPDETokens>, s: Vec<SPDETokens>| {
            let mut ai = a.map(|i| i.indexables()).unwrap_or_default();
            s.iter().for_each(|i| ai.append(&mut i.indexables()));
            ai
        };
        match self.clone().convert() {
            // the result under asume that everything as been converted so the .convert() is needed to guaranty the asumption
            Add(a, s, _) => all(Some(*a), s),
            Sub(a, b, _) => all(None, vec![*a, *b]),
            Mul(a, s, _) => all(Some(*a), s),
            Div(a, b, _) => all(None, vec![*a, *b]),
            Pow(a, b, _) => all(None, vec![*a, *b]),
            Func(_, v, _) => all(None, v),
            Symb(..) => vec![],
            Const(..) => vec![],
            Vect(v, _) => all(None, v),
            Indx(i) => vec![i],
        }
    }

    fn is_indexable(&self) -> bool {
        use SPDETokens::*;
        let is = |v: Vec<Self>| v.iter().map(|i| i.is_indexable()).any(|b| b);
        match self.clone().convert() {
            // the result under asume that everything as been converted so the .convert() is needed to guaranty the asumption
            Add(a, s, _) => a.is_indexable() || is(s),
            Sub(a, b, _) => a.is_indexable() || b.is_indexable(),
            Mul(a, s, _) => a.is_indexable() || is(s),
            Div(a, b, _) => a.is_indexable() || b.is_indexable(),
            Pow(a, b, _) => a.is_indexable() || b.is_indexable(),
            Func(_, v, _) => is(v),
            Symb(..) => false,
            Const(..) => false,
            Vect(v, _) => is(v),
            Indx(..) => true,
        }
    }

    fn dim(&self) -> usize {
        use SPDETokens::*;
        macro_rules! andthen {
            ($a:ident $b:ident) => { usize::max($a,$b) };
            (dim $a:ident dim  $b:ident) => { usize::max($a.dim(), $b.dim()) };
            (dim $a:ident  $b:ident) => { usize::max($a.dim(),$b) };
            ($a:ident dim $b:ident) => { usize::max($a, $b.dim()) };
            ($a:ident vec $v:ident) => {
                $v.iter().fold($a.dim(), |acc,i| {
                    let i = i.dim();
                    andthen!(acc i)
                })
            };
        }
        match self {
            Add(a, t, _) => andthen!(a vec t),
            Sub(a, b, _) => andthen!(dim a dim b),
            Mul(a, t, _) => andthen!(a vec t),
            Div(a, b, _) => andthen!(dim a dim b),
            Pow(a, b, _) => andthen!(dim a dim b),
            Func(_, v, _) => vdim(v),
            Symb(_) => 0,
            Const(_) => 0,
            Vect(v, _) => vdim(v),
            Indx(i) => i.dim,
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
            Sub(a, b, _) => format!("({} - ({}))", a._to_ocl(), b._to_ocl()),
            Mul(a, b, _) => foldop!(a * b),
            Div(a, b, _) => format!("{} / ({})", a._to_ocl(), b._to_ocl()),
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
            s => panic!("Not expected during SPDEToken::to_ocl: {:?}", s),
        }
    }
    pub fn to_ocl(&self) -> Vec<String> {
        use SPDETokens::*;
        match self.clone().optimize() {
            Vect(v, _) => v.into_iter().map(|i| i._to_ocl()).collect(),
            s => vec![s._to_ocl()],
        }
    }

    pub fn optimize(self) -> Self {
        use ir_helper::*;
        use SPDETokens::*;
        //TODO optimize
        macro_rules! op {
            ($a:ident $op:tt $b:ident, $Op:ident) => {{
                let a = $a.optimize();
                let b = $b.optimize();
                if let (Const(a),Const(b)) = (a.clone(),b.clone()) { Const(a $op b) } else { $Op(Box::new(a),Box::new(b),true) }
            }};
            ($a:ident $op:tt vec $b:ident, $Op:ident) => {{ // NOTE here the vec b should be a monoid
                let mut consts = vec![];
                let mut rest = vec![];
                let a = $a.optimize();
                let b = $b.into_iter().map(|i| i.optimize()).collect::<Vec<_>>();
                if let Const(a) = a.clone() {
                    consts.push(a);
                } else {
                    rest.push(a);
                }
                for i in b {
                    if let Const(i) = i {
                        consts.push(i);
                    } else {
                        rest.push(i)
                    }
                }
                if consts.len() > 0 {
                    if rest.len() > 0{
                        $Op(Box::new(Const(consts.into_iter().fold(1.0, |a,i| a*i))), rest, true)
                    } else {
                        Const(consts.into_iter().fold(1.0, |a,i| a*i))
                    }
                } else {
                    let a = rest.remove(0);
                    $Op(Box::new(a),rest,true)
                }
            }};
        }
        let c = self.convert();
        match c {
            Add(a, b, _) => op! {a + vec b, Add},
            Sub(a, b, _) => op! {a - b, Sub},
            Mul(a, b, _) => op! {a * vec b, Mul},
            Div(a, b, _) => op! {a / b, Div},
            Pow(a, b, _) => {
                let a = a.optimize();
                let b = b.optimize();
                if let (Const(a), Const(b)) = (a.clone(), b.clone()) {
                    Const(a.powf(b))
                } else {
                    Pow(Box::new(a), Box::new(b), true)
                }
            }
            Func(n, a, _) => func(&n, a.into_iter().map(|i| i.optimize()).collect()),
            Vect(v, _) => vect(v.into_iter().map(|i| i.optimize()).collect()),
            s => s,
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
                Add(a, b, _) => conv!(a + b),
                Sub(a, b, _) => a - b,
                Mul(a, b, _) => conv!(a * b),
                Div(a, b, _) => a / b,
                Pow(a, b, _) => a ^ b,
                Func(n, a, _) => func(&n, a),
                Vect(v, _) => Vect(v.into_iter().map(|i| i.convert()).collect(), true),
                _ => self,
            }
        }
    }

    fn apply_indexable<F: Fn(Indexable) -> SPDETokens>(self, f: &F) -> Self {
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
            Sub(a, b, c) => Sub(apply!(a), apply!(b), c),
            Mul(a, b, c) => apply!(a Mul|c b),
            Div(a, b, c) => Div(apply!(a), apply!(b), c),
            Pow(a, b, c) => Pow(apply!(a), apply!(b), c),
            Func(n, a, c) => Func(n, apply!(vec a), c),
            Indx(a) => f(a),
            Vect(v, c) => Vect(apply!(vec v), c),
            _ => self,
        }
    }

    fn apply_idx(self, idx: &[i32; 4]) -> Self {
        self.apply_indexable(&|s: Indexable| SPDETokens::Indx(s.apply_idx(idx)))
    }

    fn is_zero(&self) -> bool {
        match self {
            SPDETokens::Const(a) => a == &0.0,
            _ => false,
        }
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
            Add(a, s, _) => s
                .into_iter()
                .map(|a| a.apply_diff(dir.clone()))
                .fold(a.apply_diff(dir.clone()), |a, b| a + b),
            Sub(a, b, _) => a.apply_diff(dir.clone()) - b.apply_diff(dir.clone()),
            Mul(a, s, _) => Add(
                Box::new(Mul(
                    Box::new(a.clone().apply_diff(dir.clone())),
                    s.clone(),
                    false,
                )),
                {
                    let mut res = vec![];
                    for i in 0..s.len() {
                        let di = s[i].clone().apply_diff(dir.clone());
                        if !di.is_zero() {
                            let mut sd = s.clone();
                            sd[i] = di;
                            res.push(Mul(a.clone(), sd, false));
                        }
                    }
                    res
                },
                false,
            )
            .convert(),
            Div(a, b, _) => {
                (a.clone().apply_diff(dir.clone()) * b.clone()
                    - a * b.clone().apply_diff(dir.clone()))
                    / b
                    ^ Const(2.0)
            }
            Pow(a, b, _) => {
                if b.is_indexable() {
                    panic!("Differenciation of Indexable at the exponent is not supported")
                } else {
                    b.clone() * a.clone() ^ (b - Const(1.0)) * a.apply_diff(dir.clone())
                }
            }
            Func(n, v, _) => {
                if dirs.len() == 0 {
                    let d = vdim(&v);
                    if d > 3 {
                        panic!("Diff cannot be applied to an expression of dim>3 (as the dim of a pde variable should not be greater than 3).")
                    }
                    dirs = (0..d).map(|i| _dir[i]).collect::<Vec<_>>()
                }
                let mut res = dirs
                    .into_iter()
                    .map(|d| div(Func(n.clone(), v.clone(), false), d))
                    .collect::<Vec<_>>();
                if res.len() == 1 {
                    res.pop().unwrap()
                } else {
                    ir_helper::vect(res)
                }
            }
            Symb(..) => 0.0.into(),
            Const(..) => 0.0.into(),
            Indx(i) => {
                if dirs.len() == 0 {
                    let d = i.dim;
                    if d > 3 {
                        panic!("Diff cannot be applied to an expression of dim>3 (as the dim of a pde variable should not be greater than 3).")
                    }
                    dirs = (0..d).map(|i| _dir[i]).collect::<Vec<_>>()
                }
                let mut res = dirs
                    .into_iter()
                    .map(|d| div(Indx(i.clone()), d))
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

    pub fn kt<T: Into<SPDETokens>>(
        u: T,
        fu: T,
        eigs: Vec<T>,
        theta: f64,
        dirs: Vec<DimDir>,
    ) -> SPDETokens {
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
            .map(|i| kt(&u, &fu, &eigs, theta, i as usize))
            .collect::<Vec<_>>();
        if res.len() == 1 {
            res.pop().unwrap()
        } else {
            vect(res)
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

impl Add for SPDETokens {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        use SPDETokens::*;
        let a = self.convert();
        let b = other.convert();
        if a.is_zero() {
            b
        } else if b.is_zero() {
            a
        } else if let Vect(a, _) = a {
            if let Vect(b, _) = b {
                if a.len() != b.len() {
                    panic!("Vect must have the same size to be added")
                } else {
                    ir_helper::vect(
                        a.into_iter()
                            .zip(b.into_iter())
                            .map(|(a, b)| a + b)
                            .collect(),
                    )
                }
            } else {
                panic!("The operation 'vector + scalar' is not allowed.")
            }
        } else if let Vect(..) = b {
            panic!("The operation 'scalar + vector' is not allowed.")
        } else if let Add(a1, mut b1, _) = a {
            if let Add(a2, mut b2, _) = b {
                b1.push(*a2);
                b1.append(&mut b2);
            } else {
                b1.push(b);
            }
            Add(a1, b1, true)
        } else if let Add(a2, mut b2, _) = b {
            b2.insert(0, *a2);
            Add(Box::new(a), b2, true)
        } else {
            Add(Box::new(a), vec![b], true)
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
        if a.is_zero() {
            Const(-1.0) * b
        } else if b.is_zero() {
            a
        } else if let Vect(a, _) = a {
            if let Vect(b, _) = b {
                if a.len() != b.len() {
                    panic!("Vect must have the same size to be substracted")
                } else {
                    ir_helper::vect(
                        a.into_iter()
                            .zip(b.into_iter())
                            .map(|(a, b)| a - b)
                            .collect(),
                    )
                }
            } else {
                panic!("The operation 'vector - scalar' is not allowed.")
            }
        } else if let Vect(..) = b {
            panic!("The operation 'scalar - vector' is not allowed.")
        } else if let Sub(a1, b1, _) = a {
            if let Sub(a2, b2, _) = b {
                Sub(Box::new(a1 + b2), Box::new(b1 + a2), true)
            } else {
                Sub(a1, Box::new(b1 + b), true)
            }
        } else if let Sub(a2, b2, _) = b {
            Sub(Box::new(a + b2), a2, true)
        } else {
            Sub(Box::new(a), Box::new(b), true)
        }
    }
}
impl_cs! {Sub,sub}

impl Mul for SPDETokens {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        use ir_helper::vect;
        use SPDETokens::*;
        let a = Box::new(self.convert());
        let b = Box::new(other.convert());
        if a.is_zero() || b.is_zero() {
            Const(0.0)
        } else {
            match *a {
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
                    _ => vect(a.into_iter().map(|v| v * b.clone()).collect()),
                },
                _ => match *b {
                    Vect(b, _) => vect(b.into_iter().map(|v| a.clone() * v).collect()),
                    _ => {
                        let a = *a;
                        let b = *b;
                        if let Mul(a1, mut b1, _) = a {
                            if let Mul(a2, mut b2, _) = b {
                                b1.push(*a2);
                                b1.append(&mut b2);
                            } else {
                                b1.push(b);
                            }
                            Mul(a1, b1, true)
                        } else if let Mul(a2, mut b2, _) = b {
                            b2.insert(0, *a2);
                            Mul(Box::new(a), b2, true)
                        } else {
                            Mul(Box::new(a), vec![b], true)
                        }
                    }
                },
            }
        }
    }
}
impl_cs! {Mul,mul}

impl BitXor for SPDETokens {
    type Output = Self;
    fn bitxor(self, other: Self) -> Self {
        use SPDETokens::*;
        let dead = || panic!("A vector cannot be an exponant.");
        let o = other.convert();
        let s = self.convert();
        if s.is_zero() {
            if o.is_zero() {
                panic!("Cannot exponentiate 0 by 0.")
            }
            Const(0.0)
        } else if o.is_zero() {
            if !s.is_scalar() {
                panic!("Vector to the 0th exponent are not supported.")
            }
            Const(1.0)
        } else {
            if let Vect(..) = o {
                dead();
            }
            if let Vect(..) = s {
                if let Const(b) = o {
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
                    if let Pow(a1, b1, _) = a {
                        Pow(a1, Box::new(b1 * b), true)
                    } else {
                        Pow(Box::new(a), Box::new(b), true)
                    }
                }
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
            panic!("Cannot divide by a Vect.");
        }
        if other.is_zero() {
            panic!("Cannot divide by zero.");
        }
        if self.is_zero() {
            Const(0.0)
        } else {
            match self.convert() {
                Vect(a, _) => ir_helper::vect(a.into_iter().map(|v| v / other.clone()).collect()),
                s => {
                    let a = s;
                    let b = other.convert();
                    if let Div(a1, b1, _) = a {
                        if let Div(a2, b2, _) = b {
                            Div(Box::new(a1 * b2), Box::new(b1 * a2), true)
                        } else {
                            Div(a1, Box::new(b1 * b), true)
                        }
                    } else if let Div(a2, b2, _) = b {
                        Div(Box::new(a * b2), a2, true)
                    } else {
                        Div(Box::new(a), Box::new(b), true)
                    }
                }
            }
        }
    }
}
impl_cs! {Div,div}
