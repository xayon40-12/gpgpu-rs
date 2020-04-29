use crate::dim::DimDir;
use serde::{Serialize,Deserialize};

#[derive(Debug,Clone,Serialize,Deserialize)]
pub enum DiffDir {
    Forward(Vec<DimDir>),
    Backward(Vec<DimDir>),
}
use DiffDir::*;

#[derive(Debug,Clone,PartialEq,Eq,PartialOrd,Ord,Serialize,Deserialize)]
pub struct Token {
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
impl Token {
    pub fn apply_idx(&mut self, idx: &[i32;4]) {
        for i in 0..4 {
            self.coord[i] += idx[i];
        }
    }
    pub fn to_string(&self) -> String {
        format!("{}({},{})",self.boundary,coords_str(&self.coord,self.dim,self.vector),self.var_name)
    }
}

#[derive(Debug,Clone,PartialEq,Eq,PartialOrd,Ord,Serialize,Deserialize)]
pub enum IndexingTypes {
    Vector(Vec<Token>),// for a real vector use one more coord in each token and set it manually to 0,1,2
    Scalar(Token),
}
use IndexingTypes::*;

impl IndexingTypes {
    pub fn new_scalar<'a>(dim: usize, var_name: &'a str, boundary: &'a str) -> IndexingTypes {
        Scalar(Token { coord: [0;4], dim, vector: false, var_name: var_name.into(), boundary: boundary.into() })
    }
    pub fn new_vector<'a>(dim: usize, var_name: &'a str, boundary: &'a str) -> IndexingTypes {
        if dim > 3 { panic!("Dimension of IndexingTypes::Vector must be 1, 2 or 3.") }
        Vector((0..dim).map(|i| {
            let mut coord = [0;4];
            coord[3] = i as i32;
            Token { coord, dim, vector: true, var_name: var_name.into(), boundary: boundary.into() }
        }).collect())
    }
    pub fn apply_idx(mut self, idx: &[i32;4]) -> Self {
        match &mut self {
            Vector(v) => v.iter_mut().for_each(|i| i.apply_idx(idx)),
            Scalar(s) => s.apply_idx(idx),
        }
        self
    }
}

#[derive(Debug,Clone)]
pub enum PDETokens<'a> {
    Add(&'a PDETokens<'a>,&'a PDETokens<'a>),
    Sub(&'a PDETokens<'a>,&'a PDETokens<'a>),
    Mul(&'a PDETokens<'a>,&'a PDETokens<'a>),
    Div(&'a PDETokens<'a>,&'a PDETokens<'a>),
    Diff(&'a PDETokens<'a>,&'a DiffDir),
    Func(&'a str,&'a PDETokens<'a>),
    // the usize is the dim of the vector that the funcion returns
    //TODO FuncVec(&'a str,&'a PDETokens<'a>,usize), 
    Symb(&'a str),
    Const(f64),
    Vect(&'a [&'a PDETokens<'a>]),
    Indx(IndexingTypes),
}

#[derive(Debug,Clone,Serialize,Deserialize)]
pub enum SPDETokens {
    Add(Box<SPDETokens>,Box<SPDETokens>),
    Sub(Box<SPDETokens>,Box<SPDETokens>),
    Mul(Box<SPDETokens>,Box<SPDETokens>),
    Div(Box<SPDETokens>,Box<SPDETokens>),
    Diff(Box<SPDETokens>,DiffDir),
    Func(String,Box<SPDETokens>),
    Symb(String),
    Const(f64),
    Vect(Vec<SPDETokens>),
    Indx(IndexingTypes),
}

impl<'a> From<&PDETokens<'a>> for SPDETokens {
    fn from(pde: &PDETokens<'a>) -> Self {
        match pde {
            PDETokens::Add(a,b) => Self::Add(Box::new((*a).into()),Box::new((*b).into())),
            PDETokens::Sub(a,b) => Self::Sub(Box::new((*a).into()),Box::new((*b).into())),
            PDETokens::Mul(a,b) => Self::Mul(Box::new((*a).into()),Box::new((*b).into())),
            PDETokens::Div(a,b) => Self::Div(Box::new((*a).into()),Box::new((*b).into())),
            PDETokens::Diff(a,d) => Self::Diff(Box::new((*a).into()),(*d).clone()),
            PDETokens::Func(n,a) => Self::Func((*n).into(),Box::new((*a).into())),
            PDETokens::Symb(a) => Self::Symb((*a).into()),
            PDETokens::Const(a) => Self::Const(*a),
            PDETokens::Vect(a) => Self::Vect(a.iter().map(|i| (*i).into()).collect::<Vec<_>>()),
            PDETokens::Indx(a) => Self::Indx(a.clone()),
        }
    }
}

impl<'a> PDETokens<'a> {
    pub fn to_ocl(&self) -> Vec<String> {
        SPDETokens::from(self).to_ocl()
    }
}

impl SPDETokens {
    pub fn _to_ocl(self) -> String {
        use SPDETokens::*;
        match self.convert() {
            Add(a,b) => format!("({} + {})", a._to_ocl(), b._to_ocl()),
            Sub(a,b) => format!("({} - {})", a._to_ocl(), b._to_ocl()),
            Mul(a,b) => format!("{} * {}", a._to_ocl(), b._to_ocl()),
            Div(a,b) => format!("{} / {}", a._to_ocl(), b._to_ocl()),
            Func(n,a) => format!("{}({})",n,a._to_ocl()),
            Symb(a) => a,
            Const(a) => format!("{:e}",a),
            Indx(Scalar(a)) => a.to_string(),
            s @ _ => panic!("Not expected during SPDEToken::to_ocl: {:?}", s),
        }
    }
    pub fn to_ocl(self) -> Vec<String> {
        use SPDETokens::*;
        match self {
            Indx(Vector(v)) => v.into_iter().map(|i| i.to_string()).collect(),
            Vect(v) => v.into_iter().map(|i| i._to_ocl()).collect(),
            s @ _ => vec![s._to_ocl()],
        }
    }

    fn convert(self) -> Self {
        use SPDETokens::*;
        match self {
            Mul(a,b) => {
                match *a {
                    Add(c,d) => Add(Box::new(Mul(c,b.clone())),Box::new(Mul(d,b))),
                    Sub(c,d) => Sub(Box::new(Mul(c,b.clone())),Box::new(Mul(d,b))),
                    Diff(c,d) => Mul(Box::new(c.apply_diff(d)),b).convert(),
                    _ => match *b {
                        Add(c,d) => Add(Box::new(Mul(a.clone(),c)),Box::new(Mul(a,d))),
                        Sub(c,d) => Sub(Box::new(Mul(a.clone(),c)),Box::new(Mul(a,d))),
                        Diff(c,d) => Mul(a,Box::new(c.apply_diff(d))).convert(),
                        _ => match *a {
                            Vect(a) => match *b {
                                Vect(b) => if a.len() != b.len() { 
                                    panic!("Vect must have the same len in SPDEToken::Mul") 
                                } else { 
                                    let mut tmp = a.into_iter().enumerate().map(|(i,v)| Mul(Box::new(v),Box::new(b[i].clone())));
                                    let first = tmp.next().unwrap();
                                    tmp.fold(first,|a,i| Add(Box::new(a),Box::new(i)))
                                },
                                Indx(Vector(b)) => {
                                    if a.len() != b.len() { 
                                        panic!("Vect and Indx(Vector(..)) must have the same len in SPDEToken::Mul") 
                                    } else { 
                                        let mut tmp = a.into_iter().enumerate().map(|(i,v)| Mul(Box::new(v),Box::new(Indx(Scalar(b[i].clone())))));
                                        let first = tmp.next().unwrap();
                                        tmp.fold(first,|a,i| Add(Box::new(a),Box::new(i)))
                                    }
                                },
                                _ => panic!("Vect must be multiplied with Vect or Indx(Vector(..)). given {:?}.", b)
                            },
                            Indx(Vector(a)) => match *b {
                                Vect(b) => if a.len() != b.len() { 
                                    panic!("Vect must have the same len in SPDEToken::Mul") 
                                } else {
                                    let mut tmp = a.into_iter().enumerate().map(|(i,v)| Mul(Box::new(Indx(Scalar(v))),Box::new(b[i].clone())));
                                    let first = tmp.next().unwrap();
                                    tmp.fold(first,|a,i| Add(Box::new(a),Box::new(i)))
                                },
                                Indx(Vector(b)) => {
                                    if a.len() != b.len() { 
                                        panic!("Vect and Indx(Vector(..)) must have the same len in SPDEToken::Mul") 
                                    } else { 
                                        let mut tmp = a.into_iter().enumerate().map(|(i,v)| Mul(Box::new(Indx(Scalar(v))),Box::new(Indx(Scalar(b[i].clone())))));
                                        let first = tmp.next().unwrap();
                                        tmp.fold(first,|a,i| Add(Box::new(a),Box::new(i)))
                                    }
                                },
                                _ => panic!("Vect must be multiplied with Vect or Indx(Vector(..)). given {:?}.", b)
                            },
                            _ => Mul(a,b),    
                        },
                    },
                }
            },
            Diff(a,d) => a.apply_diff(d),
            Vect(_) => panic!("Cannot convert SPDETokens::Vector, it should have been multiplied by an other vector."),
            Indx(a) => if let Scalar(_) = &a { Indx(a) } else { panic!("Cannot convert IndexingTypes::Vector, it should have been multiplied by an other vector.") },
            _ => self,
        }
    }

    fn apply_idx(self, idx: &[i32;4]) -> Self {
        use SPDETokens::*;
        match self {
            Add(a,b) => Add(Box::new(a.apply_idx(idx)),Box::new(b.apply_idx(idx))),
            Sub(a,b) => Sub(Box::new(a.apply_idx(idx)),Box::new(b.apply_idx(idx))),
            Mul(a,b) => Mul(Box::new(a.apply_idx(idx)),Box::new(b.apply_idx(idx))),
            Div(a,b) => Div(Box::new(a.apply_idx(idx)),Box::new(b.apply_idx(idx))),
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
            Mul(Box::new(Sub(Box::new(l),Box::new(r))),Box::new(Symb(["ivdx","ivdy","ivdz"][d as usize].into())))
        };
        match self {
            Diff(a,d) => a.apply_diff(d).apply_diff(dir),
            Vect(v) => {
                if v.len() != dirs.len() { panic!("Could not apply diff on Vect as the dimension of the Vect and the diff array are different.") }
                let mut vals = v.into_iter().enumerate().map(|(i,t)| {
                    div(t,dirs[i])
                });
                let first = vals.next().expect("There must be at least one element in Vect");
                vals.fold(first, |a,i| Add(Box::new(a),Box::new(i)))
            },
            Indx(Vector(v)) => {
                if v.len() != dirs.len() { panic!("Could not apply diff on Indx(Vector) as the dimension of the Vector and the diff array are different.") }
                let mut vals = v.into_iter().enumerate().map(|(i,t)| {
                    div(Indx(Scalar(t)),dirs[i])
                });
                let first = vals.next().expect("There must be at least one element in Vector");
                vals.fold(first, |a,i| Add(Box::new(a),Box::new(i)))
            },
            a @ _ => {
                Vect(dirs.into_iter().map(|d| div(a.clone(),d)).collect())
            },
        }
    }
}

