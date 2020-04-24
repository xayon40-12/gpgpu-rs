use crate::descriptors::{KernelConstructor::*,ConstructorTypes::*,KernelArg::*};
use crate::functions::Needed::*;
use crate::kernels::{Kernel};
use crate::algorithms::{SAlgorithm,SNeeded::*};
use crate::Handler;
use crate::dim::{Dim,DimDir};
use std::any::Any;
use serde::{Serialize,Deserialize};
use crate::descriptors::Types;

#[derive(Clone)]
pub struct PDE<'a> {
    dependant_var: &'a str,
    expr: &'a str,
}

#[derive(Clone,Serialize,Deserialize)]
pub struct SPDE {
    dependant_var: String,
    expr: String,
}

impl<'a> From<&PDE<'a>> for SPDE {
    fn from(de: &PDE) -> SPDE {
        SPDE {
            dependant_var: de.dependant_var.into(),
            expr: de.expr.into(),
        }
    }
}

#[derive(Clone,Serialize,Deserialize)]
pub enum DiffDir {
    Forward(Vec<DimDir>),
    Backward(Vec<DimDir>),
}
use DiffDir::*;

#[derive(Clone,Eq,PartialOrd,Ord,Serialize,Deserialize)]
pub struct Token {
    coord: [i32;4],
    divided: [u32;3],
    coef: i32,
    dim: usize,
    var_name: String,
    boundary: String,
}

impl Token {
    pub fn to_string(&self) -> String {
        format!("{}*{}({},{})",self.coef,self.boundary,(1..=self.dim).fold(self.coord[0].to_string(), |a,i| format!("{},{}",a,self.coord[i])),self.var_name)
    }
}

impl PartialEq for Token {
    fn eq(&self, other: &Self) -> bool {
        self.coord == other.coord && self.divided == other.divided
    }
}

#[derive(Clone,PartialEq,Eq,PartialOrd,Ord,Serialize,Deserialize)]
pub enum IndexingTypes {
    Vector(Vec<Token>),// for a real vector use one more coord in each token and set it manually to 0,1,2
    Scalar(Token),
}
use IndexingTypes::*;

impl IndexingTypes {
    pub fn new_scalar<'a>(dim: usize, var_name: &'a str, boundary: &'a str) -> IndexingTypes {
        Scalar(Token { coord: [0;4], divided: [0;3], coef: 1, dim, var_name: var_name.into(), boundary: boundary.into() })
    }
    pub fn new_vector<'a>(dim: usize, var_name: &'a str, boundary: &'a str) -> IndexingTypes {
        Vector((0..dim).map(|i| {
            let mut coord = [0;4];
            coord[dim] = i as i32;
            Token { coord, divided: [0;3], coef: 1, dim, var_name: var_name.into(), boundary: boundary.into() }
        }).collect())
    }
    pub fn apply_diffs(self, dirs: &[DiffDir]) -> Vec<IndexingTypes> {
        let mut tmp = vec![self];
        for dir in dirs {
            let (coordinc,dirs) = match dir {
                Forward(dirs) => (1,dirs),
                Backward(dirs) => (0,dirs)
            };
            tmp = IndexingTypes::factorize(tmp.into_iter().flat_map(|v|  match v {
                Vector(ts) => {
                    if ts.len() != dirs.len() { panic!("Could not apply diff on Vector as the dimension of the Vector and the diff array are different.") }
                    ts.into_iter().enumerate().flat_map(|(i,mut tl)| {
                        let d = dirs[i];
                        tl.divided[d as usize] += 1;
                        let mut tr = tl.clone();
                        tl.coord[d as usize] += coordinc;
                        tr.coord[d as usize] += coordinc-1;
                        tr.coef *= -1;
                        vec![Scalar(tl),Scalar(tr)]
                    }).collect()
                },
                Scalar(t) => {
                    let vl = dirs.iter().map(|i| {
                        let mut t = t.clone();
                        t.coord[*i as usize] += coordinc;
                        t.divided[*i as usize] += 1;
                        t
                    }).collect::<Vec<_>>();
                    let mut vr = vl.clone();
                    vr.iter_mut().for_each(|t| { t.coord.iter_mut().for_each(|i| *i -= 1); t.coef *= -1; });
                    vec![Vector(vl),Vector(vr)]
                }
            }).collect());
        }
        tmp
    }
    pub fn add_coef(self, other: Self) -> Self {
        match self {
            Scalar(mut t) => if let Scalar(to) = other {
                t.coef += to.coef;
                Scalar(t)
            } else {
                panic!("Could not add coef of different variant of IndexingTypes enum")
            },
            Vector(mut t) => if let Vector(to) = other {
                if t.len() != to.len() { panic!("Could not add coef of different size IndexingTypes::Vector.") }
                t.iter_mut().enumerate().for_each(|(i,v)| v.coef += to[i].coef);
                Vector(t)
            } else {
                panic!("Could not add coef of different variant of IndexingTypes enum.")
            }
        }
    }
    pub fn factorize(mut vals: Vec<IndexingTypes>) -> Vec<IndexingTypes> {
        if vals.len() == 0 { return vals }
        vals.sort();
        let mut fact = vals.into_iter();
        let first = fact.next().unwrap();
        let mut fact = fact.fold((vec![],first), |mut a,i| if a.1 == i { (a.0,a.1.add_coef(i)) } else { a.0.push(a.1); a.1 = i; a });
        fact.0.push(fact.1);

        fact.0
    }
}

#[derive(Clone)]
pub enum PDETokens<'a> {
    Add(&'a PDETokens<'a>,&'a PDETokens<'a>),
    Sub(&'a PDETokens<'a>,&'a PDETokens<'a>),
    Mul(&'a PDETokens<'a>,&'a PDETokens<'a>),
    Div(&'a PDETokens<'a>,&'a PDETokens<'a>),
    Diff(&'a PDETokens<'a>),
    Symbol(&'a str),
    Constant(Types),
    Indexable(IndexingTypes),
}

#[derive(Serialize,Deserialize)]
pub enum SPDETokens {
    Add(Box<SPDETokens>,Box<SPDETokens>),
    Sub(Box<SPDETokens>,Box<SPDETokens>),
    Mul(Box<SPDETokens>,Box<SPDETokens>),
    Div(Box<SPDETokens>,Box<SPDETokens>),
    Diff(Box<SPDETokens>),
    Symbol(String),
    Constant(Types),
    Indexable(IndexingTypes),
}

impl<'a> From<&PDETokens<'a>> for SPDETokens {
    fn from(pde: &PDETokens<'a>) -> Self {
        match pde {
            PDETokens::Add(a,b) => Self::Add(Box::new((*a).into()),Box::new((*b).into())),
            PDETokens::Sub(a,b) => Self::Sub(Box::new((*a).into()),Box::new((*b).into())),
            PDETokens::Mul(a,b) => Self::Mul(Box::new((*a).into()),Box::new((*b).into())),
            PDETokens::Div(a,b) => Self::Div(Box::new((*a).into()),Box::new((*b).into())),
            PDETokens::Diff(a) => Self::Diff(Box::new((*a).into())),
            PDETokens::Symbol(a) => Self::Symbol((*a).into()),
            PDETokens::Constant(a) => Self::Constant(*a),
            PDETokens::Indexable(a) => Self::Indexable(a.clone()),
        }
    }
}

pub struct PDECreator {
    expr: String,
}

impl PDECreator {
    pub fn add<'a>(&mut self, var_name: &'a str, boundary: &'a str, var: IndexingTypes, diffs: Vec<DiffDir>) {
        let var = var.apply_diffs(&diffs);//TODO use vars
    }
}

// Each PDE must be first order in time. A higher order PDE can be cut in multiple first order PDE.
// Example: d2u/dt2 + du/dt = u   =>   du/dt = z, dz/dt = u.
// It is why the parameter pdes is a Vec.
pub fn create_euler_pde<'a>(name: &'a str, dt: f64, pdes: Vec<SPDE>) -> SAlgorithm {
    let name = name.to_string();
    let needed = pdes.iter().map(|d| {
        NewKernel((&Kernel {
            name: &format!("{}_{}", &name, &d.dependant_var),
            args: vec![KCBuffer("dst",CF64),KCBuffer(&d.dependant_var,CF64)],
            src: &format!("uint id = x+x_size*(y+y_size*z); dst[id] = {}[id] +{}*({})", d.dependant_var, dt, d.expr),
            needed: vec![FuncName("mid")],
        }).into())
    }).collect::<Vec<_>>();
    let vars = pdes.iter().map(|d| (format!("{}_{}", &name, &d.dependant_var),d.dependant_var.clone())).collect::<Vec<_>>();
    SAlgorithm {
        name: name.clone(),
        callback: std::rc::Rc::new(move |h: &mut Handler, dim: Dim, _dimdir: &[DimDir], bufs: &[&str], _other: Option<&dyn Any>| {
            // bufs[0] = dst
            // bufs[1,2,...] = differential equation buffer holders in the same order as giver for
            // create_euler function
            // bufs[i] must write in bufs[i-1]
            let num = vars.len()+1;
            if bufs.len() != num { panic!("Euler algorithm \"{}\" must be given {} buffer arguments.", &name, &num); }
            let mut i = num;
            for (name,var) in vars.iter().rev() {
                i -= 1;
                h.run_arg(name,dim,&[BufArg(bufs[i-1],"dst"),BufArg(bufs[i],var)])?;
            }

            Ok(())
        }),
        needed
    }
}
