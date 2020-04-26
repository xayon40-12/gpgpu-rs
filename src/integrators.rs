use crate::descriptors::{KernelConstructor::*,ConstructorTypes::*,KernelArg::*};
use crate::kernels::{Kernel};
use crate::algorithms::{SAlgorithm,SNeeded::*};
use crate::Handler;
use crate::dim::{Dim,DimDir};
use std::any::Any;
use serde::{Serialize,Deserialize};
use crate::descriptors::{Types,ConstructorTypes};

#[derive(Debug,Clone,Serialize,Deserialize)]
pub enum DiffDir {
    Forward(Vec<DimDir>),
    Backward(Vec<DimDir>),
}
use DiffDir::*;

#[derive(Debug,Clone,Eq,PartialOrd,Ord,Serialize,Deserialize)]
pub struct Token {
    coord: [i32;4],
    divided: [u32;3],
    coef: i32,
    dim: usize,
    var_name: String,
    boundary: String,
}

fn divided_str(d: &[u32;3]) -> String {
    let mut tmp = vec![];
    let mut push = |n: &'static str,d| if d > 0 { if d > 1 { tmp.push(format!("pow({},{})",n,d)) } else { tmp.push(n.to_string()) } };
    push("dx",d[0]);
    push("dy",d[1]);
    push("dz",d[2]);
    if tmp.len() == 0 { return "".into() };
    let mut tmp = tmp.into_iter();
    let first = tmp.next().unwrap();
    format!("/({})",tmp.fold(first,|a,i| format!("{}*{}",a,i)))
}
fn coords_str(c: &[i32;4], dim: usize) -> String {
    let coo = |i| if i == dim { format!("{}",c[3]) } else { format!("{}{}{}",
        ['x','y','z'][i],
        if c[i]<=0 { "" } else { "+" },
        if c[i]==0 { "".to_string() } else { format!("{}",c[i]) 
    })};
    (1..=dim).fold(coo(0), |a,i| format!("{},{}",a,coo(i)))
}
impl Token {
    pub fn apply_idx(&mut self, idx: &[i32;4]) {
        for i in 0..4 {
            self.coord[i] += idx[i];
        }
    }
    pub fn to_string(&self) -> String {
        format!("({})*{}({},{}){}",self.coef,self.boundary,coords_str(&self.coord,self.dim),self.var_name,divided_str(&self.divided))
    }
}

impl PartialEq for Token {
    fn eq(&self, other: &Self) -> bool {
        self.coord == other.coord && self.divided == other.divided
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
        Scalar(Token { coord: [0;4], divided: [0;3], coef: 1, dim, var_name: var_name.into(), boundary: boundary.into() })
    }
    pub fn new_vector<'a>(dim: usize, var_name: &'a str, boundary: &'a str) -> IndexingTypes {
        Vector((0..dim).map(|i| {
            let mut coord = [0;4];
            coord[dim] = i as i32;
            Token { coord, divided: [0;3], coef: 1, dim, var_name: var_name.into(), boundary: boundary.into() }
        }).collect())
    }
    pub fn apply_diffs(&self, dirs: &[DiffDir]) -> Vec<IndexingTypes> {
        let mut tmp = vec![self.clone()];
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
                    vr.iter_mut().for_each(|t| { dirs.iter().for_each(|i| t.coord[*i as usize] -= 1); t.coef *= -1; });
                    vec![Vector(vl),Vector(vr)]
                }
            }).collect());
        }
        tmp
    }
    pub fn apply_idx(mut self, idx: &[i32;4]) -> Self {
        match &mut self {
            Vector(v) => v.iter_mut().for_each(|i| i.apply_idx(idx)),
            Scalar(s) => s.apply_idx(idx),
        }
        self
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
    Vect(Vec<f64>),
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
    Vect(Vec<f64>),
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
            PDETokens::Vect(a) => Self::Vect((*a).clone()),
            PDETokens::Indx(a) => Self::Indx(a.clone()),
        }
    }
}

impl<'a> PDETokens<'a> {
    pub fn to_ocl(&self) -> String {
        SPDETokens::from(self).to_ocl()
    }
}

impl SPDETokens {
    pub fn to_ocl(self) -> String {
        use SPDETokens::*;
        match self.convert() {
            Add(a,b) => format!("{} + {}", a.to_ocl(), b.to_ocl()),
            Sub(a,b) => format!("{} - {}", a.to_ocl(), b.to_ocl()),
            Mul(a,b) => format!("{} * {}", a.to_ocl(), b.to_ocl()),
            Div(a,b) => format!("{} / {}", a.to_ocl(), b.to_ocl()),
            Func(n,a) => format!("{}({})",n,a.to_ocl()),
            Symb(a) => a,
            Const(a) => format!("{:e}",a),
            Indx(a) => if let Scalar(s) = a { s.to_string() } else { panic!("Cannot convert IndexingTypes::Vector to ocl String, it must be handled by SPDEToken::convert().") },
            s @ _ => panic!("Not expected during SPDEToken::to_ocl: {:?}", s),
        }
    }

    fn convert(self) -> Self {
        use SPDETokens::*;
        match self {
            Mul(a,b) => {
                match *a {
                    Add(c,d) => Add(Box::new(Mul(c,b.clone())),Box::new(Mul(d,b))),
                    Sub(c,d) => Sub(Box::new(Mul(c,b.clone())),Box::new(Mul(d,b))),
                    Diff(c,d) => Mul(Box::new(c.apply_diff(vec![d])),b).convert(),
                    _ => match *b {
                        Add(c,d) => Add(Box::new(Mul(a.clone(),c)),Box::new(Mul(a,d))),
                        Diff(c,d) => Mul(a,Box::new(c.apply_diff(vec![d]))).convert(),
                        _ => match *a {
                            Vect(a) => match *b {
                                Vect(b) => if a.len() != b.len() { 
                                    panic!("Vect must have the same len in SPDEToken::Mul") 
                                } else { 
                                    let mut tmp = a.into_iter().enumerate().map(|(i,v)| Mul(Box::new(Const(v)),Box::new(Const(b[i]))));
                                    let first = tmp.next().unwrap();
                                    tmp.fold(first,|a,i| Add(Box::new(a),Box::new(i)))
                                },
                                Indx(b) => if let Vector(b) = b {
                                    if a.len() != b.len() { 
                                        panic!("Vect and Indx(Vector(..)) must have the same len in SPDEToken::Mul") 
                                    } else { 
                                        let mut tmp = a.into_iter().enumerate().map(|(i,v)| Mul(Box::new(Const(v)),Box::new(Indx(Scalar(b[i].clone())))));
                                        let first = tmp.next().unwrap();
                                        tmp.fold(first,|a,i| Add(Box::new(a),Box::new(i)))
                                    }
                                } else {
                                    panic!("Vect must be multiplied with Vect or Indx(Vector(..)). given {:?}.", b)
                                },
                                _ => panic!("Vect must be multiplied with Vect or Indx(Vector(..)). given {:?}.", b)
                            },
                            Indx(a) => match a{
                                Vector(a) =>
                                    match *b {
                                        Vect(b) => if a.len() != b.len() { 
                                            panic!("Vect must have the same len in SPDEToken::Mul") 
                                        } else { 
                                            let mut tmp = a.into_iter().enumerate().map(|(i,v)| Mul(Box::new(Indx(Scalar(v))),Box::new(Const(b[i]))));
                                            let first = tmp.next().unwrap();
                                            tmp.fold(first,|a,i| Add(Box::new(a),Box::new(i)))
                                        },
                                        Indx(b) => if let Vector(b) = b {
                                            if a.len() != b.len() { 
                                                panic!("Vect and Indx(Vector(..)) must have the same len in SPDEToken::Mul") 
                                            } else { 
                                                let mut tmp = a.into_iter().enumerate().map(|(i,v)| Mul(Box::new(Indx(Scalar(v))),Box::new(Indx(Scalar(b[i].clone())))));
                                                let first = tmp.next().unwrap();
                                                tmp.fold(first,|a,i| Add(Box::new(a),Box::new(i)))
                                            }
                                        } else {
                                            panic!("Vect must be multiplied with Vect or Indx(Vector(..)). given {:?}.", b)
                                        },
                                        _ => panic!("Vect must be multiplied with Vect or Indx(Vector(..)). given {:?}.", b)
                                    },
                                Scalar(a) => {
                                    Mul(Box::new(Indx(Scalar(a))),b)
                                }
                            },
                            _ => Mul(a,b),    
                        },
                    },
                }
            },
            Diff(a,d) => a.apply_diff(vec![d]),
            Vect(_) => panic!("Cannot convert SPDETokens::Vector, it should have been multiplied by an other vector."),
            Indx(a) => if let Scalar(_) = &a { Indx(a) } else { panic!("Cannot convert IndexingTypes::Vector, it should have been multiplied by an other vector.") },
            _ => self,
        }
    }

    fn apply_idx(self, idx: &[i32;4]) -> Box<Self> {
        use SPDETokens::*;
        match self {
            Add(a,b) => Add(a.apply_idx(idx),b.apply_idx(idx)),
            Sub(a,b) => Sub(a.apply_idx(idx),b.apply_idx(idx)),
            Mul(a,b) => Mul(a.apply_idx(idx),b.apply_idx(idx)),
            Div(a,b) => Div(a.apply_idx(idx),b.apply_idx(idx)),
            Diff(a,d) => Diff(a,d),//TODO remove and put recursive diff: a.apply_diff(d).apply_idx(idx)
            Func(n,a) => Func(n,a.apply_idx(idx)),
            Indx(a) => Indx(a.apply_idx(idx)),
            _ => self
        }.into()
    }

    fn apply_diff(self, mut diffs: Vec<DiffDir>) -> Self {
        use SPDETokens::*;
        match self {
            Diff(a,d) => { diffs.push(d); a.apply_diff(diffs) },
            Indx(a) => {
                let mut it = a.apply_diffs(&diffs[..]).into_iter().map(|i| Indx(i));
                let first = it.next().unwrap();
                it.fold(first, |a,i| Add(Box::new(a),Box::new(i)))
            },//TODO add the possibility to Diff over Sum,Mul,...
            _ => panic!("Diff can only be applied to Diff or Indexable."),
        }
        //TODO remove up, then change diffs to d: DiffDir, then match d and create
        //Div(Sub(left.apply_idx(1/0),right.apply_idx(0/-1)),Symb(dx))
    }
}

#[derive(Clone)]
pub struct PDE<'a> {
    pub dependant_var: &'a str,
    pub expr: PDETokens<'a>,
}

#[derive(Clone,Serialize,Deserialize)]
pub struct SPDE {
    pub dependant_var: String,
    pub expr: String,
}

impl<'a> From<&PDE<'a>> for SPDE {
    fn from(de: &PDE) -> SPDE {
        SPDE {
            dependant_var: de.dependant_var.into(),
            expr: de.expr.to_ocl(),
        }
    }
}

// Each PDE must be first order in time. A higher order PDE can be cut in multiple first order PDE.
// Example: d2u/dt2 + du/dt = u   =>   du/dt = z, dz/dt = u.
// It is why the parameter pdes is a Vec.
pub fn create_euler_pde<'a>(name: &'a str, dt: f64, pdes: Vec<SPDE>, params: Vec<(String,ConstructorTypes)>) -> SAlgorithm {
    let name = name.to_string();
        let mut args = vec![KCBuffer("dst",CF64)];
        args.extend(pdes.iter().map(|pde| KCBuffer(&pde.dependant_var,CF64)));
        args.extend(params.iter().map(|t| KCParam(&t.0,t.1)));
    let needed = pdes.iter().map(|d| {
        NewKernel((&Kernel {
            name: &format!("{}_{}", &name, &d.dependant_var),
            args: args.clone(),
            src: &format!("    uint _i = x+x_size*(y+y_size*z);\n    dst[_i] = {}[_i] +{}*({});", d.dependant_var, dt, d.expr),
            needed: vec![],
        }).into())
    }).collect::<Vec<_>>();
    let vars = pdes.iter().map(|d| (format!("{}_{}", &name, &d.dependant_var),d.dependant_var.clone())).collect::<Vec<_>>();
    SAlgorithm {
        name: name.clone(),
        callback: std::rc::Rc::new(move |h: &mut Handler, dim: Dim, _dimdir: &[DimDir], bufs: &[&str], other: Option<&dyn Any>| {
            // bufs[0] = dst
            // bufs[1,2,...] = differential equation buffer holders in the same order as giver for
            // create_euler function
            // bufs[i] must write in bufs[i-1]
            let num = vars.len()+1;
            if bufs.len() != num { panic!("Euler algorithm \"{}\" must be given {} buffer arguments.", &name, &num); }
            let mut args = vec![BufArg(&bufs[0],"dst")];
            for i in 0..vars.len() {
                args.push(BufArg(&bufs[i+1],&vars[i].1));
            }
            if let Some(params) = other {
                args.extend(params.downcast_ref::<Vec<(String,Types)>>().expect(&format!("Parameters of \"{}\" Euler Algorithm must be Vec<(String,Types)>.",&name)).iter().map(|i| Param(&i.0,i.1)));
            }
            for i in (0..vars.len()).rev() {
                h.run_arg(&vars[i].0,dim,&args)?;
                h.copy(bufs[0],bufs[i+1])?;
            }

            Ok(())
        }),
        needed
    }
}