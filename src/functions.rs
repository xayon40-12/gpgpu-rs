use crate::descriptors::{FunctionConstructor::{self,*},SFunctionConstructor};
use crate::descriptors::empty_types::*;
use std::collections::HashMap;
use serde::{Serialize,Deserialize};

#[derive(Clone)]
pub struct Function<'a> {
    pub name: &'a str,
    pub args: Vec<FunctionConstructor<'a>>,
    pub ret_type: Option<&'a dyn EmptyType>,
    pub src: &'a str,
    pub needed: Vec<Needed<'a>>,
}

#[derive(Serialize,Deserialize)]
pub struct SFunction {
    pub name: String,
    pub args: Vec<SFunctionConstructor>,
    pub ret_type: Option<Box<dyn EmptyType>>,
    pub src: String,
    pub needed: Vec<SNeeded>,
}

impl<'a> From<&Function<'a>> for SFunction {
    fn from(f: &Function<'a>) -> Self {
        SFunction {
            name: f.name.into(),
            args: f.args.iter().map(|i| i.into()).collect(),
            ret_type: f.ret_type.and_then(|i| Some(i.into())),
            src: f.src.into(),
            needed: f.needed.iter().map(|i| i.into()).collect(),
        }
    }
}

#[derive(Clone)]
pub enum Needed<'a> {
    FuncName(&'a str),
    CreateFunc(Function<'a>)
}

#[derive(Serialize,Deserialize)]
pub enum SNeeded {
    FuncName(String),
    CreateFunc(SFunction)
}

impl<'a> From<&Needed<'a>> for SNeeded {
    fn from(n: &Needed<'a>) -> Self {
        match n {
            Needed::FuncName(n) => SNeeded::FuncName((*n).into()),
            Needed::CreateFunc(f) => SNeeded::CreateFunc(f.into()),
        }
    }
}


pub fn functions() -> HashMap<&'static str,Function<'static>> {
    vec![
        Function {
            name: "swap",
            args: vec![GlobalPtr("a",&F64),GlobalPtr("b",&F64)],
            ret_type: None,
            src: "double tmp = *a; *a = *b; *b = tmp;",
            needed: vec![],
        },
        Function {
            name: "c_sqrmod",
            args: vec![Param("src",&F64_2)],
            ret_type: Some(&F64),
            src: "return src.x*src.x + src.y*src.y;",
            needed: vec![],
        },
        Function {
            name: "c_mod",
            args: vec![Param("src",&F64_2)],
            ret_type: Some(&F64),
            src: "return sqrt(src.x*src.x + src.y*src.y);",
            needed: vec![],
        },
        Function {
            name: "c_conj",
            args: vec![Param("a",&F64_2)],
            ret_type: Some(&F64_2),
            src: "return (double2)(a.x, -a.y);",
            needed: vec![],
        },
        Function {
            name: "c_times",
            args: vec![Param("a",&F64_2),Param("b",&F64_2)],
            ret_type: Some(&F64_2),
            src: "return (double2)(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);",
            needed: vec![],
        },
        Function {
            name: "c_divides",
            args: vec![Param("a",&F64_2),Param("b",&F64_2)],
            ret_type: Some(&F64_2),
            src: "return c_times(a,c_conj(b))/c_sqrmod(b);",
            needed: vec![],
        },
        Function {
            name: "c_exp",
            args: vec![Param("x",&F64)],
            ret_type: Some(&F64_2),
            src: "
            double c, s = sincos(x,&c);
            return (double2)(c,s);",
            needed: vec![],
        },
    ].into_iter().map(|f| (f.name,f)).collect()
}
