use crate::descriptors::{FunctionConstructor::{self,*},SFunctionConstructor};
use crate::descriptors::ConstructorTypes::{self,*};
use std::collections::HashMap;
use serde::{Serialize,Deserialize};

#[derive(Clone,Debug)]
pub struct Function<'a> {
    pub name: &'a str,
    pub args: Vec<FunctionConstructor<'a>>,
    pub ret_type: Option<ConstructorTypes>,
    pub src: &'a str,
    pub needed: Vec<Needed<'a>>,
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct SFunction {
    pub name: String,
    pub args: Vec<SFunctionConstructor>,
    pub ret_type: Option<ConstructorTypes>,
    pub src: String,
    pub needed: Vec<SNeeded>,
}

impl<'a> From<&Function<'a>> for SFunction {
    fn from(f: &Function<'a>) -> Self {
        SFunction {
            name: f.name.into(),
            args: f.args.iter().map(|i| i.into()).collect(),
            ret_type: f.ret_type,
            src: f.src.into(),
            needed: f.needed.iter().map(|i| i.into()).collect(),
        }
    }
}

#[derive(Clone,Debug)]
pub enum Needed<'a> {
    FuncName(&'a str),
    CreateFunc(Function<'a>)
}

#[derive(Clone,Debug,Serialize,Deserialize)]
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
            args: vec![FCGlobalPtr("a",CF64),FCGlobalPtr("b",CF64)],
            ret_type: None,
            src: "double tmp = *a; *a = *b; *b = tmp;",
            needed: vec![],
        },
        Function {
            name: "mid",
            args: vec![FCParam("x",CU32),FCParam("y",CU32),FCParam("z",CU32)],
            ret_type: Some(CU32),
            src: "return (x%get_global_work_size(0))+get_global_work_size(0)*((y%get_global_work_size(1))+get_global_work_size(1)*(z%get_global_work_size(2)));",
            needed: vec![],
        },
        Function {
            name: "c_sqrmod",
            args: vec![FCParam("src",CF64_2)],
            ret_type: Some(CF64),
            src: "return src.x*src.x + src.y*src.y;",
            needed: vec![],
        },
        Function {
            name: "c_mod",
            args: vec![FCParam("src",CF64_2)],
            ret_type: Some(CF64),
            src: "return sqrt(src.x*src.x + src.y*src.y);",
            needed: vec![],
        },
        Function {
            name: "c_conj",
            args: vec![FCParam("a",CF64_2)],
            ret_type: Some(CF64_2),
            src: "return (double2)(a.x, -a.y);",
            needed: vec![],
        },
        Function {
            name: "c_times",
            args: vec![FCParam("a",CF64_2),FCParam("b",CF64_2)],
            ret_type: Some(CF64_2),
            src: "return (double2)(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);",
            needed: vec![],
        },
        Function {
            name: "c_times_conj",
            args: vec![FCParam("a",CF64_2),FCParam("b",CF64_2)],
            ret_type: Some(CF64_2),
            src: "return (double2)(a.x*b.x+a.y*b.y, -a.x*b.y+a.y*b.x);",
            needed: vec![],
        },
        Function {
            name: "c_divides",
            args: vec![FCParam("a",CF64_2),FCParam("b",CF64_2)],
            ret_type: Some(CF64_2),
            src: "return c_times(a,c_conj(b))/c_sqrmod(b);",
            needed: vec![],
        },
        Function {
            name: "c_exp",
            args: vec![FCParam("x",CF64)],
            ret_type: Some(CF64_2),
            src: "
            double c, s = sincos(x,&c);
            return (double2)(c,s);",
            needed: vec![],
        },
    ].into_iter().map(|f| (f.name,f)).collect()
}
