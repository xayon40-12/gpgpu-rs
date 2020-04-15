use crate::descriptors::FunctionConstructor::{self,*};
use crate::descriptors::EmptyType::{*,self};
use std::collections::HashMap;

#[derive(Clone)]
pub struct Function<'a> {
    pub name: String,
    pub args: Vec<FunctionConstructor<'a>>,
    pub ret_type: Option<EmptyType>,
    pub src: String,
    pub needed: Vec<Needed<'a>>,
}

#[derive(Clone)]
pub enum Needed<'a> {
    FuncName(String),
    CreateFunc(Function<'a>)
}


pub fn functions<'a>() -> HashMap<String,Function<'a>> {
    vec![
        Function {
            name: "swap".into(),
            args: vec![GlobalPtr("a",F64),GlobalPtr("b",F64)],
            ret_type: None,
            src: "double tmp = *a; *a = *b; *b = tmp;".into(),
            needed: vec![],
        },
        Function {
            name: "c_sqrmod".into(),
            args: vec![Param("src",F64_2)],
            ret_type: Some(F64),
            src: "return src.x*src.x + src.y*src.y;".into(),
            needed: vec![],
        },
        Function {
            name: "c_mod".into(),
            args: vec![Param("src",F64_2)],
            ret_type: Some(F64),
            src: "return sqrt(src.x*src.x + src.y*src.y);".into(),
            needed: vec![],
        },
        Function {
            name: "c_conj".into(),
            args: vec![Param("a",F64_2)],
            ret_type: Some(F64_2),
            src: "return (double2)(a.x, -a.y);".into(),
            needed: vec![],
        },
        Function {
            name: "c_times".into(),
            args: vec![Param("a",F64_2),Param("b",F64_2)],
            ret_type: Some(F64_2),
            src: "return (double2)(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);".into(),
            needed: vec![],
        },
        Function {
            name: "c_divides".into(),
            args: vec![Param("a",F64_2),Param("b",F64_2)],
            ret_type: Some(F64_2),
            src: "return c_times(a,c_conj(b))/c_sqrmod(b);".into(),
            needed: vec![],
        },
        Function {
            name: "c_exp".into(),
            args: vec![Param("x",F64)],
            ret_type: Some(F64_2),
            src: "
            double c, s = sincos(x,&c);
            return (double2)(c,s);".into(),
            needed: vec![],
        },
    ].into_iter().map(|f| (f.name.clone(),f)).collect()
}
