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
        }
    ].into_iter().map(|f| (f.name.clone(),f)).collect()
}
