use crate::descriptors::FunctionConstructor::{self,*};
use crate::descriptors::EmptyType::{*,self};
use std::collections::HashMap;

#[derive(Clone)]
pub struct Function<'a> {
    pub name: &'a str,
    pub args: Vec<FunctionConstructor<'a>>,
    pub ret_type: Option<EmptyType>,
    pub src: &'a str
}


pub fn functions<'a>() -> HashMap<&'static str,Function<'a>> {
    vec![
        Function {
            name: "swap",
            args: vec![GlobalPtr("a",F64),GlobalPtr("b",F64)],
            ret_type: None,
            src: "double tmp = *a; *a = *b; *b = tmp;"
        }
    ].into_iter().map(|f| (f.name,f)).collect()
}
