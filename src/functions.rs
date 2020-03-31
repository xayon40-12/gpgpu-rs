use crate::descriptors::FunctionConstructor::{self,*};
use crate::descriptors::EmptyType::{*,self};
use std::collections::HashMap;

#[derive(Clone)]
pub struct Function<'a,S: Into<String>+Clone,T: Into<String>+Clone> {
    pub name: S,
    pub args: Vec<FunctionConstructor<'a>>,
    pub ret_type: Option<EmptyType>,
    pub src: T
}

impl<'a,S: Into<String>+Clone,T: Into<String>+Clone> Function<'a,S,T> {
    pub fn convert(&self) -> Function<'a,String,String> {
        Function {
            name: self.name.clone().into(),
            args: self.args.clone(),
            ret_type: self.ret_type.clone(),
            src: self.src.clone().into()
        }
    }
}


pub fn functions<'a>() -> HashMap<&'static str,Function<'a,&'static str,&'static str>> {
    vec![
        Function {
            name: "swap",
            args: vec![GlobalPtr("a",F64),GlobalPtr("b",F64)],
            ret_type: None,
            src: "double tmp = *a; *a = *b; *b = tmp;"
        }
    ].into_iter().map(|f| (f.name,f)).collect()
}
