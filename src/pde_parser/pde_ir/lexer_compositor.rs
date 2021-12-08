use crate::functions::SFunction;
use crate::pde_parser::pde_ir::SPDETokens;
use serde::{Deserialize, Serialize};
use std::ops::{Add, BitXor, Div, Mul, Sub};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexerComp {
    pub token: SPDETokens,
    pub funs: Vec<SFunction>,
}

impl LexerComp {
    pub fn map<T: FnOnce(SPDETokens) -> SPDETokens>(self, f: T) -> LexerComp {
        LexerComp {
            token: f(self.token),
            funs: self.funs,
        }
    }

    pub fn bind_id<T: FnOnce(SPDETokens, usize) -> LexerComp>(mut self, f: T) -> LexerComp {
        let mut res = f(self.token, self.funs.len());
        self.funs.append(&mut res.funs); // conserve order of function creation
        res.funs = self.funs;
        res
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Compacted {
    tokens: Vec<SPDETokens>,
    funs: Vec<SFunction>,
}

impl Compacted {
    pub fn empty() -> Compacted {
        Compacted {
            tokens: vec![],
            funs: vec![],
        }
    }

    pub fn map<T: FnOnce(Vec<SPDETokens>) -> SPDETokens>(self, f: T) -> LexerComp {
        LexerComp {
            token: f(self.tokens),
            funs: self.funs,
        }
    }

    pub fn bind_id<T: FnOnce(Vec<SPDETokens>, usize) -> LexerComp>(mut self, f: T) -> LexerComp {
        let mut res = f(self.tokens, self.funs.len());
        self.funs.append(&mut res.funs);
        res.funs = self.funs;
        res
    }
}

pub fn compact(tab: Vec<LexerComp>) -> Compacted {
    tab.into_iter().fold(Compacted::empty(), |mut acc, mut i| {
        acc.tokens.push(i.token);
        acc.funs.append(&mut i.funs);
        acc
    })
}

impl<T: Into<SPDETokens>> From<T> for LexerComp {
    fn from(pde: T) -> Self {
        LexerComp {
            token: pde.into(),
            funs: vec![],
        }
    }
}

macro_rules! op {
    ($name:ident|$fun:ident $op:tt) => {
impl $name for LexerComp {
    type Output = Self;
    fn $fun(self, r: Self) -> Self {
        let LexerComp {
            token: lt,
            funs: mut lf,
        } = self;
        let LexerComp {
            token: rt,
            funs: mut rf,
        } = r;
        lf.append(&mut rf);

        LexerComp {
            token: lt $op rt,
            funs: lf,
        }
    }
}
    };
}

op! {Add|add +}
op! {Sub|sub -}
op! {Mul|mul *}
op! {Div|div /}
op! {BitXor|bitxor ^}
