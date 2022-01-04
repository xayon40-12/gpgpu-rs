use crate::pde_parser::pde_ir::ir_helper::Indexable;
use crate::pde_parser::pde_lexer::range;
use crate::pde_parser::pde_lexer::GLOBAL_DIM;
use crate::pde_parser::pde_lexer::VARS;
use crate::pde_parser::SPDETokens::Symb;
use crate::pde_parser::DPDE;
use nom::branch::alt;
use nom::bytes::complete::take_while;
use nom::bytes::complete::take_while1;
use nom::sequence::pair;
use nom::IResult;
use std::ops::Range;

use crate::pde_parser::pde_ir::ir_helper::lexer_compositor::LexerComp;

use super::array;

pub fn extract_variable(name: String, r: &[Range<usize>]) -> LexerComp {
    let mut dpde = None;
    VARS.with(|vars| {
        vars.borrow().iter().for_each(|v| {
            if v.var_name == name {
                dpde = Some(v.clone());
            }
        })
    });
    let mut gd = None;
    GLOBAL_DIM.with(|d| {
        gd = Some(*d.borrow());
    });
    let gd = gd.expect("Thread error, could not retreive global dim.");
    if let Some(DPDE {
        var_name: v,
        boundary: b,
        var_dim: d,
        vec_dim: vd,
    }) = dpde
    {
        if !r.is_empty() {
            Indexable::new_slice(d, gd, vd, r, &v, &b)
        } else if vd > 1 {
            Indexable::new_vector(d, gd, vd, &v, &b)
        } else {
            Indexable::new_scalar(d, gd, &v, &b)
        }
    } else if !r.is_empty() {
        panic!("Symbol \"{}\" is not indexable.", name)
    } else {
        Symb(name)
    }
    .into()
}

pub fn aanum(s: &str) -> IResult<&str, String> {
    let is_alpha = |c| ('A'..='Z').contains(&c) || ('a'..='z').contains(&c);
    let is_num = |c| ('0'..='9').contains(&c);
    pair(
        take_while1(move |c| is_alpha(c) || c == '_'),
        take_while(move |c| is_alpha(c) || is_num(c) || c == '_'),
    )(s)
    .map(|(s, (a, b))| (s, format!("{}{}", a, b)))
}

pub fn var(s: &str) -> IResult<&str, LexerComp> {
    let alone = |s| aanum(s).map(|(s, v)| (s, extract_variable(v, &[])));
    let vect_idx = |s| pair(aanum, array(range))(s).map(|(s, (v, u))| (s, extract_variable(v, &u)));
    let x = alt((vect_idx, alone))(s);
    x
}
