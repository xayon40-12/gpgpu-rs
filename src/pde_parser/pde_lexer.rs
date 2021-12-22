use crate::pde_parser::pde_ir::{ir_helper::*, lexer_compositor::*};
use crate::pde_parser::pde_lexer::math::expr;
use crate::pde_parser::pde_lexer::math::factor;
use crate::pde_parser::pde_lexer::var::var;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::one_of;
use nom::character::complete::space0;
use nom::character::complete::u32;
use nom::combinator::{eof, opt};
use nom::error::Error;
use nom::multi::separated_list1;
use nom::number::complete::double;
use nom::sequence::delimited;
use nom::sequence::pair;
use nom::sequence::preceded;
use nom::sequence::terminated;
use nom::Finish;
use nom::IResult;
use std::cell::RefCell;
use std::ops::Range;

use super::pde_ir::lexer_compositor::LexerComp;
use super::{SPDETokens, DPDE};

mod fun;
mod math;
mod var;

thread_local!(
    static VARS: RefCell<Vec<DPDE>> = RefCell::new(vec![]);
    static CURRENT_VAR: RefCell<Option<SPDETokens>> = RefCell::new(None);
    static GLOBAL_DIM: RefCell<usize> = RefCell::new(0);
    static FUN_LEN: RefCell<usize> = RefCell::new(0);
);

fn next_id() -> usize {
    let mut fun_len = None;
    FUN_LEN.with(|fl| {
        fun_len = Some(*fl.borrow());
        *fl.borrow_mut() += 1;
    });
    fun_len.expect("Could not retreive FUN_LEN in fix_constructor.")
}

pub fn parse<'a>(
    context: &[DPDE],                 // var name
    current_var: &Option<SPDETokens>, // boundary funciton name
    fun_len: usize,                   // number of existing functions
    global_dim: usize,                // dim
    math: &'a str,
) -> Result<(&'a str, LexerComp), Error<&'a str>> {
    VARS.with(|v| *v.borrow_mut() = context.to_vec());
    CURRENT_VAR.with(|v| *v.borrow_mut() = current_var.clone());
    GLOBAL_DIM.with(|v| *v.borrow_mut() = global_dim);
    FUN_LEN.with(|v| *v.borrow_mut() = fun_len);
    terminated(delimited(space0, expr, space0), eof)(math).finish()
}

pub fn stag(t: &'static str) -> impl Fn(&str) -> IResult<&str, &str> {
    move |s| delimited(space0, tag(t), space0)(s)
}

pub fn array<T: Clone>(
    f: impl Fn(&str) -> IResult<&str, T> + Copy,
) -> impl Fn(&str) -> IResult<&str, Vec<T>> {
    move |s| {
        delimited(
            stag("["),
            separated_list1(delimited(space0, one_of(",;"), space0), f),
            stag("]"),
        )(s)
        .map(|(s, vs)| (s, vs.to_vec()))
    }
}

pub fn term(s: &str) -> IResult<&str, LexerComp> {
    let parens = delimited(stag("("), expr, stag(")"));
    let arr = |s| array(expr)(s).map(|(s, v)| (s, compact(v).map(vect)));
    let unary_minus =
        |s| preceded(stag("-"), factor)(s).map(|(s, v)| (s, LexerComp::from(Const(-1.0)) * v));
    alt((parens, arr, num, unary_minus, var))(s)
}

pub fn range(s: &str) -> IResult<&str, Range<usize>> {
    pair(u32, opt(preceded(stag(".."), u32)))(s)
        .map(|(s, (a, b))| (s, a as usize..(b.unwrap_or(a) + 1) as usize))
}
pub fn num(s: &str) -> IResult<&str, LexerComp> {
    double(s).map(|(s, d)| (s, d.into()))
}
