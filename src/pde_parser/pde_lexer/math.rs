use super::stag;
use crate::pde_parser::pde_lexer::fun::fun;
use crate::pde_parser::pde_lexer::LexerComp;
use nom::branch::alt;
use nom::character::complete::one_of;
use nom::character::complete::space0;
use nom::multi::fold_many0;
use nom::sequence::{delimited, pair, tuple};
use nom::IResult;

pub fn expr(s: &str) -> IResult<&str, LexerComp> {
    let (i, init) = factor(s)?;
    let addsub = pair(delimited(space0, one_of("+-"), space0), factor);
    fold_many0(
        addsub,
        move || init.clone(),
        |acc, (op, val)| {
            if op == '+' {
                acc + val
            } else {
                acc - val
            }
        },
    )(i)
}

pub fn factor(s: &str) -> IResult<&str, LexerComp> {
    let (i, init) = pow(s)?;
    let addsub = pair(delimited(space0, one_of("*/"), space0), pow);
    fold_many0(
        addsub,
        move || init.clone(),
        |acc, (op, val)| {
            if op == '*' {
                acc * val
            } else {
                acc / val
            }
        },
    )(i)
}

pub fn pow(s: &str) -> IResult<&str, LexerComp> {
    let po =
        |s| tuple((fun, alt((stag("^"), stag("**"))), pow))(s).map(|(s, (l, _, r))| (s, l ^ r));
    alt((po, fun))(s)
}
