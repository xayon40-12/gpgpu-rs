use super::stag;
use crate::pde_parser::pde_lexer::fun::fun;
use crate::pde_parser::pde_lexer::LexerComp;
use nom::branch::alt;
use nom::sequence::tuple;
use nom::IResult;

pub fn expr(s: &str) -> IResult<&str, LexerComp> {
    let add = |s| tuple((expr, stag("+"), factor))(s).map(|(s, (l, _, r))| (s, l + r));
    let sub = |s| tuple((expr, stag("-"), factor))(s).map(|(s, (l, _, r))| (s, l - r));
    alt((add, sub, factor))(s)
}

pub fn factor(s: &str) -> IResult<&str, LexerComp> {
    let mul = |s| tuple((factor, stag("*"), pow))(s).map(|(s, (l, _, r))| (s, l * r));
    let div = |s| tuple((factor, stag("/"), pow))(s).map(|(s, (l, _, r))| (s, l / r));
    alt((mul, div, pow))(s)
}

pub fn pow(s: &str) -> IResult<&str, LexerComp> {
    let po =
        |s| tuple((fun, alt((stag("^"), stag("**"))), pow))(s).map(|(s, (l, _, r))| (s, l ^ r));
    alt((po, fun))(s)
}
