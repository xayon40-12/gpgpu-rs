use crate::functions::SFunction;
#[allow(unused_imports)]
pub use crate::integrators::{
    pde_ir::{ir_helper::DPDE, SPDETokens},
    SPDE,
};
#[allow(unused_imports)]
use crate::pde_lexer;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parsed {
    pub ocl: Vec<String>,
    pub funs: Vec<SFunction>,
}

///context should contains the current pde, so one of the 'name' in &[DPDE]
/// should be the same as 'name' in this parse function.
/// 'fun_len' correspond to how many functions were already generated by other call to parse().
/// 'math' is the mathematical expression to parse.
pub fn parse<'a>(
    context: &[DPDE],
    fun_len: usize,
    math: &'a str,
) -> Result<Parsed, lalrpop_util::ParseError<usize, lalrpop_util::lexer::Token<'a>, &'a str>> {
    let parsed = pde_lexer::ExprParser::new().parse(context, fun_len, math)?;
    Ok(Parsed {
        ocl: parsed.token.to_ocl(),
        funs: parsed.funs,
    })
}

#[test]
fn pde_lexer() {
    let parse = |c: &[DPDE], f: usize, m: &str| parse(c, f, m).unwrap();
    let u = DPDE {
        var_name: "u".into(),
        boundary: "b".into(),
        var_dim: 3,
        vec_dim: 1,
    };
    let pu = [u];
    let v = DPDE {
        var_name: "v".into(),
        boundary: "b".into(),
        var_dim: 3,
        vec_dim: 3,
    };
    let vu = [v];
    println!("{:?}", parse(&pu, 0, "22 + (1 - 4+-15.7^(1--0.3+D+u))*7"));
    println!("{:?}", parse(&[], 0, "2^3^4"));
    println!("{:?}", parse(&[], 0, "[1,cos(2),4]"));
    println!("{:?}", parse(&[], 0, "[1;cos(2);4]"));
    println!("{:?}", parse(&pu, 0, "#>xz u"));
    println!("{:?}", parse(&pu, 0, "#<xz u"));
    println!("{:?}", parse(&pu, 0, "#>xz^2 u"));
    println!("{:?}", parse(&pu, 0, "#<xz^2 u"));
    println!("{:?}", parse(&pu, 0, "#>xz [u,u]"));
    println!("{:?}", parse(&vu, 0, "#> v"));
    println!("{:?}", parse(&pu, 0, "#> u"));
    println!("{:?}", parse(&[], 0, "cos (sin (3)) + atan2(3, 4)^1.5"));
    println!("{:?}", parse(&[], 0, "cos (sin (3)) + atan2(3, 4)^1.5"));
    println!("{:?}", parse(&[], 1, "fix(f,fix(b,r))"))
}
