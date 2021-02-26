use crate::functions::SFunction;
#[allow(unused_imports)]
use crate::integrators::{
    pde_ir::{ir_helper::DPDE, SPDETokens},
    SPDE,
};
#[allow(unused_imports)]
use crate::pde_lexer;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parsed {
    ocl: Vec<String>,
    funs: Vec<SFunction>,
}

///context should contains the current pde, so one of the 'name' in &[DPDE]
/// should be the same as 'name' in this parse function
pub fn parse(context: &[DPDE], fun_len: usize, math: &str) -> Parsed {
    let parsed = pde_lexer::ExprParser::new()
        .parse(context, fun_len, math)
        .expect("Parse error:");
    Parsed {
        ocl: parsed.token.to_ocl(),
        funs: parsed.funs,
    }
}

#[test]
fn pde_lexer() {
    let u = DPDE {
        var_name: "u".into(),
        boundary: "b".into(),
        dim: 3,
        vector: false,
    };
    let v = DPDE {
        var_name: "v".into(),
        boundary: "b".into(),
        dim: 3,
        vector: true,
    };
    println!(
        "{:?}",
        parse(&[u.clone()], 0, "22 + (1 - 4+-15.7^(1--0.3+D+u))*7")
    );
    println!("{:?}", parse(&[], 0, "2^3^4"));
    println!("{:?}", parse(&[u.clone()], 0, "#>xz u"));
    println!("{:?}", parse(&[u.clone()], 0, "#<xz u"));
    println!("{:?}", parse(&[u.clone()], 0, "#>xz^2 u"));
    println!("{:?}", parse(&[u.clone()], 0, "#<xz^2 u"));
    println!("{:?}", parse(&[u.clone()], 0, "#>xz [u,u]"));
    println!("{:?}", parse(&[v.clone()], 0, "#> v"));
    println!("{:?}", parse(&[u.clone()], 0, "#> u"));
    println!("{:?}", parse(&[], 0, "cos (sin 3) + atan2(3, 4)^1.5"));
    println!("{:?}", parse(&[], 0, "cos (sin 3) + atan2(3, 4)^1.5"));
    println!("{:?}", parse(&[], 1, "fix(f,fix(b,r))"))
}
