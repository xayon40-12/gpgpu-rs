use crate::functions::SFunction;
use nom::error::Error;
pub mod pde_ir;
pub mod pde_lexer;
pub use crate::integrators::SPDE;
pub use pde_ir::{ir_helper::DPDE, SPDETokens};
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
    current_var: &Option<SPDETokens>,
    fun_len: usize,
    global_dim: usize,
    math: &'a str,
) -> Result<Parsed, Error<&'a str>> {
    let (_, parsed) = pde_lexer::parse(context, current_var, fun_len, global_dim, math)?;
    Ok(Parsed {
        ocl: parsed.token.to_ocl(),
        funs: parsed.funs,
    })
}

#[test]
fn pde_lexer() {
    let parse = |c: &[DPDE], cur: &Option<SPDETokens>, f: usize, gd: usize, m: &str| {
        parse(c, cur, gd, f, m).unwrap()
    };
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
    let large = DPDE {
        var_name: "l".into(),
        boundary: "b".into(),
        var_dim: 2,
        vec_dim: 10,
    };
    let lu = [large];
    println!("{:?}", parse(&pu, &None, 0, 3, "- 2 - 4"));
    assert_eq!(parse(&[], &None, 0, 3, "- 2 - 4").ocl, &["-6e0"]);
    println!("{:?}", parse(&pu, &None, 0, 3, "22 + (1 - 4 + 15)"));
    println!("{:?}", parse(&pu, &None, 0, 3, "22 + x_size"));
    println!(
        "{:?}",
        parse(&pu, &None, 0, 3, "22 + (1 - 4+-15.7^(1--0.3+D+u))*7")
    );
    println!("{:?}", parse(&[], &None, 0, 3, "2^3^4"));
    println!("{:?}", parse(&[], &None, 0, 3, "[1,4]"));
    println!("{:?}", parse(&[], &None, 0, 3, "[1,cos(2),4]"));
    println!("{:?}", parse(&[], &None, 0, 3, "[1;cos(2);4]"));
    println!("{:?}", parse(&pu, &None, 0, 3, "#>_{xz} u"));
    println!("{:?}", parse(&pu, &None, 0, 3, "#<_{xz} u"));
    println!("{:?}", parse(&pu, &None, 0, 3, "#>_{xz}^2 u"));
    println!("{:?}", parse(&pu, &None, 0, 3, "#<_{xz}^2 u"));
    println!("{:?}", parse(&pu, &None, 0, 3, "#>_{xz} [u,u]"));
    println!("{:?}", parse(&vu, &None, 0, 3, "#> v"));
    println!("{:?}", parse(&pu, &None, 0, 3, "#> u"));
    println!(
        "{:?}",
        parse(&[], &None, 0, 3, "cos (sin (3)) + atan2(3, 4)^1.5")
    );
    println!("{:?}", parse(&[], &None, 1, 3, "fix(f,fix(b,r))"));
    println!("{:?}", parse(&lu, &None, 0, 2, "l[4..6,2,7..9]*l[0..6]"));
}
