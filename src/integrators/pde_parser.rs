use crate::integrators::{SPDE,pde_ir::{SPDETokens,ir_helper::PDE}};
use crate::pde_lexer;

/*
fn parse(math: String) -> Vec<SPDE> {
    vec![]
}
*/

#[test]
fn pde_lexer() {
    assert!(pde_lexer::ExprParser::new().parse(&vec![],"22").is_ok());
    let pde = PDE { var_name: "u".into(), boundary: "b".into(), dim: 3 };
    println!("{:?}", pde_lexer::ExprParser::new().parse(&vec![pde],"22 + (4-15.7^(1-0.3+D+u))*7"));
    println!("{:?}", pde_lexer::ExprParser::new().parse(&vec![],"2^3^4"));
    println!("{:?}", pde_lexer::ExprParser::new().parse(&vec![],"cos (sin 3) + atan2 3 4^1.5"));
    println!("{:?}", pde_lexer::ExprParser::new().parse(&vec![],"cos (sin 3) + atan2 3 4^1.5").unwrap().to_ocl());
}
