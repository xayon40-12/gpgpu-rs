use crate::integrators::{SPDE,pde_ir::SPDETokens};
use crate::pde_lexer;

/*
fn parse(math: String) -> Vec<SPDE> {
    vec![]
}
*/

#[test]
fn pde_lexer() {
    assert!(pde_lexer::ExprParser::new().parse("22").is_ok());
    println!("{:?}", pde_lexer::ExprParser::new().parse("22 + (4-15.7^(1-0.3))*7"));
    println!("{:?}", pde_lexer::ExprParser::new().parse("2^3^4"));
    println!("{:?}", pde_lexer::ExprParser::new().parse("cos (sin 3) + atan2 3 4^1.5"));
    println!("{:?}", pde_lexer::ExprParser::new().parse("cos (sin 3) + atan2 3 4^1.5").unwrap().to_ocl());
}
