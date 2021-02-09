use crate::integrators::SPDE;
use crate::pde_lexer;

fn parse(math: String) -> Vec<SPDE> {
    vec![]
}

#[test]
fn pde_lexer() {
    assert!(pde_lexer::ExprParser::new().parse("22").is_ok());
    println!("{:?}", pde_lexer::ExprParser::new().parse("22 + (4-15.7^(1-0.3))*7"));
}
