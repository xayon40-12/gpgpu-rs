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
    let u = PDE { var_name: "u".into(), boundary: "b".into(), dim: 3, vector: false};
    let v = PDE { var_name: "v".into(), boundary: "b".into(), dim: 3, vector: true};
    println!("{:?}", pde_lexer::ExprParser::new().parse(&vec![u.clone()],"22 + (1 - 4+-15.7^(1--0.3+D+u))*7"));
    println!("{:?}", pde_lexer::ExprParser::new().parse(&vec![],"2^3^4"));
    println!("{:?}", pde_lexer::ExprParser::new().parse(&vec![u.clone()],"#>xz u"));
    println!("{:?}", pde_lexer::ExprParser::new().parse(&vec![u.clone()],"#<xz u"));
    println!("{:?}", pde_lexer::ExprParser::new().parse(&vec![u.clone()],"#>xz^2 u").unwrap().to_ocl());
    println!("{:?}", pde_lexer::ExprParser::new().parse(&vec![u.clone()],"#<xz^2 u").unwrap().to_ocl());
    println!("{:?}", pde_lexer::ExprParser::new().parse(&vec![u.clone()],"#>xz [u,u]"));
    println!("{:?}", pde_lexer::ExprParser::new().parse(&vec![v.clone()],"#> v"));
    println!("{:?}", pde_lexer::ExprParser::new().parse(&vec![u.clone()],"#> u"));
    println!("{:?}", pde_lexer::ExprParser::new().parse(&vec![],"cos (sin 3) + atan2(3, 4)^1.5"));
    println!("{:?}", pde_lexer::ExprParser::new().parse(&vec![],"cos (sin 3) + atan2(3, 4)^1.5").unwrap().to_ocl());
}
