use crate::functions::fix_newton;
use crate::pde_parser::pde_ir::ir_helper::func;
use crate::pde_parser::pde_ir::ir_helper::lexer_compositor::compact;
use crate::pde_parser::pde_lexer::var::var;
use crate::pde_parser::pde_lexer::FUN_LEN;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::char;
use nom::character::complete::u32;
use nom::combinator::opt;
use nom::multi::separated_list1;
use nom::number::complete::double;
use nom::sequence::delimited;
use nom::sequence::pair;
use nom::sequence::preceded;
use nom::sequence::tuple;
use nom::IResult;

use crate::pde_parser::pde_ir::ir_helper::lexer_compositor::LexerComp;
use crate::pde_parser::pde_lexer::term;

use super::next_id;
use super::stag;
use super::var::aanum;

// fix[e,max_iter]<func_name, init>(params...)
pub fn fix(s: &str) -> IResult<&str, LexerComp> {
    let e = preceded(tag("e="), double);
    let max_iter = preceded(tag("max_iter="), u32);
    let param = delimited(
        char('['),
        tuple((opt(e), opt(preceded(tag(","), max_iter)))),
        char(']'),
    );
    let init = delimited(char('<'), pair(aanum, var), char('>'));
    let args = delimited(char('('), separated_list1(stag(","), var), char(')'));
    tuple((tag("fix"), opt(param), init, opt(args)))(s)
        .map(|(s, (_, p, i, a))| (s, fix_constructor(p, i, a)))
}
fn fix_constructor(
    params: Option<(Option<f64>, Option<u32>)>,
    (fun_name, init): (String, LexerComp),
    args: Option<Vec<LexerComp>>,
) -> LexerComp {
    let novec = |mut i: Vec<String>| {
        if i.len() > 1 {
            panic!("Expected single value in fix_constructor, found Vect.");
        } else {
            i.pop().unwrap()
        }
    };
    let id = next_id();
    let mut args = args.unwrap_or_default();
    args.insert(0, init);
    compact(args).bind(|p| {
        let p = p.into_iter().map(|i| novec(i.to_ocl())).collect::<Vec<_>>();
        let next = p.iter().map(|i| &i[..]).collect::<Vec<&str>>();
        // next[0] is the initial value for the newton iterativ method
        let newton_names = (0..next.len() - 1)
            .map(|i| format!("_v_{}", i))
            .collect::<Vec<_>>();
        let newton_names = newton_names.iter().map(|i| &i[..]).collect::<Vec<_>>();
        let (e, max_iter) = if let Some((e, max_iter)) = params {
            (e.unwrap_or(1e-3), max_iter.unwrap_or(1000) as u32)
        } else {
            (1e-3, 1000)
        };
        let name = format!("_{}_fix", id);
        let fix = fix_newton(&name, &fun_name, &newton_names[..], e, max_iter);
        LexerComp {
            token: func(&name, next),
            funs: vec![fix],
        }
    })
}
pub fn fun(s: &str) -> IResult<&str, LexerComp> {
    alt((fix, term))(s)
    //alt((fix,kt,fun,diff,term))(s)
}

/*


Func: LexerComp = {

    <d:r"#KTx?y?z?"> "[" <name:(<Symb> ";")?> <mut eigenvalues:(<Expr> ",")*> <end:Expr?> "]" <v:Term> => {
        if let Some(end) = end {eigenvalues.push(end);}
        let dirs = d[3..].chars().map(|c| match c {
            'x' => DimDir::X,
            'y' => DimDir::Y,
            'z' => DimDir::Z,
            a @ _ => panic!("Character '{}' not expected in Func lexer for diff regex.",a)
        }).collect::<Vec<_>>();
         name.unwrap_or(current_var.clone().expect("KT call must be given the name of the variable it operate on if it is not in the context of an equation deffinition. For instance for a variable 'u' with eigenvalue of the Jacobian '2u' and for the expression 'u^2': KT[u;2u](u^2)").into()).fuse_apply(|name,_|
         compact(eigenvalues).fuse_apply(|eigenvalues,_|
             v.apply(|v| kt(name, v, eigenvalues, dirs))
         ))
    },
    <name:r"[a-zA-Z]\w*"> "(" <mut vals:(<Expr> ",")*> <end:Expr> ")" => {vals.push(end); compact(vals).fuse_apply(|i,_| func(name, i).into())},
    <d:r"#[<>]?x?y?z?(\^ *[0-9]+)?"> <v:Term> => {
        let (start,n,dirs) = if d.len() > 1 {
            let (start,s) = if &d[1..2] == ">" { (0,2) } else if &d[1..2] == "<" { (1,2) } else { (0,1) };
            let (e,n) = match d.find("^") {
                Some(e) => (e,d[e+1..].parse::<usize>().unwrap()),
                None => (d.len(),1)
            };
            let dirs = d[s..e].chars().map(|c| match c {
                'x' => DimDir::X,
                'y' => DimDir::Y,
                'z' => DimDir::Z,
                a @ _ => panic!("Character '{}' not expected in Func lexer for diff regex.",a)
            }).collect::<Vec<_>>();
            (start,n,dirs)
        } else {
            (0,1,vec![])
        };
        (0..n).fold(v, |v,i| v.apply(|v| diff(v, if (i+start)%2 == 0 { Forward(dirs.clone()) } else { Backward(dirs.clone()) })))
    },
    Term,
};

*/
