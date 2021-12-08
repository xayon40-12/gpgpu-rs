use crate::functions::fix_newton;
use crate::pde_parser::pde_ir::{ir_helper::*, lexer_compositor::*, DiffDir::*, DimDir};
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::char;
use nom::character::complete::one_of;
use nom::combinator::map_res;
use nom::multi::separated_list1;
use nom::number::complete::double;
use nom::sequence::delimited;
use nom::sequence::tuple;
use nom::IResult;
use std::str::FromStr;

use super::pde_ir::lexer_compositor::LexerComp;
use super::{Parsed, SPDETokens, DPDE};

pub fn parse<'a>(
    context: &[DPDE],                 // var name
    current_var: &Option<SPDETokens>, // boundary funciton name
    fun_len: usize,                   // number of existing functions
    global_dim: usize,                // dim
    math: &'a str,
) -> IResult<&'a str, LexerComp> {
    expr(math)
}

fn expr(s: &str) -> IResult<&str, LexerComp> {
    let a = |s| tuple((expr, tag("+"), factor))(s).map(|(s, (l, _, r))| (s, l + r));
    let b = |s| tuple((expr, tag("-"), factor))(s).map(|(s, (l, _, r))| (s, l - r));
    let c = |s| factor(s);
    alt((a, b, c))(s)
}

fn factor(s: &str) -> IResult<&str, LexerComp> {
    let a = |s| tuple((factor, tag("*"), pow))(s).map(|(s, (l, _, r))| (s, l * r));
    let b = |s| tuple((factor, tag("/"), pow))(s).map(|(s, (l, _, r))| (s, l / r));
    let c = |s| pow(s);
    alt((a, b, c))(s)
}

fn pow(s: &str) -> IResult<&str, LexerComp> {
    let a = |s| tuple((func, alt((tag("^"), tag("**"))), pow))(s).map(|(s, (l, _, r))| (s, l ^ r));
    let b = |s| func(s);
    alt((a, b))(s)
}

fn func(s: &str) -> IResult<&str, LexerComp> {}

fn term(s: &str) -> IResult<&str, LexerComp> {
    let parens = delimited(char('('), expr, char(')'));
    let arr = |s| {
        delimited(char('['), separated_list1(one_of(",;"), expr), char(']'))(s)
            .map(|(s, v)| (s, compact(v).map(|i| vect(i))))
    };
    alt((num, symb, parens, arr))(s)
}

fn symb(s: &str) -> IResult<&str, LexerComp> {}

fn num(s: &str) -> IResult<&str, LexerComp> {
    double(s).map(|(s, d)| (s, d.into()))
}

/*


Symb: LexerComp = {
    r"[a-zA-Z]\w*" => vars.iter().fold(symb(<>),
        |acc,DPDE{var_name: v, boundary: b, var_dim: d, vec_dim: vd}| if <> == v { if *vd>1 { Indexable::new_vector(*d,gd,*vd,&v,&b) } else { Indexable::new_scalar(*d,gd,&v,&b) } } else { acc }).into(),
    r"[a-zA-Z]\w*\[\d+(\.\.\d+)?(,\d+(\.\.\d+)?)*\]" => {
        let tmp = <>.split("[").collect::<Vec<_>>();
        let name = tmp[0];
        let vals = tmp[1][..tmp[1].len()-1].split(",").collect::<Vec<_>>();
        let DPDE{var_name: v, boundary: b, var_dim: d, vec_dim: vd} = vars.iter().fold(None, |acc,d|
            if name == d.var_name { Some(d) } else { acc }).expect(&format!("symbol \"{}\" not found in equations or pdes.", name));
        let slices = vals.into_iter().map(|i| {
            let s = i.split("..").map(|d| d.parse::<usize>().unwrap()).collect::<Vec<_>>();
            if s.len() == 1 {
                s[0]..s[0]+1
            } else {
                s[0]..s[1]+1 // inclusive range for lexer
            }
        }).collect::<Vec<_>>();
        Indexable::new_slice(*d, gd, *vd, &slices[..], &v, &b).into()
    },
};

Func: LexerComp = {
    // fix[e,max_iter](func_name, params...)
    "fix" <params:("[" <Num> "," <Num> "]")?> "(" <f:r"[a-zA-Z]\w*"> <vals:("," <Expr>)+> ")" => {
          let novec = |mut i: Vec<String>| if i.len() > 1 {
              panic!("Expected single value, found Vect.");
          } else {
              i.pop().unwrap()
          };
          compact(vals).fuse_apply(|p,len| {
                let p = p.into_iter().map(|i| novec(i.to_ocl())).collect::<Vec<_>>();
                let next = p.iter().map(|i| &i[..]).collect::<Vec<&str>>();
                // next[0] is the initial value for the nexton iterativ method
                let newton_names = (0..next.len()-1).map(|i| format!("_v_{}", i)).collect::<Vec<_>>();
                let newton_names = newton_names.iter().map(|i| &i[..]).collect::<Vec<_>>();
                let (e,max_iter) = if let Some((e,max_iter)) = params {
                    (e,max_iter as i32)
                } else {
                    (1e-3,1000)
                };
                let name = format!("_{}_fix", fun_len+len);
                let fix = fix_newton(&name,f,&newton_names[..],e,max_iter);
                LexerComp {
                    token: func(&name, next),
                    funs: vec![fix],
                }
          })
    },
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
