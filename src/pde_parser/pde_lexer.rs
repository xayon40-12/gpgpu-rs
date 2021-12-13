use crate::pde_parser::pde_ir::{ir_helper::*, lexer_compositor::*};
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::alpha0;
use nom::character::complete::alphanumeric0;
use nom::character::complete::one_of;
use nom::character::complete::space0;
use nom::character::complete::u32;
use nom::combinator::opt;
use nom::error::Error;
use nom::multi::separated_list1;
use nom::number::complete::double;
use nom::sequence::delimited;
use nom::sequence::pair;
use nom::sequence::preceded;
use nom::sequence::tuple;
use nom::Finish;
use nom::IResult;
use std::cell::RefCell;
use std::ops::Range;

use super::pde_ir::lexer_compositor::LexerComp;
use super::{SPDETokens, DPDE};

thread_local!(
    static VARS: RefCell<Vec<DPDE>> = RefCell::new(vec![]);
    static CURRENT: RefCell<Option<SPDETokens>> = RefCell::new(None);
    static GLOBAL_DIM: RefCell<usize> = RefCell::new(0);
    static FUN_LEN: RefCell<usize> = RefCell::new(0);
);

pub fn parse<'a>(
    context: &[DPDE],                 // var name
    current_var: &Option<SPDETokens>, // boundary funciton name
    fun_len: usize,                   // number of existing functions
    global_dim: usize,                // dim
    math: &'a str,
) -> Result<(&'a str, LexerComp), Error<&'a str>> {
    VARS.with(|v| *v.borrow_mut() = context.to_vec());
    CURRENT.with(|v| *v.borrow_mut() = current_var.clone());
    GLOBAL_DIM.with(|v| *v.borrow_mut() = global_dim);
    FUN_LEN.with(|v| *v.borrow_mut() = fun_len);
    delimited(space0, expr, space0)(math).finish()
}

fn stag(t: &'static str) -> impl Fn(&str) -> IResult<&str, &str> {
    move |s| delimited(space0, tag(t), space0)(s)
}

fn expr(s: &str) -> IResult<&str, LexerComp> {
    let add = |s| tuple((expr, stag("+"), factor))(s).map(|(s, (l, _, r))| (s, l + r));
    let sub = |s| tuple((expr, stag("-"), factor))(s).map(|(s, (l, _, r))| (s, l - r));
    alt((add, sub, factor))(s)
}

fn factor(s: &str) -> IResult<&str, LexerComp> {
    let mul = |s| tuple((factor, stag("*"), pow))(s).map(|(s, (l, _, r))| (s, l * r));
    let div = |s| tuple((factor, stag("/"), pow))(s).map(|(s, (l, _, r))| (s, l / r));
    alt((mul, div, pow))(s)
}

fn pow(s: &str) -> IResult<&str, LexerComp> {
    let pow =
        |s| tuple((func, alt((stag("^"), stag("**"))), pow))(s).map(|(s, (l, _, r))| (s, l ^ r));
    alt((pow, func))(s)
}

fn func(s: &str) -> IResult<&str, LexerComp> {
    term(s) // TODO implement func
}

fn array<T: Clone>(
    f: impl Fn(&str) -> IResult<&str, T> + Copy,
) -> impl Fn(&str) -> IResult<&str, Vec<T>> {
    move |s| {
        delimited(
            stag("["),
            separated_list1(delimited(space0, one_of(",;"), space0), f),
            stag("]"),
        )(s)
        .map(|(s, vs)| (s, vs.to_vec()))
    }
}

fn term(s: &str) -> IResult<&str, LexerComp> {
    let parens = delimited(stag("("), expr, stag(")"));
    let arr = |s| array(expr)(s).map(|(s, v)| (s, compact(v).map(vect)));
    let unary_minus = preceded(stag("-"), expr);
    alt((num, symb, parens, arr, unary_minus))(s)
}

fn extract_variable(name: String, r: &[Range<usize>]) -> LexerComp {
    let mut dpde = None;
    VARS.with(|vars| {
        vars.borrow().iter().for_each(|v| {
            if v.var_name == name {
                dpde = Some(v.clone());
            }
        })
    });
    let mut gd = None;
    GLOBAL_DIM.with(|d| {
        gd = Some(*d.borrow());
    });
    let gd = gd.expect("Thread error, could not retreive global dim.");
    if let Some(DPDE {
        var_name: v,
        boundary: b,
        var_dim: d,
        vec_dim: vd,
    }) = dpde
    {
        if !r.is_empty() {
            Indexable::new_slice(d, gd, vd, r, &v, &b)
        } else if vd > 1 {
            Indexable::new_vector(d, gd, vd, &v, &b)
        } else {
            Indexable::new_scalar(d, gd, &v, &b)
        }
    } else if !r.is_empty() {
        panic!("Symbol \"{}\" is not indexable.", name)
    } else {
        Symb(name)
    }
    .into()
}

fn range(s: &str) -> IResult<&str, Range<usize>> {
    pair(u32, opt(preceded(stag(".."), u32)))(s)
        .map(|(s, (a, b))| (s, a as usize..b.unwrap_or(a + 1) as usize))
}

fn symb(s: &str) -> IResult<&str, LexerComp> {
    let name = |s| pair(alpha0, alphanumeric0)(s).map(|(s, (a, b))| (s, format!("{}{}", a, b)));
    let var = |s| name(s).map(|(s, v)| (s, extract_variable(v, &[])));
    let vect_idx = |s| pair(name, array(range))(s).map(|(s, (v, u))| (s, extract_variable(v, &u)));
    let x = alt((vect_idx, var))(s);
    x
}

fn num(s: &str) -> IResult<&str, LexerComp> {
    double(s).map(|(s, d)| (s, d.into()))
}

/*


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
