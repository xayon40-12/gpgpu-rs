use crate::functions::fix_newton;
use crate::pde_parser::pde_ir::ir_helper::diff;
use crate::pde_parser::pde_ir::ir_helper::func;
use crate::pde_parser::pde_ir::ir_helper::kt;
use crate::pde_parser::pde_ir::ir_helper::lexer_compositor::compact;
use crate::pde_parser::pde_lexer::expr;
use crate::pde_parser::pde_lexer::var::var;
use crate::pde_parser::pde_lexer::DiffDir::{Backward, Forward};
use crate::DimDir;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::char;
use nom::character::complete::one_of;
use nom::character::complete::space0;
use nom::character::complete::u32;
use nom::combinator::opt;
use nom::multi::many1;
use nom::multi::separated_list0;
use nom::multi::separated_list1;
use nom::number::complete::double;
use nom::sequence::delimited;
use nom::sequence::pair;
use nom::sequence::preceded;
use nom::sequence::terminated;
use nom::sequence::tuple;
use nom::IResult;

use crate::pde_parser::pde_ir::ir_helper::lexer_compositor::LexerComp;
use crate::pde_parser::pde_lexer::term;

use super::next_id;
use super::stag;
use super::var::aanum;
use super::CURRENT_VAR;

// fix[e,max_iter]<func_name, init>(params...)
pub fn fix(s: &str) -> IResult<&str, LexerComp> {
    let e = preceded(tag("e="), double);
    let max_iter = preceded(tag("max_iter="), u32);
    let param = delimited(
        char('['),
        tuple((opt(e), opt(preceded(tag(","), max_iter)))),
        char(']'),
    );
    let init = delimited(char('<'), pair(aanum, preceded(stag(","), expr)), char('>'));
    let args = delimited(char('('), separated_list1(stag(","), expr), char(')'));
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

fn diff_dir(s: &str) -> IResult<&str, Vec<char>> {
    preceded(
        char('_'),
        alt((
            |s| one_of("xyz")(s).map(|(s, i)| (s, vec![i])),
            delimited(char('{'), many1(one_of("xyz")), char('}')),
        )),
    )(s)
}

fn kt_diff(s: &str) -> IResult<&str, LexerComp> {
    let d = preceded(tag("#KT"), opt(diff_dir));
    let options = opt(delimited(
        char('['),
        tuple((
            opt(terminated(var, stag(";"))),
            opt(delimited(stag("theta="), double, stag(";"))),
            separated_list0(stag(","), expr),
        )),
        char(']'),
    ));
    tuple((d, options, space0, term))(s).map(|(s, (d, o, _, t))| (s, kt_constructor(d, o, t)))
}

fn kt_constructor(
    dirs: Option<Vec<char>>,
    options: Option<(Option<LexerComp>, Option<f64>, Vec<LexerComp>)>,
    term: LexerComp,
) -> LexerComp {
    let (name, theta, eigenvalues) = options.unwrap_or((None, None, vec![]));
    let dirs = dirs
        .unwrap_or_default()
        .iter()
        .map(|c| match c {
            'x' => DimDir::X,
            'y' => DimDir::Y,
            'z' => DimDir::Z,
            a => panic!(
                "Character '{}' not expected in Func lexer for diff regex.",
                a
            ),
        })
        .collect::<Vec<_>>();
    let mut current_var = None;
    CURRENT_VAR.with(|d| {
        current_var = Some(d.borrow().clone());
    });
    let current_var = current_var.expect("Thread error, could not retreive current variable.");
    let theta = theta.unwrap_or(1.1); // MUSIC default
    name.unwrap_or_else(|| current_var.clone().expect("KT call must be given the name of the variable it operate on if it is not in the context of an equation deffinition. For instance for a variable 'u' with eigenvalue of the Jacobian '2u' and for the expression 'u^2': KT[u;2u](u^2)").into()).bind(|name|
         compact(eigenvalues).bind(|eigenvalues|
             term.map(|v| kt(name, v, eigenvalues, theta, dirs))
         ))
}

fn fun_call(s: &str) -> IResult<&str, LexerComp> {
    pair(
        aanum,
        delimited(stag("("), separated_list1(stag(","), expr), stag(")")),
    )(s)
    .map(|(s, (name, l))| (s, compact(l).bind(|l| func(&name, l).into())))
}

fn bf_diff(s: &str) -> IResult<&str, LexerComp> {
    let d = preceded(
        char('#'),
        pair(
            opt(|s| one_of("<>")(s).map(|(s, c)| (s, c == '>'))),
            opt(diff_dir),
        ),
    );
    let pow = preceded(char('^'), u32);
    tuple((d, opt(pow), space0, term))(s)
        .map(|(s, ((bf, dirs), p, _, t))| (s, bf_diff_constructor(bf, dirs, p, t)))
}

fn bf_diff_constructor(
    forward: Option<bool>,
    dirs: Option<Vec<char>>,
    pow: Option<u32>,
    term: LexerComp,
) -> LexerComp {
    let start = if forward.unwrap_or(true) { 0 } else { 1 };
    let n = pow.unwrap_or(1);
    let dirs = dirs
        .unwrap_or_default()
        .iter()
        .map(|c| match c {
            'x' => DimDir::X,
            'y' => DimDir::Y,
            'z' => DimDir::Z,
            a => panic!(
                "Character '{}' not expected in Func lexer for diff regex.",
                a
            ),
        })
        .collect::<Vec<_>>();
    (0..n).fold(term, |v, i| {
        v.map(|v| {
            diff(
                v,
                if (i + start) % 2 == 0 {
                    Forward(dirs.clone())
                } else {
                    Backward(dirs.clone())
                },
            )
        })
    })
}

pub fn fun(s: &str) -> IResult<&str, LexerComp> {
    alt((fix, kt_diff, fun_call, bf_diff, term))(s)
}
