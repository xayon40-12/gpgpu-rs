use crate::integrators::pde_ir::SPDETokens::*;
use crate::integrators::pde_ir::{ir_helper::*, *};

pub fn kt(u: &SPDETokens, fu: &SPDETokens, eigs: &Vec<SPDETokens>, d: usize) -> SPDETokens {
    let iv = Symb(["ivdx", "ivdy", "ivdz"][d].into());
    Const(-1f64) * (h(u, fu, eigs, d, 1) - h(u, fu, eigs, d, -1)) * iv
}

pub fn h(
    u: &SPDETokens,
    fu: &SPDETokens,
    eigs: &Vec<SPDETokens>,
    d: usize,
    idir: i32,
) -> SPDETokens {
    let p = &idx(1, idir, d);
    let m = &idx(-1, idir, d);
    let theta = 2.0;
    let min = |a, b| func("min".into(), vec![a, b]);
    let max = |a, b| func("max".into(), vec![a, b]);
    let abs = |a| func("fabs".into(), vec![a]);
    let sign = |a| func("sign".into(), vec![a]);
    let minmod = |a: SPDETokens, b: SPDETokens| {
        (sign(a.clone()) + sign(b.clone())) * Const(0.5) * min(abs(a), abs(b))
    };
    let ux = |u: Indexable| {
        let up = Indx(u.clone().apply_idx(&idx(1, 1, d)));
        let uc = Indx(u.clone());
        let um = Indx(u.apply_idx(&idx(-1, -1, d)));
        minmod(
            Const(theta) * (uc.clone() - um.clone()),
            minmod((up.clone() - um) * Const(0.5), Const(theta) * (up - uc)),
        )
    };
    let up = |u: Indexable| {
        let u = u.clone().apply_idx(p);
        Indx(u.clone()) - ux(u) * Const(0.5)
    };
    let um = |u: Indexable| {
        let u = u.clone().apply_idx(m);
        Indx(u.clone()) + ux(u) * Const(0.5)
    };
    let a = |eigs: &Vec<SPDETokens>| {
        let mut tmp = eigs
            .iter()
            .map(|e| max(abs(ap(e, up)), abs(ap(e, um))))
            .collect::<Vec<_>>();
        let fst = tmp
            .pop()
            .expect("There must be at least one eigenvalue given for KT scheme.");
        tmp.into_iter().fold(fst, |acc, i| max(acc, i))
    };
    (ap(fu, up) + ap(fu, um) - a(eigs) * (ap(u, up) - ap(u, um))) * Const(0.5)
}
fn ap(u: &SPDETokens, f: impl Fn(Indexable) -> SPDETokens + Copy) -> SPDETokens {
    u.clone().apply_indexable(f)
}

fn idx(dir: i32, idir: i32, d: usize) -> [i32; 4] {
    let i = (dir + idir) / 2;
    [[i, 0, 0, 0], [0, i, 0, 0], [0, 0, i, 0]][d]
}
