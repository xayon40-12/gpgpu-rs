use crate::integrators::pde_ir::SPDETokens::*;
use crate::integrators::pde_ir::*;

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
    let min = |a, b| Func("min".into(), vec![a, b]);
    let max = |a, b| Func("max".into(), vec![a, b]);
    let abs = |a| Func("abs".into(), vec![a]);
    let sign = |a| Func("sign".into(), vec![a]);
    let sgn = |a, b, c| ((sign(a) + sign(b)) * Const(0.5) + sign(c)) * Const(0.5);
    let minmod = |a: SPDETokens, b: SPDETokens, c: SPDETokens| {
        sgn(a.clone(), b.clone(), c.clone()) * min(abs(a), min(abs(b), abs(c)))
    };
    let ux = |u: Indexable| {
        let up = Indx(u.clone().apply_idx(p));
        let uc = Indx(u.clone());
        let um = Indx(u.apply_idx(m));
        minmod(
            Const(theta) * (uc.clone() - um.clone()),
            (up.clone() - um) * Const(0.5),
            Const(theta) * (up - uc),
        )
    };
    let up = |u: Indexable| Indx(u.clone().apply_idx(p)) - ux(u) * Const(0.5);
    let um = |u: Indexable| Indx(u.clone().apply_idx(m)) + ux(u) * Const(0.5);
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
