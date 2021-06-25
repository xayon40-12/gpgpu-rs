use crate::integrators::pde_ir::*;

pub fn kt(u: &SPDETokens, eigs: &Vec<SPDETokens>, d: usize) -> SPDETokens {
    let iv = SPDETokens::Symb(["ivdx", "ivdy", "ivdz"][d].into());
    -(h(u, eigs, d, 1) - h(u, eigs, d, -1)) / iv
}

pub fn h(u: &SPDETokens, eigs: &Vec<SPDETokens>, d: usize, dir: usize) -> SPDETokens {
    let up = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]][d];
    u.apply_idx()
}
