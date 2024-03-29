use crate::algorithms::{
    AlgorithmParam, SAlgorithm,
    SNeeded::{self, *},
};
use crate::descriptors::{ConstructorTypes, Types};
use crate::descriptors::{ConstructorTypes::*, KernelArg::*, KernelConstructor::*};
use crate::dim::{
    Dim::{self, *},
    DimDir,
};
use crate::kernels::Kernel;
use crate::Handler;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum STEP {
    PDE,
    EQPDE,
    BETWEENPDE(u32),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SPDE {
    pub dvar: String,
    pub expr: Vec<String>, //one String for each dimension of the vectorial pde
    pub step: STEP,
}

pub struct IntegratorParam {
    pub t: f64,
    pub increment_name: String,
    pub args: Vec<(String, Types)>,
}

// Each PDE must be first order in time. A higher order PDE can be cut in multiple first order PDE.
// Example: d2u/dt2 + du/dt = u   =>   du/dt = z, dz/dt = u.
// It is why the parameter pdes is a Vec.
pub type CreatePDE =
    fn(&str, f64, Vec<SPDE>, Option<Vec<String>>, Vec<(String, ConstructorTypes)>) -> SAlgorithm;
pub fn create_euler_pde(
    name: &str,
    dt: f64,
    pdes: Vec<SPDE>,
    needed_buffers: Option<Vec<String>>,
    params: Vec<(String, ConstructorTypes)>,
) -> SAlgorithm {
    multistages_algorithm(name, &pdes, needed_buffers, params, dt, vec![vec![1.0]])
}
pub fn create_projector_corrector_pde(
    name: &str,
    dt: f64,
    pdes: Vec<SPDE>,
    needed_buffers: Option<Vec<String>>,
    params: Vec<(String, ConstructorTypes)>,
) -> SAlgorithm {
    multistages_algorithm(
        name,
        &pdes,
        needed_buffers,
        params,
        dt,
        vec![vec![1.0], vec![0.5, 0.5]],
    )
}
pub fn create_rk4_pde(
    name: &str,
    dt: f64,
    pdes: Vec<SPDE>,
    needed_buffers: Option<Vec<String>>,
    params: Vec<(String, ConstructorTypes)>,
) -> SAlgorithm {
    multistages_algorithm(
        name,
        &pdes,
        needed_buffers,
        params,
        dt,
        vec![
            vec![0.5],
            vec![0.0, 0.5],
            vec![0.0, 0.0, 1.0],
            vec![1. / 6., 1. / 3., 1. / 3., 1. / 6.],
        ],
    )
}

fn multistages_kernels(
    name: &str,
    pdes: &Vec<SPDE>,
    needed_buffers: &Option<Vec<String>>,
    params: Vec<(String, ConstructorTypes)>,
    stages: &Vec<Vec<f64>>,
) -> Vec<SNeeded> {
    let mut args = vec![KCBuffer("dst", CF64)];
    args.extend(pdes.iter().map(|pde| KCBuffer(&pde.dvar, CF64)));
    if let Some(ns) = &needed_buffers {
        args.extend(ns.iter().map(|n| KCBuffer(&n, CF64)));
    }
    args.extend(params.iter().map(|t| KCParam(&t.0, t.1)));
    let mut needed = pdes
        .iter()
        .map(|d| {
            let mut id = "x+x_size*(y+y_size*z)".to_string();
            let len = d.expr.len();
            if len > 1 {
                id = format!("{}*({})", len, id);
            }
            let mut expr = String::new();
            for i in 0..len {
                expr += &format!("    dst[{i}+_i] = {};\n", &d.expr[i], i = i);
            }
            NewKernel(
                (&Kernel {
                    name: &format!("{}_{}", &name, &d.dvar),
                    args: args.clone(),
                    src: &format!("    uint _i = {};\n{}", id, expr),
                    needed: vec![],
                })
                    .into(),
            )
        })
        .collect::<Vec<_>>();
    for (i, v) in stages.iter().enumerate() {
        let mut args = vec![
            KCBuffer("dst", CF64),
            KCBuffer("src", CF64),
            KCParam("h", CF64),
        ];
        let argnames = (0..v.len())
            .map(|i| format!("src{}", i))
            .collect::<Vec<_>>();
        let mut src = String::new();
        for (i, v) in v.iter().enumerate() {
            args.push(KCBuffer(&argnames[i], CF64));
            if *v != 0.0 {
                src = format!("{} + {}*{}[i]", src, v, &argnames[i]);
            }
        }
        src = format!(
            "    uint i = x+x_size*(y+y_size*z);\n    dst[i] = src[i] + h*({});",
            &src[3..]
        );
        let src_eq = format!(
            "    uint i = x+x_size*(y+y_size*z);\n    dst[i] = {}[i];",
            &argnames[i] // TODO: optimize: use argnames[0] here and in multistages_algorithm for eq, then only alocate one buffer for them
        );

        needed.push(NewKernel(
            (&Kernel {
                name: &format!("{}_stage{}", name, i),
                args: args.clone(),
                src: &src,
                needed: vec![],
            })
                .into(),
        ));
        needed.push(NewKernel(
            (&Kernel {
                name: &format!("{}_stage{}_eq", name, i),
                args,
                src: &src_eq,
                needed: vec![],
            })
                .into(),
        ));
    }
    needed
}

fn multistages_algorithm(
    name: &str,
    pdes: &Vec<SPDE>,
    needed_buffers: Option<Vec<String>>,
    params: Vec<(String, ConstructorTypes)>,
    dt: f64,
    stages: Vec<Vec<f64>>,
) -> SAlgorithm {
    let name = name.to_string();
    let vars = pdes
        .iter()
        .map(|d| {
            (
                format!("{}_{}", &name, &d.dvar),
                d.dvar.clone(),
                d.expr.len(),
                d.step.clone(),
            )
        })
        .collect::<Vec<_>>();
    let mut vars_ranges = vars
        .iter()
        .enumerate()
        .map(|(i, (_, _, _, s))| match s {
            STEP::PDE | STEP::EQPDE | STEP::BETWEENPDE(0) => (0, i),
            STEP::BETWEENPDE(j) => (*j, i),
        })
        .collect::<Vec<_>>();
    vars_ranges.sort();
    let (_, vars_ranges) = vars_ranges
        .into_iter()
        .fold((0, vec![vec![]]), |(c, mut a), (j, i)| {
            if j == c {
                let l = a.len() - 1;
                a[l].push(i);
                (c, a)
            } else {
                a.push(vec![i]);
                (j, a)
            }
        });
    let mut len = (if stages.len() > 1 { 2 } else { 1 } + stages.len()) * vars.len();
    let nb_pde_buffers = len;
    if let Some(ns) = &needed_buffers {
        len += ns.len();
    }
    for (i, v) in stages.iter().enumerate() {
        if v.len() != i + 1 {
            panic!("In multisatges algorithm the coefficients must be given as a vector for each stages in the order, the first stage does not need a coefficent nor an empty, then each stage needs one more coefficient (3 coefficient for the fourth stage for instance) and the las vector correspond to how to sum each of the computed stages at the end thus it needs as many coefficient as there are stages.")
        }
    }

    let nb_per_stages = if stages.len() > 1 { 2 } else { 1 } + stages.len();
    let tmpid = nb_per_stages - 1;
    let nb_stages = stages.len();
    let needed = multistages_kernels(&name, &pdes, &needed_buffers, params, &stages);
    SAlgorithm {
        name: name.clone(),
        callback: std::rc::Rc::new(
            move |h: &mut Handler,
                  dim: Dim,
                  _dimdir: &[DimDir],
                  bufs: &[&str],
                  other: AlgorithmParam| {
                // bufs[0] = dst
                // bufs[1,2,...] = differential equation buffer holders in the same order as giver for
                // create_euler function
                // bufs[i] must write in bufs[i-1]
                let _dim: [usize; 3] = dim.into();
                let d = _dim.iter().fold(1, |a, i| a * i);
                if bufs.len() != len {
                    panic!(
                        "Multistages algorithm \"{}\" must be given {} buffer arguments.",
                        &name, &len
                    );
                }
                let IntegratorParam{t,ref increment_name,args: iargs} = other
                .downcast_ref("There must be an Ref(&IntegratorParam) given as optional argument in Multistages integrator algorithm.");
                let mut args = vec![BufArg("", ""); vars.len() + 1];
                if let Some(ns) = &needed_buffers {
                    let mut i = nb_pde_buffers;
                    for n in ns {
                        args.push(BufArg(&bufs[i], &n));
                        i += 1;
                    }
                }
                args.extend(iargs.iter().map(|i| Param(&i.0, i.1)));
                args.push(Param(increment_name, (*t).into()));
                let time_id = args.len() - 1;
                let argnames = (0..nb_per_stages)
                    .map(|i| format!("src{}", i))
                    .collect::<Vec<_>>();
                for s in 0..nb_stages {
                    let mut pred = vec![];
                    for r in &vars_ranges {
                        for &i in r {
                            args[0] = BufArg(&bufs[nb_per_stages * i + (s + 1)], "dst");
                            for i in 0..vars.len() {
                                args[1 + i] = BufArg(
                                    &bufs[nb_per_stages * i
                                        + if (s == 0 && !pred.contains(&i))
                                            || (s == nb_stages - 1 && pred.contains(&i))
                                        {
                                            0
                                        } else {
                                            tmpid
                                        }],
                                    &vars[i].1,
                                );
                            }
                            h.run_arg(&vars[i].0, dim, &args)?;
                        }
                        args[time_id] = Param(
                            increment_name,
                            (*t + stages[s].iter().fold(0.0, |a, i| a + i) * dt).into(),
                        ); // increment time for next stage
                        r.iter().for_each(|i| pred.push(*i));
                        for &i in r {
                            let mut stage_args = vec![
                                BufArg(
                                    &bufs[nb_per_stages * i
                                        + if s == nb_stages - 1 { 0 } else { tmpid }],
                                    "dst",
                                ),
                                BufArg(&bufs[nb_per_stages * i], "src"),
                                Param("h", dt.into()),
                            ];
                            stage_args.extend(
                                (0..s + 1).map(|j| {
                                    BufArg(&bufs[nb_per_stages * i + (j + 1)], &argnames[j])
                                }),
                            );
                            let step = &vars[i].3;
                            let stage_name = &match step {
                                STEP::PDE => format!("{}_stage{}", name, s),
                                STEP::EQPDE | STEP::BETWEENPDE(_) => {
                                    format!("{}_stage{}_eq", name, s)
                                }
                            };
                            h.run_arg(stage_name, D1(d * vars[i].2), &stage_args)?;
                            // vars[i].2 correspond to the vectorial dim of the current pde
                        }
                    }
                }

                Ok(None)
            },
        ),
        needed,
    }
}
