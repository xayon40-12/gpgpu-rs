use crate::descriptors::ConstructorTypes::{self, *};
use crate::descriptors::{
    FunctionConstructor::{self, *},
    SFunctionConstructor,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Function<'a> {
    pub name: &'a str,
    pub args: Vec<FunctionConstructor<'a>>,
    pub ret_type: Option<ConstructorTypes>,
    pub src: &'a str,
    pub needed: Vec<Needed<'a>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SFunction {
    pub name: String,
    pub args: Vec<SFunctionConstructor>,
    pub ret_type: Option<ConstructorTypes>,
    pub src: String,
    pub needed: Vec<SNeeded>,
}

impl<'a> From<&Function<'a>> for SFunction {
    fn from(f: &Function<'a>) -> Self {
        SFunction {
            name: f.name.into(),
            args: f.args.iter().map(|i| i.into()).collect(),
            ret_type: f.ret_type,
            src: f.src.into(),
            needed: f.needed.iter().map(|i| i.into()).collect(),
        }
    }
}

impl<'a> From<Function<'a>> for SFunction {
    fn from(f: Function<'a>) -> Self {
        (&f).into()
    }
}

#[derive(Clone, Debug)]
pub enum Needed<'a> {
    FuncName(&'a str),
    CreateFunc(Function<'a>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SNeeded {
    FuncName(String),
    CreateFunc(SFunction),
}

impl<'a> From<&Needed<'a>> for SNeeded {
    fn from(n: &Needed<'a>) -> Self {
        match n {
            Needed::FuncName(n) => SNeeded::FuncName((*n).into()),
            Needed::CreateFunc(f) => SNeeded::CreateFunc(f.into()),
        }
    }
}

pub fn functions() -> HashMap<&'static str, Function<'static>> {
    vec![
        Function {
            name: "swap",
            args: vec![FCGlobalPtr("a",CF64),FCGlobalPtr("b",CF64)],
            ret_type: None,
            src: "    double tmp = *a; *a = *b; *b = tmp;",
            needed: vec![],
        },
        Function {
            name: "mid",
            args: vec![FCParam("x",CU32),FCParam("y",CU32),FCParam("z",CU32)],
            ret_type: Some(CU32),
            src: "    return (x%get_global_work_size(0))+get_global_work_size(0)*((y%get_global_work_size(1))+get_global_work_size(1)*(z%get_global_work_size(2)));",
            needed: vec![],
        },
        Function {
            name: "c_sqrmod",
            args: vec![FCParam("src",CF64_2)],
            ret_type: Some(CF64),
            src: "    return src.x*src.x + src.y*src.y;",
            needed: vec![],
        },
        Function {
            name: "c_mod",
            args: vec![FCParam("src",CF64_2)],
            ret_type: Some(CF64),
            src: "    return sqrt(src.x*src.x + src.y*src.y);",
            needed: vec![],
        },
        Function {
            name: "c_conj",
            args: vec![FCParam("a",CF64_2)],
            ret_type: Some(CF64_2),
            src: "    return (double2)(a.x, -a.y);",
            needed: vec![],
        },
        Function {
            name: "c_times",
            args: vec![FCParam("a",CF64_2),FCParam("b",CF64_2)],
            ret_type: Some(CF64_2),
            src: "    return (double2)(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);",
            needed: vec![],
        },
        Function {
            name: "c_times_conj",
            args: vec![FCParam("a",CF64_2),FCParam("b",CF64_2)],
            ret_type: Some(CF64_2),
            src: "    return (double2)(a.x*b.x+a.y*b.y, -a.x*b.y+a.y*b.x);",
            needed: vec![],
        },
        Function {
            name: "c_divides",
            args: vec![FCParam("a",CF64_2),FCParam("b",CF64_2)],
            ret_type: Some(CF64_2),
            src: "    return c_times(a,c_conj(b))/c_sqrmod(b);",
            needed: vec![],
        },
        Function {
            name: "c_exp",
            args: vec![FCParam("x",CF64)],
            ret_type: Some(CF64_2),
            src: "    double c, s = sincos(x,&c);
    return (double2)(c,s);"
                ,
            needed: vec![],
        },
        Function {
            name: "nullIfNaN",
            args: vec![FCParam("x",CF64)],
            ret_type: Some(CF64),
            src: "    if(x!=x) {{ return 0.0; }} else {{ return x; }}",
            needed: vec![],
        },
    ].into_iter().map(|f| (f.name,f)).collect()
}

pub fn null_if_nan(x: f64) -> f64 {
    if x != x {
        0.0
    } else {
        x
    }
}

/// fix_newton create an OpenCL function called 'name' that will take one parameter (the initial point) and uses the already defined (global scope) function 'f' to which it will search the fixed point using Newton's method.
pub fn fix_newton<'a>(
    name: &'a str,
    f: &'a str,
    extra_param_f: &'a [&'a str],
    e: f64,
    max_iter: i32,
) -> SFunction {
    let mut extra = extra_param_f.join(",");
    if extra.len() > 0 {
        extra.insert(0, ',');
    }
    Function {
        name: &name,
        args: extra_param_f
            .iter()
            .fold(vec![FCParam("x", CF64)], |mut acc, i| {
                acc.push(FCParam(i, CF64));
                acc
            }),
        ret_type: Some(CF64),
        src: &format!(
            "    const double e = {};
    const double h = {};
    const double invh = {};
    double v = {f}(x{extra});
    double vp = ({f}(x+h{extra})-v)*invh;
    double d = 2*e;
    int i;
    for(i = 0; i<{max} && d>e; i++){{
        v = {f}(x{extra});
        vp = ({f}(x+h{extra})-v)*invh;
        d = x;
        x -= v/vp;
        d = fabs(d-x);
    }}
    if (i=={max}) {{
        return 0.0/0.0;
    }}
    return x;",
            e,
            e / 10.0,
            10.0 / e,
            max = max_iter,
            f = f,
            extra = extra
        ),
        needed: vec![Needed::FuncName(&f)],
    }
    .into()
}

#[test]
fn function_test() -> crate::Result<()> {
    use crate::descriptors::{
        BufferConstructor::*, ConstructorTypes::*, KernelArg::*, KernelConstructor::*, VecTypes::*,
    };
    use crate::kernels::Kernel;
    use crate::Dim;
    use crate::Handler;

    let num = 1 << 6;
    let mut gpu = Handler::builder()?
        .add_buffer("u", Data(VF64((0..num).map(|i| i as f64 - 1.0).collect())))
        .load_function("nullIfNaN")
        .create_function(Function {
            name: "f",
            args: vec![FCParam("x", CF64), FCParam("a", CF64)],
            ret_type: Some(CF64),
            src: "    return x*x - a;",
            needed: vec![],
        })
        .create_function(fix_newton("fix", "f", &["a"], 1e-4, 1000))
        .create_kernel(&Kernel {
            name: "_main",
            src: "    double a = u[x];
    u[x] = nullIfNaN(floor(fix(a,a)*1000)/1000.0);",
            args: vec![KCBuffer("u", CF64)],
            needed: vec![],
        })
        .build()?;

    gpu.run_arg("_main", Dim::D1(num), &[Buffer("u")])?;
    assert_eq!(
        gpu.get("u")?.VF64(),
        (0..num)
            .map(|j| null_if_nan((f64::sqrt(j as f64 - 1.0) * 1000.0).floor() / 1000.0))
            .collect::<Vec<_>>()
    );

    Ok(())
}
