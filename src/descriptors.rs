#[derive(Clone)]
pub enum KernelDescriptor<'a> {
    Param(&'a str,Type),
    Buffer(&'a str),
    BufArg(&'a str,&'a str) // BufArg(mem buf, kernel buf)
}

pub enum BufferDescriptor {
    Len(f64,usize), // Len(repeated value, len)
    Data(Vec<f64>)
}

use std::any::{Any,type_name};
use std::collections::BTreeMap;

macro_rules! impl_types {
    ($name:ident, $($case:ident|$case_t:ident) +) => {
        impl $name {
            pub fn iner_any(&self) -> &dyn Any {
                iner_each!(self,$name,buf,buf as &dyn Any)
            }

            pub fn type_name(&self) -> &str {
                match self {
                    $($name::$case(_) => type_name::<$case_t>(),) +
                }
            }

            pub fn type_name_ocl(&self) -> &str {
                let ocl_names: BTreeMap<&'static str,&'static str> = vec![
                    ("f64","double"),
                    ("f32","float"),
                ].into_iter().collect();
                ocl_names[self.type_name()]
            }
        }
    };
}
macro_rules! gen_types {
    ($namebuftype:ident $nametype:ident, $($case:ident|$case_t:ident) +) => {
        macro_rules! iner_each {
            ($match:expr, $enum:ident, $var:pat, $todo:expr) => {
                match $match {
                    $($enum::$case($var) => $todo,) +
                }
            };
        }

        #[derive(Debug)]
        pub enum $namebuftype {
            $($case(ocl::Buffer<$case_t>),) +
        }

        #[derive(Debug,Clone)]
        pub enum $nametype {
            $($case($case_t),) +
        }

        impl_types!($namebuftype, $($case|$case_t) +);
        impl_types!($nametype, $($case|$case_t) +);

    };
}

gen_types!(BufType Type, F64|f64 F32|f32);
