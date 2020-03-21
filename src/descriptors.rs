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

macro_rules! impl_types {
    ($name:ident, $($case:ident|$case_t:ident|$case_ocl:literal) +) => {
        impl $name {
            pub fn iner_any(&self) -> &dyn Any {
                iner_each!(self,$name,buf,buf as &dyn Any)
            }

            pub fn type_name(&self) -> &str {
                match self {
                    $($name::$case(_) => type_name::<$case_t>(),)+
                }
            }

            pub fn type_name_ocl(&self) -> &str {
                match self {
                    $($name::$case(_) => $case_ocl,)+
                }
            }
        }
    };
}
macro_rules! gen_types {
    ($namebuftype:ident $nametype:ident $namevectype:ident, $($case:ident|$case_t:ident|$case_ocl:literal) +) => {
        macro_rules! iner_each {
            ($match:expr, $enum:ident, $var:pat, $todo:expr) => {
                match $match {
                    $($enum::$case($var) => $todo,)+
                }
            };
        }

        macro_rules! iner_each_gen {
            ($match:expr, $enum:ident $enumother:ident, $var:pat, $todo:expr) => {
                match $match {
                    $($enum::$case($var) => $enumother::$case($todo),)+
                }
            };
        }

        #[derive(Debug)]
        pub enum $namebuftype {
            $($case(ocl::Buffer<$case_t>),)+
        }

        #[derive(Debug,Clone)]
        pub enum $nametype {
            $($case($case_t),)+
        }

        #[derive(Debug)]
        pub enum $namevectype {
            $($case(Vec<$case_t>),)+
        }


        impl_types!($namebuftype, $($case|$case_t|$case_ocl) +);
        impl_types!($nametype, $($case|$case_t|$case_ocl) +);
        impl_types!($namevectype, $($case|$case_t|$case_ocl) +);

    };
}

gen_types!(BufType Type VecType,
    F64|f64|"double"
    F32|f32|"float"
    U64|u64|"unsigned long"
    I64|i64|"long"
    U32|u32|"unsigned int"
    I32|i32|"int"
);
