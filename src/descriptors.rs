use serde::{Serialize,Deserialize};

#[derive(Clone)]
pub enum KernelArg<'a> { //TODO use one SC for each &'a str
    Param(&'a str,Type),
    Buffer(&'a str),
    BufArg(&'a str,&'a str), // BufArg(mem buf, kernel buf)
}

#[derive(Clone)]
pub enum SKernelArg {
    Param(String, Type),
    Buffer(String),
    BufArg(String,String), // BufArg(mem buf, kernel buf)
}

impl<'a> From<&KernelArg<'a>> for SKernelArg {
    fn from(n: &KernelArg<'a>) -> Self {
        match n {
            KernelArg::Param(s,e) => SKernelArg::Param((*s).into(),*e),
            KernelArg::Buffer(s) => SKernelArg::Buffer((*s).into()),
            KernelArg::BufArg(s,ss) => SKernelArg::BufArg((*s).into(),(*ss).into()),
        }
    }
}

#[derive(Clone)]
pub enum BufferConstructor {
    Len(Type,usize), // Len(repeated value, len)
    Data(VecType),
}

#[derive(Clone,Serialize,Deserialize)]
pub enum KernelConstructor<'a> { //TODO use one SC for each &'a str
    Param(&'a str, EmptyType),
    Buffer(&'a str, EmptyType),
    ConstBuffer(&'a str, EmptyType),
}

#[derive(Clone,Serialize,Deserialize)]
pub enum SKernelConstructor {
    Param(String, EmptyType),
    Buffer(String, EmptyType),
    ConstBuffer(String, EmptyType),}

impl<'a> From<&KernelConstructor<'a>> for SKernelConstructor {
    fn from(n: &KernelConstructor<'a>) -> Self {
        match n {
            KernelConstructor::Param(s,e) => SKernelConstructor::Param((*s).into(),*e),
            KernelConstructor::Buffer(s,e) => SKernelConstructor::Buffer((*s).into(),*e),
            KernelConstructor::ConstBuffer(s,e) => SKernelConstructor::ConstBuffer((*s).into(),*e),
        }
    }
}

#[derive(Clone,Serialize,Deserialize)]
pub enum FunctionConstructor<'a> { //TODO use one SC for each &'a str
    Param(&'a str, EmptyType),
    Ptr(&'a str, EmptyType),
    GlobalPtr(&'a str, EmptyType),
    ConstPtr(&'a str, EmptyType),
}

#[derive(Clone,Serialize,Deserialize)]
pub enum SFunctionConstructor {
    Param(String, EmptyType),
    Ptr(String, EmptyType),
    GlobalPtr(String, EmptyType),
    ConstPtr(String, EmptyType),
}

impl<'a> From<&FunctionConstructor<'a>> for SFunctionConstructor {
    fn from(n: &FunctionConstructor<'a>) -> Self {
        match n {
            FunctionConstructor::Param(s,e) => SFunctionConstructor::Param((*s).into(),*e),
            FunctionConstructor::Ptr(s,e) => SFunctionConstructor::Ptr((*s).into(),*e),
            FunctionConstructor::GlobalPtr(s,e) => SFunctionConstructor::GlobalPtr((*s).into(),*e),
            FunctionConstructor::ConstPtr(s,e) => SFunctionConstructor::ConstPtr((*s).into(),*e),
        }
    }
}

use std::any::{Any,type_name};
use ocl::prm::*;

macro_rules! impl_types {
    ($name:ident, $($case:ident|$case_t:ident|$case_ocl:literal) +) => {
        impl $name {
            pub fn iner_any(&self) -> &dyn Any {
                iner_each!(self,$name,buf,buf as &dyn Any)
            }

            pub fn iner<T: ocl::OclPrm>(&self) -> Option<&ocl::Buffer<T>> {
                self.iner_any().downcast_ref::<ocl::Buffer<T>>()
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
    ($namebuftype:ident $nametype:ident $namevectype:ident $nameemptytype:ident, $($case:ident|$case_t:ident|$case_ocl:literal) +) => {
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

        #[derive(Debug,Clone,Copy)]
        pub enum $nametype {
            $($case($case_t),)+
        }
        $(impl From<$case_t> for $nametype {
            fn from(t: $case_t) -> $nametype {
                $nametype::$case(t)
            }
        })+

        #[derive(Debug,Clone)]
        pub enum $namevectype {
            $($case(Vec<$case_t>),)+
        }
        $(impl From<Vec<$case_t>> for $namevectype {
            fn from(t: Vec<$case_t>) -> $namevectype {
                $namevectype::$case(t)
            }
        })+

        impl_types!($namebuftype, $($case|$case_t|$case_ocl) +);
        impl_types!($nametype, $($case|$case_t|$case_ocl) +);
        impl_types!($namevectype, $($case|$case_t|$case_ocl) +);


        #[derive(Debug,Clone,Copy,Serialize,Deserialize)]
        pub enum $nameemptytype {
            $($case,)+
        }
        impl $nameemptytype {
            pub fn type_name_ocl(&self) -> &str {
                match self {
                    $($nameemptytype::$case => $case_ocl,)+
                }
            }
        }
        macro_rules! each_default {
            (param, $match:ident, $kernel:ident) => {
                match $match {
                    $($nameemptytype::$case => $kernel.arg($case_t::default())),+
                }
            };
            (buffer, $match:ident, $kernel:expr) => {
                match $match {
                    $($nameemptytype::$case => $kernel.arg(None::<&ocl::Buffer<$case_t>>)),+
                }
            };
        }

    };
}


gen_types!(BufType Type VecType EmptyType,
    F64|f64|"double"
    F32|f32|"float"
    U64|u64|"ulong"
    I64|i64|"long"
    U32|u32|"uint"
    I32|i32|"int"
    U16|u16|"ushort"
    I16|i16|"short"
    U8 |u8 |"uchar"
    I8 |i8 |"char"
    F64_2|Double2|"double2"
    F32_2|Float2 |"float2"
    U64_2|Ulong2 |"ulong2"
    I64_2|Long2  |"long2"
    U32_2|Uint2  |"uint2"
    I32_2|Int2   |"int2"
    U16_2|Ushort2|"ushort2"
    I16_2|Short2 |"short2"
    U8_2 |Uchar2 |"uchar2"
    I8_2 |Char2  |"char2"
    F64_3|Double3|"double3"
    F32_3|Float3 |"float3"
    U64_3|Ulong3 |"ulong3"
    I64_3|Long3  |"long3"
    U32_3|Uint3  |"uint3"
    I32_3|Int3   |"int3"
    U16_3|Ushort3|"ushort3"
    I16_3|Short3 |"short3"
    U8_3 |Uchar3 |"uchar3"
    I8_3 |Char3  |"char3"
    F64_4|Double4|"double4"
    F32_4|Float4 |"float4"
    U64_4|Ulong4 |"ulong4"
    I64_4|Long4  |"long4"
    U32_4|Uint4  |"uint4"
    I32_4|Int4   |"int4"
    U16_4|Ushort4|"ushort4"
    I16_4|Short4 |"short4"
    U8_4 |Uchar4 |"uchar4"
    I8_4 |Char4  |"char4"
);
