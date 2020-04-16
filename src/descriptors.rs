use serde::{Serialize,Deserialize};
pub use types::Type;
pub use empty_types::EmptyType;
pub use buffer_types::BufferType;
pub use vec_types::VecType;

#[derive(Clone)]
pub enum KernelArg<'a> { //TODO use one SC for each &'a str
    Param(&'a str, &'a dyn Type),
    Buffer(&'a str),
    BufArg(&'a str,&'a str), // BufArg(mem buf, kernel buf)
}

#[derive(Clone)]
pub enum SKernelArg {
    Param(String, Box<dyn Type>),
    Buffer(String),
    BufArg(String,String), // BufArg(mem buf, kernel buf)
}

impl<'a> From<&KernelArg<'a>> for SKernelArg {
    fn from(n: &KernelArg<'a>) -> Self {
        match n {
            KernelArg::Param(s,e) => SKernelArg::Param((*s).into(),(*e).into()),
            KernelArg::Buffer(s) => SKernelArg::Buffer((*s).into()),
            KernelArg::BufArg(s,ss) => SKernelArg::BufArg((*s).into(),(*ss).into()),
        }
    }
}

#[derive(Clone)]
pub enum BufferConstructor<T: Type, V: VecType> {
    Len(T,usize), // Len(repeated value, len)
    Data(V),
}

pub enum SBufferConstructor {
    Len(Box<dyn Type>,usize), // Len(repeated value, len)
    Data(Box<dyn VecType>),
}

impl<T: Type, V: VecType> From<&BufferConstructor<T,V>> for SBufferConstructor {
    fn from(n: &BufferConstructor<T,V>) -> Self {
        match n {
            BufferConstructor::Len(t,s) => SBufferConstructor::Len(t.into(),s),
            BufferConstructor::Data(v) => SBufferConstructor::Data(v.into()),
        }
    }
}

#[derive(Clone,Serialize,Deserialize)]
pub enum KernelConstructor<'a> { //TODO use one SC for each &'a str
    Param(&'a str, &'a dyn EmptyType),
    Buffer(&'a str, &'a dyn EmptyType),
    ConstBuffer(&'a str, &'a dyn EmptyType),
}

#[derive(Clone,Serialize,Deserialize)]
pub enum SKernelConstructor {
    Param(String, Box<dyn EmptyType>),
    Buffer(String, Box<dyn EmptyType>),
    ConstBuffer(String, Box<dyn EmptyType>),
}

impl<'a> From<&KernelConstructor<'a>> for SKernelConstructor {
    fn from(n: &KernelConstructor<'a>) -> Self {
        match n {
            KernelConstructor::Param(s,e) => SKernelConstructor::Param((*s).into(),(*e).into()),
            KernelConstructor::Buffer(s,e) => SKernelConstructor::Buffer((*s).into(),(*e).into()),
            KernelConstructor::ConstBuffer(s,e) => SKernelConstructor::ConstBuffer((*s).into(),(*e).into()),
        }
    }
}

#[derive(Clone,Serialize,Deserialize)]
pub enum FunctionConstructor<'a> { //TODO use one SC for each &'a str
    Param(&'a str, &'a dyn EmptyType),
    Ptr(&'a str, &'a dyn EmptyType),
    GlobalPtr(&'a str, &'a dyn EmptyType),
    ConstPtr(&'a str, &'a dyn EmptyType),
}

#[derive(Clone,Serialize,Deserialize)]
pub enum SFunctionConstructor {
    Param(String, Box<dyn EmptyType>),
    Ptr(String, Box<dyn EmptyType>),
    GlobalPtr(String, Box<dyn EmptyType>),
    ConstPtr(String, Box<dyn EmptyType>),
}

impl<'a> From<&FunctionConstructor<'a>> for SFunctionConstructor {
    fn from(n: &FunctionConstructor<'a>) -> Self {
        match n {
            FunctionConstructor::Param(s,e) => SFunctionConstructor::Param((*s).into(),(*e).into()),
            FunctionConstructor::Ptr(s,e) => SFunctionConstructor::Ptr((*s).into(),(*e).into()),
            FunctionConstructor::GlobalPtr(s,e) => SFunctionConstructor::GlobalPtr((*s).into(),(*e).into()),
            FunctionConstructor::ConstPtr(s,e) => SFunctionConstructor::ConstPtr((*s).into(),(*e).into()),
        }
    }
}

macro_rules! impl_types {
    ($type:ident, $type_opencl:literal) => {
        impl $type {
            pub fn type_name(&self) -> &str {
                std::any::type_name::<$type>()
            }

            pub fn type_name_ocl(&self) -> &str {
                $type_opencl
            }
        }
    };
}

//TODO to_buffer(len: usize) -> Buffer<Self>

macro_rules! gen_traits_to_box {
    ($name:ident) => {
        impl From<&dyn $name> for Box<dyn $name> {
            fn from(t: &dyn $name) -> Box<dyn $name> {
                t.to_box()
            }
        }
    };
}

macro_rules! impl_traits_to_box {
    ($trait:ident) => {
        fn to_box(&self) -> Box<dyn $trait> {
            Box::new(self.clone())
        }
    };
}


macro_rules! gen_types {
    ($types:ident|$trait:ident $empty_types:ident|$empty_trait:ident $buffer_types:ident|$buffer_trait:ident $vec_types:ident|$vec_trait:ident, $($type:ident|$type_rust:ident $type_dim:literal|$type_ocl:ident|$type_opencl:literal) +) => {
        pub mod $types {
            pub trait $trait {
                fn to_box(&self) -> Box<dyn $trait>;
            }
            gen_traits_to_box!($trait);
            $(
                #[derive(Debug,Clone,Copy)]
                #[repr(C)]
                pub struct $type([$type_rust;$type_dim]);
                impl $type {

                }
                impl_types!($type, $type_opencl);
                impl $trait for $type {
                    impl_traits_to_box!($trait);

                }
                impl From<[$type_rust;$type_dim]> for $type {
                    fn from(v: [$type_rust;$type_dim]) -> Self {
                        $type(v)
                    }
                }
            )+
        }

        pub mod $empty_types {
            pub trait $empty_trait {
                fn to_box(&self) -> Box<dyn $empty_trait>;
            }
            gen_traits_to_box!($empty_trait);
            $(
                #[derive(Debug,Clone,Copy)]
                pub struct $type;
                impl_types!($type, $type_opencl);
                impl $type {

                }
                impl $empty_trait for $type {

                }
            )+
        }

        pub mod $buffer_types {
            pub trait $buffer_trait {
                fn to_box(&self) -> Box<dyn $buffer_trait>;
            }
            //gen_traits_to_box!($buffer_trait);
            use ocl::prm::*;
            $(
                pub struct $type(ocl::Buffer<$type_ocl>);
                impl $type {
                    pub fn set_arg(&self, kernel: &(ocl::Kernel,std::collections::BTreeMap<String,u32>), kernel_name: &str, param: &str) {
                        kernel.0.set_arg(*kernel.1.get(param).expect(&format!("Param \"{}\" not present in kernel \"{}\"",param,kernel_name)),self.0);
                    }
                }
                impl $buffer_trait for $type {

                }
            )+
        }

        pub mod $vec_types {
            pub trait $vec_trait {
                fn to_box(&self) -> Box<dyn $vec_trait>;
            }
            gen_traits_to_box!($vec_trait);
            $(
                #[derive(Debug,Clone)]
                pub struct $type(Vec<super::$types::$type>);
                impl $type {

                }
                impl $vec_trait for $type {

                }
                impl From<Vec<super::$types::$type>> for $type {
                    fn from(v: Vec<super::$types::$type>) -> Self {
                        $type(v)
                    }
                }
            )+
        }



        //$kernel.arg($case_t::default())),+
        //$kernel.arg(None::<&ocl::Buffer<$case_t>>)),+

    };
}

gen_types!(types|Type empty_types|EmptyType buffer_types|BufferType vec_types|VecType,
    F64  |f64 1|f64    |"double"
    F32  |f32 1|f32    |"float"
    U64  |u64 1|u64    |"ulong"
    I64  |i64 1|i64    |"long"
    U32  |u32 1|u32    |"uint"
    I32  |i32 1|i32    |"int"
    U16  |u16 1|u16    |"ushort"
    I16  |i16 1|i16    |"short"
    U8   |u8  1|u8     |"uchar"
    I8   |i8  1|i8     |"char"
    F64_2|f64 2|Double2|"double2"
    F32_2|f32 2|Float2 |"float2"
    U64_2|u64 2|Ulong2 |"ulong2"
    I64_2|i64 2|Long2  |"long2"
    U32_2|u32 2|Uint2  |"uint2"
    I32_2|i32 2|Int2   |"int2"
    U16_2|u16 2|Ushort2|"ushort2"
    I16_2|i16 2|Short2 |"short2"
    U8_2 |u8  2|Uchar2 |"uchar2"
    I8_2 |i8  2|Char2  |"char2"
    F64_3|f64 3|Double3|"double3"
    F32_3|f32 3|Float3 |"float3"
    U64_3|u64 3|Ulong3 |"ulong3"
    I64_3|i64 3|Long3  |"long3"
    U32_3|u32 3|Uint3  |"uint3"
    I32_3|i32 3|Int3   |"int3"
    U16_3|u16 3|Ushort3|"ushort3"
    I16_3|i16 3|Short3 |"short3"
    U8_3 |u8  3|Uchar3 |"uchar3"
    I8_3 |i8  3|Char3  |"char3"
    F64_4|f64 4|Double4|"double4"
    F32_4|f32 4|Float4 |"float4"
    U64_4|u64 4|Ulong4 |"ulong4"
    I64_4|i64 4|Long4  |"long4"
    U32_4|u32 4|Uint4  |"uint4"
    I32_4|i32 4|Int4   |"int4"
    U16_4|u16 4|Ushort4|"ushort4"
    I16_4|i16 4|Short4 |"short4"
    U8_4 |u8  4|Uchar4 |"uchar4"
    I8_4 |i8  4|Char4  |"char4"
);
