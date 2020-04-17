use serde::{Serialize,Deserialize};
use ocl::prm::*;
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

#[derive(Serialize,Deserialize)]
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

pub enum BufferConstructor<T: Type, V: VecType> {
    Len(T,usize), // Len(repeated value, len)
    Data(V),
}

#[derive(Serialize,Deserialize)]
pub enum SBufferConstructor {
    Len(Box<dyn Type>,usize), // Len(repeated value, len)
    Data(Box<dyn VecType>),
}

impl<T: 'static+Type, V: 'static+VecType> From<BufferConstructor<T,V>> for SBufferConstructor {
    fn from(n: BufferConstructor<T,V>) -> Self {
        match n {
            BufferConstructor::Len(t,s) => SBufferConstructor::Len(Box::new(t),s),
            BufferConstructor::Data(v) => SBufferConstructor::Data(Box::new(v)),
        }
    }
}

#[derive(Clone)]
pub enum KernelConstructor<'a> { //TODO use one SC for each &'a str
    Param(&'a str, &'a dyn EmptyType),
    Buffer(&'a str, &'a dyn EmptyType),
    ConstBuffer(&'a str, &'a dyn EmptyType),
}

#[derive(Serialize,Deserialize)]
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

#[derive(Clone)]
pub enum FunctionConstructor<'a> { //TODO use one SC for each &'a str
    Param(&'a str, &'a dyn EmptyType),
    Ptr(&'a str, &'a dyn EmptyType),
    GlobalPtr(&'a str, &'a dyn EmptyType),
    ConstPtr(&'a str, &'a dyn EmptyType),
}

#[derive(Serialize,Deserialize)]
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

macro_rules! impl_names {
    ($type:ident $type_opencl:literal) => {
        fn type_name(&self) -> &'static str {
            std::any::type_name::<$type>()
        }

        fn type_name_ocl(&self) -> &'static str {
            $type_opencl
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
    ($types:ident|$trait:ident $empty_types:ident|$empty_trait:ident $buffer_types:ident|$buffer_trait:ident $vec_types:ident|$vec_trait:ident, $($type:ident|$type_rust:ty|$type_ocl:ident|$type_opencl:literal) +) => {

        pub mod $empty_types {
            use super::*;

            #[typetag::serde]
            pub trait $empty_trait {
                fn to_box(&self) -> Box<dyn $empty_trait>;
                fn type_name(&self) -> &'static str;
                fn type_name_ocl(&self) -> &'static str;
                fn default_param(&self, kernel: &mut ocl::builders::KernelBuilder);
                fn default_buffer(&self, kernel: &mut ocl::builders::KernelBuilder);
            }
            gen_traits_to_box!($empty_trait);
            $(
                #[derive(Debug,Clone,Copy,Serialize,Deserialize)]
                pub struct $type;
                #[typetag::serde]
                impl $empty_trait for $type {
                    impl_traits_to_box!($empty_trait);
                    impl_names!($type $type_opencl);
                    fn default_param(&self, kernel: &mut ocl::builders::KernelBuilder) {
                        kernel.arg($type_ocl::default());
                    }
                    fn default_buffer(&self, kernel: &mut ocl::builders::KernelBuilder) {
                        kernel.arg(None::<&ocl::Buffer<$type_ocl>>);
                    }
                }
            )+
        }

        pub mod $types {
            use super::*;
            use std::any::Any;

            #[typetag::serde]
            pub trait $trait {
                fn to_box(&self) -> Box<dyn $trait>;
                fn type_name(&self) -> &'static str;
                fn type_name_ocl(&self) -> &'static str;
                fn set_arg(&self, kernel: &(ocl::Kernel,std::collections::BTreeMap<String,u32>), kernel_name: &str, param: &str) -> crate::Result<()>;
                fn gen_buffer(&self, pq: &ocl::ProQue, len: usize) -> crate::Result<Box<dyn $buffer_trait>>;
                fn iner_any(&self) -> &dyn Any;
                fn iner_box_any(self) -> Box<dyn Any>;
            }
            gen_traits_to_box!($trait);
            $(
                #[derive(Debug,Clone,Copy,Serialize,Deserialize)]
                #[repr(C)]
                pub struct $type(pub $type_rust);
                #[typetag::serde]
                impl $trait for $type {
                    impl_traits_to_box!($trait);
                    impl_names!($type $type_opencl);
                    fn set_arg(&self, kernel: &(ocl::Kernel,std::collections::BTreeMap<String,u32>), kernel_name: &str, param: &str) -> crate::Result<()> {
                        kernel.0.set_arg(*kernel.1.get(param).expect(&format!("Param \"{}\" not present in kernel \"{}\"",param,kernel_name)),unsafe {std::mem::transmute::<_,$type_ocl>(self.0)})
                    }
                    fn gen_buffer(&self, pq: &ocl::ProQue, len: usize) -> crate::Result<Box<dyn $buffer_trait>> {
                        Ok(Box::new($buffer_types::$type::from(pq.buffer_builder()
                        .len(len)
                        .fill_val(unsafe {std::mem::transmute::<_,$type_ocl>(self.0)})
                        .build()?)))
                    }
                    fn iner_any(&self) -> &dyn Any {
                        &self.0
                    }
                    fn iner_box_any(self) -> Box<dyn Any> {
                        Box::new(self.0)
                    }
                }
                impl From<$type_rust> for $type {
                    fn from(v: $type_rust) -> Self {
                        $type(v)
                    }
                }
                impl std::ops::Deref for $type {
                    type Target = $type_rust;

                    fn deref(&self) -> &Self::Target {
                        &self.0
                    }
                }
                impl From<$type_rust> for Box<dyn $trait> {
                    fn from(v: $type_rust) -> Self {
                        Box::new($type::from(v))
                    }
                }
            )+
        }

        pub mod $buffer_types {
            use super::*;
            use std::any::{Any,type_name};

            pub trait $buffer_trait {
                fn type_name(&self) -> &'static str;
                fn type_name_ocl(&self) -> &'static str;
                fn set_arg(&self, kernel: &(ocl::Kernel,std::collections::BTreeMap<String,u32>), kernel_name: &str, param: &str) -> crate::Result<()>;
                fn copy(&self, to: &dyn $buffer_trait) -> crate::Result<()>;
                fn get(&self) -> crate::Result<Box<dyn Any>>;
                fn get_first(&self) -> crate::Result<Box<dyn Any>>;
                fn iner_any(&self) -> &dyn Any;
            }
            $(
                pub struct $type(pub ocl::Buffer<$type_ocl>);
                impl $buffer_trait for $type {
                    impl_names!($type $type_opencl);
                    fn set_arg(&self, kernel: &(ocl::Kernel,std::collections::BTreeMap<String,u32>), kernel_name: &str, param: &str) -> crate::Result<()> {
                        kernel.0.set_arg(*kernel.1.get(param).expect(&format!("Param \"{}\" not present in kernel \"{}\"",param,kernel_name)),&self.0)
                    }
                    fn copy(&self, to: &dyn $buffer_trait) -> crate::Result<()>{
                        let to = to.iner_any().downcast_ref::<ocl::Buffer<$type_ocl>>()
                            .expect(&format!("Wrong type for buffer, expected {}, found {}",type_name::<ocl::Buffer<$type_ocl>>(),to.type_name()));
                        self.0.copy(to,None,None).enq()
                    }

                    fn get(&self) -> crate::Result<Box<dyn Any>> {
                        let buf = &self.0;
                        let len = buf.len();
                        let mut vec = Vec::with_capacity(len);
                        unsafe { vec.set_len(len); }
                        buf.read(&mut vec).enq()?;
                        Ok(Box::new(unsafe{
                            std::mem::transmute::<_,$vec_types::$type>(vec)
                        }))
                    }

                    fn get_first(&self) -> crate::Result<Box<dyn Any>> {
                        let buf = &self.0;
                        let mut val = vec![Default::default()];
                        buf.read(&mut val).enq()?;
                        Ok(Box::new(unsafe{
                            std::mem::transmute::<_,$types::$type>(val[0])
                        }))
                    }
                    fn iner_any(&self) -> &dyn Any {
                        &self.0
                    }
                }
                impl From<ocl::Buffer<$type_ocl>> for $type {
                    fn from(b: ocl::Buffer<$type_ocl>) -> Self {
                        $type(b)
                    }
                }
            )+
        }

        pub mod $vec_types {
            use super::*;
            use std::any::Any;

            #[typetag::serde]
            pub trait $vec_trait {
                fn type_name(&self) -> &'static str;
                fn type_name_ocl(&self) -> &'static str;
                fn to_box(&self) -> Box<dyn $vec_trait>;
                fn gen_buffer(&self, pq: &ocl::ProQue) -> crate::Result<Box<dyn $buffer_trait>>;
                fn iner_any(&self) -> &dyn Any;
                fn iner_box_any(self) -> Box<dyn Any>;
            }
            gen_traits_to_box!($vec_trait);
            $(
                #[derive(Debug,Clone,Serialize,Deserialize)]
                pub struct $type(pub Vec<$type_rust>);
                #[typetag::serde]
                impl $vec_trait for $type {
                    impl_traits_to_box!($vec_trait);
                    impl_names!($type $type_opencl);
                    fn gen_buffer(&self, pq: &ocl::ProQue) -> crate::Result<Box<dyn $buffer_trait>> {
                        Ok(Box::new($buffer_types::$type::from(pq.buffer_builder()
                       .len(self.0.len())
                       .copy_host_slice(unsafe {std::mem::transmute::<_,&[$type_ocl]>(&self.0[..])})
                       .build()?)))
                    }
                    fn iner_any(&self) -> &dyn Any {
                        &self.0
                    }
                    fn iner_box_any(self) -> Box<dyn Any> {
                        Box::new(self.0)
                    }
                }
                impl From<Vec<$type_rust>> for $type {
                    fn from(v: Vec<$type_rust>) -> Self {
                        $type(v)
                    }
                }
                impl std::ops::Deref for $type {
                    type Target = Vec<$type_rust>;

                    fn deref(&self) -> &Self::Target {
                        &self.0
                    }
                }
                impl From<Vec<$type_rust>> for Box<dyn $vec_trait> {
                    fn from(v: Vec<$type_rust>) -> Self {
                        Box::new($type::from(v))
                    }
                }
            )+
        }

    };
}


gen_types!(types|Type empty_types|EmptyType buffer_types|BufferType vec_types|VecType,
    F64  | f64    |f64    |"double"
    F32  | f32    |f32    |"float"
    U64  | u64    |u64    |"ulong"
    I64  | i64    |i64    |"long"
    U32  | u32    |u32    |"uint"
    I32  | i32    |i32    |"int"
    U16  | u16    |u16    |"ushort"
    I16  | i16    |i16    |"short"
    U8   | u8     |u8     |"uchar"
    I8   | i8     |i8     |"char"
    F64_2|[f64; 2]|Double2|"double2"
    F32_2|[f32; 2]|Float2 |"float2"
    U64_2|[u64; 2]|Ulong2 |"ulong2"
    I64_2|[i64; 2]|Long2  |"long2"
    U32_2|[u32; 2]|Uint2  |"uint2"
    I32_2|[i32; 2]|Int2   |"int2"
    U16_2|[u16; 2]|Ushort2|"ushort2"
    I16_2|[i16; 2]|Short2 |"short2"
    U8_2 |[u8 ; 2]|Uchar2 |"uchar2"
    I8_2 |[i8 ; 2]|Char2  |"char2"
    F64_4|[f64; 4]|Double4|"double4"
    F32_4|[f32; 4]|Float4 |"float4"
    U64_4|[u64; 4]|Ulong4 |"ulong4"
    I64_4|[i64; 4]|Long4  |"long4"
    U32_4|[u32; 4]|Uint4  |"uint4"
    I32_4|[i32; 4]|Int4   |"int4"
    U16_4|[u16; 4]|Ushort4|"ushort4"
    I16_4|[i16; 4]|Short4 |"short4"
    U8_4 |[u8 ; 4]|Uchar4 |"uchar4"
    I8_4 |[i8 ; 4]|Char4  |"char4"
);
