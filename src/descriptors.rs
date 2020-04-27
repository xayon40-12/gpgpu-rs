use serde::{Serialize,Deserialize};
use ocl::prm::*;

#[derive(Debug,Clone)]
pub enum KernelArg<'a> { //TODO use one SC for each &'a str
    Param(&'a str, Types),
    Buffer(&'a str),
    BufArg(&'a str,&'a str), // BufArg(mem buf, kernel buf)
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub enum SKernelArg {
    Param(String, Types),
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

#[derive(Clone,Debug,Serialize,Deserialize)]
pub enum BufferConstructor {
    Len(Types,usize), // Len(repeated value, len)
    Data(VecTypes),
}

#[derive(Debug,Clone)]
pub enum KernelConstructor<'a> { //TODO use one SC for each &'a str
    KCParam(&'a str, ConstructorTypes),
    KCBuffer(&'a str, ConstructorTypes),
    KCConstBuffer(&'a str, ConstructorTypes),
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub enum SKernelConstructor {
    KCParam(String, ConstructorTypes),
    KCBuffer(String, ConstructorTypes),
    KCConstBuffer(String, ConstructorTypes),
}

impl<'a> From<&KernelConstructor<'a>> for SKernelConstructor {
    fn from(n: &KernelConstructor<'a>) -> Self {
        match n {
            KernelConstructor::KCParam(s,e) => SKernelConstructor::KCParam((*s).into(),*e),
            KernelConstructor::KCBuffer(s,e) => SKernelConstructor::KCBuffer((*s).into(),*e),
            KernelConstructor::KCConstBuffer(s,e) => SKernelConstructor::KCConstBuffer((*s).into(),*e),
        }
    }
}

#[derive(Debug,Clone)]
pub enum FunctionConstructor<'a> { //TODO use one SC for each &'a str
    FCParam(&'a str, ConstructorTypes),
    FCPtr(&'a str, ConstructorTypes),
    FCGlobalPtr(&'a str, ConstructorTypes),
    FCConstPtr(&'a str, ConstructorTypes),
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub enum SFunctionConstructor {
    FCParam(String, ConstructorTypes),
    FCPtr(String, ConstructorTypes),
    FCGlobalPtr(String, ConstructorTypes),
    FCConstPtr(String, ConstructorTypes),
}

impl<'a> From<&FunctionConstructor<'a>> for SFunctionConstructor {
    fn from(n: &FunctionConstructor<'a>) -> Self {
        match n {
            FunctionConstructor::FCParam(s,e) => SFunctionConstructor::FCParam((*s).into(),*e),
            FunctionConstructor::FCPtr(s,e) => SFunctionConstructor::FCPtr((*s).into(),*e),
            FunctionConstructor::FCGlobalPtr(s,e) => SFunctionConstructor::FCGlobalPtr((*s).into(),*e),
            FunctionConstructor::FCConstPtr(s,e) => SFunctionConstructor::FCConstPtr((*s).into(),*e),
        }
    }
}

macro_rules! impl_names_empty {
    ($($type:ident|$type_rust:ty|$type_opencl:literal) +) => {
        pub fn type_name(&self) -> &'static str {
            match self {
                $(Self::$type => std::any::type_name::<$type_rust>(),)+
            }
        }
        pub fn type_name_variant(&self) -> &'static str {
            match self {
                $(Self::$type => stringify!($type),)+
            }
        }
        pub fn type_name_ocl(&self) -> &'static str {
            match self {
                $(Self::$type => $type_opencl,)+
            }
        }
    };
}
macro_rules! impl_names {
    ($($type:ident|$type_rust:ty|$type_opencl:literal) +) => {
        pub fn type_name(&self) -> &'static str {
            match self {
                $(Self::$type(..) => std::any::type_name::<$type_rust>(),)+
            }
        }
        pub fn type_name_variant(&self) -> &'static str {
            match self {
                $(Self::$type(..) => stringify!($type),)+
            }
        }
        pub fn type_name_ocl(&self) -> &'static str {
            match self {
                $(Self::$type(..) => $type_opencl,)+
            }
        }
    };
}

macro_rules! gen_types {
    ($Types:ident $ConstructorTypes:ident $BufferTypes:ident $VecTypes:ident, $($type:ident $type_ref:ident $ctype:ident $ctype_ref:ident $btype:ident $btype_ref:ident $vtype:ident $vtype_ref:ident|$type_rust:ty|$len:literal|$type_ocl:ident|$type_opencl:literal) +) => {
        #[derive(Debug,Clone,Copy,Serialize,Deserialize)]
        pub enum $ConstructorTypes {
            $($ctype,)+
        }
        impl $ConstructorTypes {
            impl_names_empty!($($ctype|$type_rust|$type_opencl)+);
            pub fn default_param(&self, kernel: &mut ocl::builders::KernelBuilder) {
                match self {
                    $(Self::$ctype => kernel.arg($type_ocl::default()),)+
                };
            }
            pub fn default_buffer(&self, kernel: &mut ocl::builders::KernelBuilder) {
                match self {
                    $(Self::$ctype => kernel.arg(None::<&ocl::Buffer<$type_ocl>>),)+
                };
            }
        }

      
        #[derive(Debug,Clone,Copy,Serialize,Deserialize)]
        #[repr(C)]
        pub enum $Types {
            $($type($type_rust),)+
        }

        impl $Types {
            impl_names!($($type|$type_rust|$type_opencl)+);
            pub fn set_arg(&self, kernel: &(ocl::Kernel,std::collections::BTreeMap<String,u32>), kernel_name: &str, param: &str) -> crate::Result<()> {
                match self {
                    $(Self::$type(v) => kernel.0.set_arg(*kernel.1.get(param).expect(&format!("Param \"{}\" not present in kernel \"{}\"",param,kernel_name)),unsafe {std::mem::transmute::<_,$type_ocl>(*v)}),)+
                }
            }
            pub fn gen_buffer(&self, pq: &ocl::ProQue, len: usize) -> crate::Result<$BufferTypes> {
                match self {
                    $(Self::$type(v) => 
                        Ok($BufferTypes::$btype(pq.buffer_builder()
                        .len(len)
                        .fill_val(unsafe {std::mem::transmute::<_,$type_ocl>(*v)})
                        .build()?)),)+
                }
            }

            pub fn to_string(&self) -> String {
                match self {
                    $(Self::$type(t) => format!("{:?}", t),)+
                }
            }
            
            pub fn len(&self) -> usize {
                match self {
                    $(Self::$type(..) => $len,)+
                }
            }

            $(#[allow(non_snake_case)] pub fn $type(self) -> $type_rust {
                if let Self::$type(v) = self { v } else { panic!("Wrong variant for {}, expected {}, found {}", stringify!($Types),stringify!($type),self.type_name_variant()) } //TODO better error description
            })+

            $(#[allow(non_snake_case)] pub fn $type_ref(&self) -> &$type_rust {
                if let Self::$type(v) = self { v } else { panic!("Wrong variant for {}, expected {}, found {}", stringify!($Types),stringify!($type),self.type_name_variant()) } //TODO better error description
            })+
        }

        $(impl From<$type_rust> for $Types {
            fn from(v: $type_rust) -> Self {
                Self::$type(v)
            }
        })+

  
        pub enum $BufferTypes {
            $($btype(ocl::Buffer<$type_ocl>),)+
        }
        impl $BufferTypes {
            impl_names!($($btype|$type_rust|$type_opencl)+);
            pub fn set_arg(&self, kernel: &(ocl::Kernel,std::collections::BTreeMap<String,u32>), kernel_name: &str, param: &str) -> crate::Result<()> {
                match self {
                    $(Self::$btype(v) => kernel.0.set_arg(*kernel.1.get(param).expect(&format!("Param \"{}\" not present in kernel \"{}\"",param,kernel_name)),v),)+
                }
            }
            pub fn copy(&self, to: &$BufferTypes) -> crate::Result<()>{
                match self {
                    $(Self::$btype(v) => v.copy(to.$btype_ref(),None,None).enq(),)+
                }
            }

            pub fn get(&self) -> crate::Result<$VecTypes> {
                match self {
                    $(Self::$btype(v) => {
                        let buf = v;
                        let len = buf.len();
                        let mut vec = Vec::with_capacity(len);
                        unsafe { vec.set_len(len); }
                        buf.read(&mut vec).enq()?;
                        Ok($VecTypes::$vtype(unsafe{
                            std::mem::transmute::<_,Vec<$type_rust>>(vec)
                        }))
                    },)+
                }
            }

            pub fn get_first(&self) -> crate::Result<$Types> {
                match self {
                    $(Self::$btype(v) => {
                        let buf = v;
                        let mut val = vec![Default::default()];
                        buf.read(&mut val).enq()?;
                        Ok($Types::$type(unsafe{
                            std::mem::transmute::<_,$type_rust>(val[0])
                        }))
                    },)+
                }
            }

            $(#[allow(non_snake_case)] pub fn $btype(self) -> ocl::Buffer<$type_ocl> {
                if let Self::$btype(v) = self { v } else { panic!("Wrong variant for {}, expected {}, found {}", stringify!($BufferTypes),stringify!($btype),self.type_name_variant()) } //TODO better error description
            })+

            $(#[allow(non_snake_case)] pub fn $btype_ref(&self) -> &ocl::Buffer<$type_ocl> {
                if let Self::$btype(v) = self { v } else { panic!("Wrong variant for {}, expected {}, found {}", stringify!($BufferTypes),stringify!($btype),self.type_name_variant()) } //TODO better error description
            })+
        }

        #[derive(Debug,Clone,Serialize,Deserialize)]
        #[repr(C)]
        pub enum $VecTypes {
            $($vtype(Vec<$type_rust>),)+
        }
        impl $VecTypes {
            impl_names!($($vtype|$type_rust|$type_opencl)+);
            pub fn gen_buffer(&self, pq: &ocl::ProQue) -> crate::Result<$BufferTypes> {
                match self {
                    $(Self::$vtype(v) => 
                        Ok($BufferTypes::$btype(pq.buffer_builder()
                       .len(v.len())
                       .copy_host_slice(unsafe {std::mem::transmute::<_,&[$type_ocl]>(&v[..])})
                       .build()?)),)+
                }
            }

            $(#[allow(non_snake_case)] pub fn $vtype(self) -> Vec<$type_rust> {
                if let Self::$vtype(v) = self { v } else { panic!("Wrong variant for {}, expected {}, found {}", stringify!($VecTypes),stringify!($vtype),self.type_name_variant()) } //TODO better error description
            })+

            $(#[allow(non_snake_case)] pub fn $vtype_ref(&self) -> &Vec<$type_rust> {
                if let Self::$vtype(v) = self { v } else { panic!("Wrong variant for {}, expected {}, found {}", stringify!($VecTypes),stringify!($vtype),self.type_name_variant()) } //TODO better error description
            })+
        }

        $(impl From<Vec<$type_rust>> for $VecTypes {
            fn from(v: Vec<$type_rust>) -> Self {
                Self::$vtype(v)
            }
        })+
    };
}


gen_types!(Types ConstructorTypes BufferTypes VecTypes,
    F64   F64_ref   CF64   CF64_ref   BF64   BF64_ref   VF64   VF64_ref  | f64    |1|f64    |"double"
    F32   F32_ref   CF32   CF32_ref   BF32   BF32_ref   VF32   VF32_ref  | f32    |1|f32    |"float"
    U64   U64_ref   CU64   CU64_ref   BU64   BU64_ref   VU64   VU64_ref  | u64    |1|u64    |"ulong"
    I64   I64_ref   CI64   CI64_ref   BI64   BI64_ref   VI64   VI64_ref  | i64    |1|i64    |"long"
    U32   U32_ref   CU32   CU32_ref   BU32   BU32_ref   VU32   VU32_ref  | u32    |1|u32    |"uint"
    I32   I32_ref   CI32   CI32_ref   BI32   BI32_ref   VI32   VI32_ref  | i32    |1|i32    |"int"
    U16   U16_ref   CU16   CU16_ref   BU16   BU16_ref   VU16   VU16_ref  | u16    |1|u16    |"ushort"
    I16   I16_ref   CI16   CI16_ref   BI16   BI16_ref   VI16   VI16_ref  | i16    |1|i16    |"short"
    U8    U8_ref    CU8    CU8_ref    BU8    BU8_ref    VU8    VU8_ref   | u8     |1|u8     |"uchar"
    I8    I8_ref    CI8    CI8_ref    BI8    BI8_ref    VI8    VI8_ref   | i8     |1|i8     |"char"
    F64_2 F64_2_ref CF64_2 CF64_2_ref BF64_2 BF64_2_ref VF64_2 VF64_2_ref|[f64; 2]|2|Double2|"double2"
    F32_2 F32_2_ref CF32_2 CF32_2_ref BF32_2 BF32_2_ref VF32_2 VF32_2_ref|[f32; 2]|2|Float2 |"float2"
    U64_2 U64_2_ref CU64_2 CU64_2_ref BU64_2 BU64_2_ref VU64_2 VU64_2_ref|[u64; 2]|2|Ulong2 |"ulong2"
    I64_2 I64_2_ref CI64_2 CI64_2_ref BI64_2 BI64_2_ref VI64_2 VI64_2_ref|[i64; 2]|2|Long2  |"long2"
    U32_2 U32_2_ref CU32_2 CU32_2_ref BU32_2 BU32_2_ref VU32_2 VU32_2_ref|[u32; 2]|2|Uint2  |"uint2"
    I32_2 I32_2_ref CI32_2 CI32_2_ref BI32_2 BI32_2_ref VI32_2 VI32_2_ref|[i32; 2]|2|Int2   |"int2"
    U16_2 U16_2_ref CU16_2 CU16_2_ref BU16_2 BU16_2_ref VU16_2 VU16_2_ref|[u16; 2]|2|Ushort2|"ushort2"
    I16_2 I16_2_ref CI16_2 CI16_2_ref BI16_2 BI16_2_ref VI16_2 VI16_2_ref|[i16; 2]|2|Short2 |"short2"
    U8_2  U8_2_ref  CU8_2  CU8_2_ref  BU8_2  BU8_2_ref  VU8_2  VU8_2_ref |[u8 ; 2]|2|Uchar2 |"uchar2"
    I8_2  I8_2_ref  CI8_2  CI8_2_ref  BI8_2  BI8_2_ref  VI8_2  VI8_2_ref |[i8 ; 2]|2|Char2  |"char2"
    F64_4 F64_4_ref CF64_4 CF64_4_ref BF64_4 BF64_4_ref VF64_4 VF64_4_ref|[f64; 4]|4|Double4|"double4"
    F32_4 F32_4_ref CF32_4 CF32_4_ref BF32_4 BF32_4_ref VF32_4 VF32_4_ref|[f32; 4]|4|Float4 |"float4"
    U64_4 U64_4_ref CU64_4 CU64_4_ref BU64_4 BU64_4_ref VU64_4 VU64_4_ref|[u64; 4]|4|Ulong4 |"ulong4"
    I64_4 I64_4_ref CI64_4 CI64_4_ref BI64_4 BI64_4_ref VI64_4 VI64_4_ref|[i64; 4]|4|Long4  |"long4"
    U32_4 U32_4_ref CU32_4 CU32_4_ref BU32_4 BU32_4_ref VU32_4 VU32_4_ref|[u32; 4]|4|Uint4  |"uint4"
    I32_4 I32_4_ref CI32_4 CI32_4_ref BI32_4 BI32_4_ref VI32_4 VI32_4_ref|[i32; 4]|4|Int4   |"int4"
    U16_4 U16_4_ref CU16_4 CU16_4_ref BU16_4 BU16_4_ref VU16_4 VU16_4_ref|[u16; 4]|4|Ushort4|"ushort4"
    I16_4 I16_4_ref CI16_4 CI16_4_ref BI16_4 BI16_4_ref VI16_4 VI16_4_ref|[i16; 4]|4|Short4 |"short4"
    U8_4  U8_4_ref  CU8_4  CU8_4_ref  BU8_4  BU8_4_ref  VU8_4  VU8_4_ref |[u8 ; 4]|4|Uchar4 |"uchar4"
    I8_4  I8_4_ref  CI8_4  CI8_4_ref  BI8_4  BI8_4_ref  VI8_4  VI8_4_ref |[i8 ; 4]|4|Char4  |"char4"
);
