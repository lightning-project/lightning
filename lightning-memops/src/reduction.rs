use lightning_codegen::CPP_NAMESPACE;
use lightning_core::{DataType, DataValue, PrimitiveType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ReductionFunction {
    Max,
    Min,
    Sum,
    Product,
    And,
    Or,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Reduction {
    fun: ReductionFunction,
    dtype: DataType,
}

impl Reduction {
    pub fn new(fun: ReductionFunction, dtype: DataType) -> Option<Self> {
        use PrimitiveType::*;
        use ReductionFunction::*;

        let supported = match dtype.to_primitive() {
            Some(U8 | U16 | U32 | U64 | I8 | I16 | I32 | I64) => true,
            Some(F32 | F64) => match fun {
                Min | Max | Sum | Product => true,
                _ => false,
            },
            _ => false,
        };

        if supported {
            Some(Self { fun, dtype })
        } else {
            None
        }
    }

    pub fn function(&self) -> ReductionFunction {
        self.fun
    }

    pub fn data_type(&self) -> DataType {
        self.dtype
    }

    pub fn identity(&self) -> DataValue {
        use PrimitiveType::*;
        use ReductionFunction::*;

        macro_rules! for_dtype {
            ($typ:ty) => {{
                for_dtype!($typ, <$typ>::MIN, <$typ>::MAX)
            }};
            ($typ:ty, $min:expr, $max:expr) => {{
                let value: $typ = match &self.fun {
                    Sum | And => 0 as $typ,
                    Or | Product => 1 as $typ,
                    Max => $min,
                    Min => $max,
                };

                DataValue::from(value)
            }};
        }

        match self.dtype.to_primitive().unwrap() {
            U8 => for_dtype!(u8),
            U16 => for_dtype!(u16),
            U32 => for_dtype!(u32),
            U64 => for_dtype!(u64),

            I8 => for_dtype!(i8),
            I16 => for_dtype!(i16),
            I32 => for_dtype!(i32),
            I64 => for_dtype!(i64),

            F32 => for_dtype!(f32, f32::NEG_INFINITY, f32::INFINITY),
            F64 => for_dtype!(f64, f64::NEG_INFINITY, f64::INFINITY),

            _ => unreachable!(),
        }
    }

    pub(crate) fn csource_identity_literal(&self) -> String {
        use PrimitiveType::*;
        use ReductionFunction::*;

        macro_rules! for_dtype {
            ($typ:ty) => {{
                for_dtype!($typ, <$typ>::MIN, <$typ>::MAX)
            }};
            ($typ:ty, $min:expr, $max:expr) => {{
                match &self.fun {
                    Sum | And => "0".to_string(),
                    Or | Product => "1".to_string(),
                    Max => $min.to_string(),
                    Min => $max.to_string(),
                }
            }};
        }

        match self.dtype.to_primitive().unwrap() {
            U8 => for_dtype!(u8),
            U16 => for_dtype!(u16),
            U32 => for_dtype!(u32),
            U64 => for_dtype!(u64),

            I8 => for_dtype!(i8),
            I16 => for_dtype!(i16),
            I32 => for_dtype!(i32),
            I64 => for_dtype!(i64),

            F32 => for_dtype!(f32, "-INFINITY", "INFINITY"), // defined in
            F64 => for_dtype!(f64, "-INFINITY", "INFINITY"),

            _ => unreachable!(),
        }
    }

    pub(crate) fn csource_function_name(&self) -> String {
        use ReductionFunction::*;
        let fun = match &self.fun {
            Max => "max",
            Min => "min",
            Sum => "sum",
            Product => "product",
            And => "bit_and",
            Or => "bit_or",
        };

        format!("{}::reductions::{}", CPP_NAMESPACE, fun)
    }
}
