use super::compile::KernelCompiler;
use lightning_core::{DataType, DataValue, Dim3, MAX_DIMS};
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug};

#[derive(Serialize, Deserialize, Clone)]
pub struct ModuleDef {
    pub source: Vec<u8>,
    pub file_name: Option<String>,
    pub kernel: KernelDef,
    pub compiler: KernelCompiler,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct KernelDef {
    pub function_name: String,
    pub parameters: Vec<KernelParam>,
    pub bounds_checking: bool,
}

impl ModuleDef {
    pub fn new(function_name: String, source: Vec<u8>, parameters: Vec<KernelParam>) -> Self {
        Self {
            source,
            file_name: None,
            kernel: KernelDef {
                function_name,
                parameters,
                bounds_checking: false,
            },
            compiler: KernelCompiler {
                command: None,
                options: vec![],
                debugging: false,
                working_dir: None,
            },
        }
    }
}

impl Debug for ModuleDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModuleDef")
            .field("source", &"...") // Special case
            .field("file_name", &self.file_name)
            .field("kernel", &self.kernel)
            .field("compiler", &self.compiler)
            .finish()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum KernelParam {
    Array {
        name: String,
        dtype: DataType,
        is_constant: bool,
        dims: Vec<KernelArrayDimension>,
    },
    Value {
        name: String,
        dtype: DataType,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum KernelArrayDimension {
    Free,
    BlockX,
    BlockY,
    BlockZ,
}

impl KernelParam {
    pub fn array(
        name: impl Into<String>,
        dtype: DataType,
        ndims: usize,
        is_constant: bool,
    ) -> Self {
        Self::Array {
            name: name.into(),
            dtype,
            is_constant,
            dims: vec![KernelArrayDimension::Free; ndims],
        }
    }

    pub fn value(name: impl Into<String>, dtype: DataType) -> Self {
        Self::Value {
            name: name.into(),
            dtype,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            KernelParam::Array { name, .. } => name,
            KernelParam::Value { name, .. } => name,
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct KernelConfig {
    pub(crate) block_size: Option<Dim3>,
    pub(crate) block_count: [Option<u64>; MAX_DIMS],
    pub(crate) strides: Vec<ConstraintStride>,
    pub(crate) arguments: Vec<ConstraintArg>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ConstraintStride {
    pub(crate) param_index: usize,
    pub(crate) axis: usize,
    pub(crate) stride: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ConstraintArg {
    pub(crate) param_index: usize,
    pub(crate) value: DataValue,
}
