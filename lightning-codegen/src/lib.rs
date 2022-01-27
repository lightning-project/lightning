pub use self::generate::make_valid_ident;
pub use self::instance::KernelArg;
pub use self::kernel::{Kernel, KernelSpecializationPolicy};
pub use self::types::{KernelArrayDimension, KernelConfig, KernelDef, KernelParam, ModuleDef};

// C++ namespace of classes
pub const CPP_NAMESPACE: &str = "::lightning";

mod compile;
mod generate;
mod instance;
mod kernel;
mod types;
