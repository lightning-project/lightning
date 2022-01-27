use super::instance::KernelInstance;
use super::types::{ConstraintArg, ConstraintStride, KernelConfig};
use super::KernelArg;
use crate::types::ModuleDef;
use lightning_core::prelude::*;
use lightning_core::{Dim3, PrimitiveType, MAX_DIMS};
use lightning_cuda::prelude::{CudaContextHandle, CudaStream};

#[derive(Debug, PartialEq, Eq, Copy, Clone, PartialOrd, Ord)]
pub enum KernelSpecializationPolicy {
    None,           // No specialization
    Mild,           // Only specialize on block size
    Standard,       // Specialize on block size and contiguous strides
    Aggressive,     // Specialize on block size and all strides
    VeryAggressive, // Specialize on block size, all strides and all arguments
}

impl Default for KernelSpecializationPolicy {
    fn default() -> Self {
        KernelSpecializationPolicy::Standard
    }
}

#[derive(Debug)]
pub struct Kernel {
    def: ModuleDef,
    policy: KernelSpecializationPolicy,
    instances: Vec<KernelInstance>,
}

impl Kernel {
    pub fn new(def: ModuleDef, policy: KernelSpecializationPolicy) -> Self {
        Self {
            def,
            policy,
            instances: vec![],
        }
    }

    fn _find_instance(
        &mut self,
        context: CudaContextHandle,
        config: KernelConfig,
    ) -> Result<&KernelInstance> {
        let index = match self
            .instances
            .iter()
            .position(|instance| instance.config == config)
        {
            Some(index) => index,
            None => {
                debug!(
                    "compiling kernel {} with config {:#?}",
                    self.def.kernel.function_name, config
                );

                let instance = KernelInstance::compile(context, &self.def, config)?;
                self.instances.push(instance);
                self.instances.len() - 1
            }
        };

        Ok(&self.instances[index])
    }

    pub fn compile(&mut self, context: CudaContextHandle, config: KernelConfig) -> Result {
        self._find_instance(context, config)?;
        Ok(())
    }

    pub unsafe fn launch_async(
        &mut self,
        context: CudaContextHandle,
        stream: &CudaStream,
        block_count: Dim3,
        block_size: Dim3,
        block_offset: Dim3,
        smem_size: u32,
        args: &[KernelArg],
    ) -> Result {
        use KernelSpecializationPolicy::*;

        let mut config = KernelConfig::default();

        if self.policy >= Mild {
            config.block_size = Some(block_size);

            for i in 0..MAX_DIMS {
                if block_count[i] == 1 {
                    config.block_count[i] = Some(1);
                }
            }
        }

        if self.policy >= Standard {
            for (param_index, arg) in enumerate(args) {
                if let KernelArg::Array {
                    strides_and_bounds, ..
                } = arg
                {
                    for (axis, dimension) in enumerate(&**strides_and_bounds) {
                        let stride = dimension.stride;

                        let specialize = match self.policy {
                            Aggressive | VeryAggressive => true,
                            Standard => stride == 0 || stride == 1,
                            _ => false,
                        };

                        if specialize {
                            config.strides.push(ConstraintStride {
                                param_index,
                                axis,
                                stride,
                            });
                        }
                    }
                }
            }
        }

        if self.policy >= VeryAggressive {
            for (param_index, arg) in enumerate(args) {
                if let KernelArg::Value(value) = arg {
                    use PrimitiveType::*;

                    match value.data_type().to_primitive() {
                        Some(I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64) => {
                            config.arguments.push(ConstraintArg {
                                param_index,
                                value: value.clone(),
                            });
                        }
                        _ => {
                            //
                        }
                    }
                }
            }
        }

        self._find_instance(context, config)?.launch_async(
            stream,
            block_count,
            block_size,
            block_offset,
            smem_size,
            args,
        )?;

        Ok(())
    }
}
