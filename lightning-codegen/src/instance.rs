use super::generate::{generate_kernel_wrapper, KernelSource};
use super::types::KernelConfig;
use crate::types::ModuleDef;
use cuda_driver_sys::{CUdeviceptr, CUfunction};
use lightning_core::prelude::*;
use lightning_core::util::div_ceil;
use lightning_core::{DataType, DataValue, Dim3, MAX_DIMS};
use lightning_cuda::prelude::*;
use lightning_cuda::Dim3 as CudaDim3;

#[derive(Debug, Clone, Copy)]
pub enum LaunchParam {
    BlockOffset(usize),
    Value {
        index: usize,
        dtype: DataType,
    },
    Array {
        index: usize,
        kind: LaunchParamArray,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum LaunchParamArray {
    Pointer { constant: bool, dtype: DataType },
    Stride(usize),
    LowerBound(usize),
    UpperBound(usize),
}

#[derive(Debug, Clone, Copy)]
pub struct KernelArgArrayAxis {
    pub(crate) stride: i64,
    pub(crate) lbnd: i64,
    pub(crate) ubnd: i64,
}

#[derive(Debug, Clone)]
pub enum KernelArg {
    Value(DataValue),
    Array {
        dtype: DataType,
        ptr: CUdeviceptr,
        strides_and_bounds: Box<[KernelArgArrayAxis]>,
    },
}

impl KernelArg {
    pub fn value<D: Into<DataValue>>(value: D) -> Self {
        Self::Value(value.into())
    }

    pub fn array_dyn(
        ndims: usize,
        dtype: DataType,
        ptr: CudaDevicePtr,
        lbnd: &[i64],
        ubnd: &[i64],
        stride: &[i64],
    ) -> Self {
        let mut strides_and_bounds = Vec::with_capacity(ndims);

        for i in 0..ndims {
            strides_and_bounds.push(KernelArgArrayAxis {
                stride: stride[i],
                lbnd: lbnd[i],
                ubnd: ubnd[i],
            });
        }

        Self::Array {
            dtype,
            ptr: ptr.raw(),
            strides_and_bounds: strides_and_bounds.into_boxed_slice(),
        }
    }

    pub fn array<const N: usize>(
        dtype: DataType,
        ptr: CudaDevicePtr,
        lbnd: [i64; N],
        ubnd: [i64; N],
        stride: [i64; N],
    ) -> Self {
        Self::array_dyn(N, dtype, ptr, &lbnd, &ubnd, &stride)
    }
}

#[derive(Debug)]
pub struct KernelInstance {
    pub(super) module: CudaModule,
    pub(super) fun_ptr: CUfunction,
    pub(super) config: KernelConfig,
    pub(super) params: Vec<LaunchParam>,
}

unsafe impl Send for KernelInstance {}
unsafe impl Sync for KernelInstance {}

#[derive(Error, Debug)]
pub enum LaunchError {
    #[error("{0}")]
    Cuda(#[from] CudaError),

    //#[error("invalid number of arguments given")]
    //InvalidNumberArguments,
    #[error("argument {0} is invalid")]
    InvalidArgument(usize),

    #[error("invalid block size given")]
    InvalidBlockSize,

    #[error("invalid grid size given")]
    InvalidGridSize,

    #[error("parameters given do not match the kernel configuration")]
    InvalidConfig,
}

impl KernelInstance {
    pub(super) fn compile(
        context: CudaContextHandle,
        definition: &ModuleDef,
        config: KernelConfig,
    ) -> Result<Self> {
        let KernelSource {
            source,
            symbol,
            params,
        } = generate_kernel_wrapper(&definition, &config)?;

        let module = definition.compiler.compile(context, &source)?;
        let fun_ptr = module.function(&symbol)?.raw();

        Ok(Self {
            module,
            fun_ptr,
            config,
            params,
        })
    }

    pub unsafe fn launch_async(
        &self,
        stream: &CudaStream,
        block_count: Dim3,
        block_size: Dim3,
        block_offset: Dim3,
        smem_size: u32,
        args: &[KernelArg],
    ) -> Result<(), LaunchError> {
        for i in 0..3 {
            if block_offset[i] + block_count[i] >= u32::MAX as u64 {
                warn!("too many thread blocks: {:?}", block_offset + block_count);
                return Err(LaunchError::InvalidGridSize);
            }
        }

        let max_grid_size = CudaDevice::current()?.max_grid_dim()?;
        let max_grid_size = Dim3::new(
            max_grid_size.x() as u64,
            max_grid_size.y() as u64,
            max_grid_size.z() as u64,
        );

        let num_grids = [
            div_ceil(block_count[0], max_grid_size[0]),
            div_ceil(block_count[1], max_grid_size[1]),
            div_ceil(block_count[2], max_grid_size[2]),
        ];

        for x in 0..num_grids[0] {
            for y in 0..num_grids[1] {
                for z in 0..num_grids[2] {
                    let grid_offset = Dim3::new(
                        x * max_grid_size[0],
                        y * max_grid_size[1],
                        z * max_grid_size[2],
                    );

                    // These casts are safe since max_grid_size was casted from an u32
                    let grid_size = Dim3::new(
                        min(block_count[0] - grid_offset[0], max_grid_size[0]),
                        min(block_count[1] - grid_offset[1], max_grid_size[1]),
                        min(block_count[2] - grid_offset[2], max_grid_size[2]),
                    );

                    self._launch_async(
                        stream,
                        grid_size,
                        block_size,
                        block_offset + grid_offset,
                        smem_size,
                        args,
                    )?;
                }
            }
        }

        Ok(())
    }

    fn is_valid_config(&self, block_count: Dim3, block_size: Dim3, args: &[KernelArg]) -> bool {
        if let Some(expected_block_size) = self.config.block_size {
            if block_size != expected_block_size {
                return false;
            }
        }

        for i in 0..MAX_DIMS {
            if let Some(n) = self.config.block_count[i] {
                if block_count[i] != n {
                    return false;
                }
            }
        }

        for constraint in &self.config.strides {
            match args.get(constraint.param_index) {
                Some(KernelArg::Array {
                    strides_and_bounds, ..
                }) => match strides_and_bounds.get(constraint.axis) {
                    Some(dim) => {
                        if dim.stride == constraint.stride {
                            return false;
                        }
                    }
                    _ => return false,
                },
                _ => return false,
            }
        }

        for constraint in &self.config.arguments {
            match args.get(constraint.param_index) {
                Some(KernelArg::Value(value)) => {
                    if value != &constraint.value {
                        return false;
                    }
                }
                _ => return false,
            }
        }

        true
    }

    unsafe fn _launch_async(
        &self,
        stream: &CudaStream,
        block_count: Dim3,
        block_size: Dim3,
        block_offset: Dim3,
        smem_size: u32,
        args: &[KernelArg],
    ) -> Result<(), LaunchError> {
        use LaunchError::*;
        let params = &self.params;

        if !self.is_valid_config(block_count, block_size, args) {
            return Err(InvalidConfig);
        }

        let block_size = match (
            block_size[0].try_into(),
            block_size[1].try_into(),
            block_size[2].try_into(),
        ) {
            (Ok(x), Ok(y), Ok(z)) => CudaDim3::new(x, y, z),
            _ => return Err(InvalidBlockSize),
        };

        let block_count = match (
            block_count[0].try_into(),
            block_count[1].try_into(),
            block_count[2].try_into(),
        ) {
            (Ok(x), Ok(y), Ok(z)) => CudaDim3::new(x, y, z),
            _ => return Err(InvalidGridSize),
        };

        let mut ptrs = Vec::with_capacity(params.len());

        for param in params {
            match param {
                &LaunchParam::BlockOffset(axis) => {
                    ptrs.push(&block_offset[axis] as *const _ as *const ());
                }
                &LaunchParam::Value { index, dtype } => {
                    let value = match args.get(index) {
                        Some(KernelArg::Value(v)) => v,
                        _ => {
                            return Err(InvalidArgument(index));
                        }
                    };

                    if value.data_type() != dtype {
                        return Err(InvalidArgument(index));
                    }

                    ptrs.push(value.as_raw_data() as *const [u8] as *const u8 as *const ());
                }
                &LaunchParam::Array { index, kind } => {
                    use LaunchParamArray::*;

                    match args.get(index) {
                        Some(KernelArg::Array {
                            dtype,
                            ptr,
                            strides_and_bounds,
                        }) => match kind {
                            Pointer {
                                dtype: expected, ..
                            } => {
                                if *dtype != expected {
                                    return Err(InvalidArgument(index));
                                }

                                // Alignment check
                                let elem_size = dtype.alignment() as u64;
                                assert_eq!(*ptr % elem_size, 0, "pointer is misaligned");

                                ptrs.push(ptr as *const CUdeviceptr as *const ());
                            }
                            Stride(axis) | LowerBound(axis) | UpperBound(axis) => {
                                let dimension = match strides_and_bounds.get(axis) {
                                    Some(x) => x,
                                    None => return Err(InvalidArgument(index)),
                                };

                                let value: &i64 = match kind {
                                    Stride(_) => &dimension.stride,
                                    LowerBound(_) => &dimension.lbnd,
                                    UpperBound(_) => &dimension.ubnd,
                                    _ => unreachable!(),
                                };

                                ptrs.push(value as *const i64 as *const ());
                            }
                        },
                        _ => {
                            return Err(InvalidArgument(index));
                        }
                    };
                }
            }
        }

        CudaFunction::from_raw(self.fun_ptr).launch_async(
            stream,
            block_count,
            block_size,
            smem_size,
            &ptrs,
        )?;

        Ok(())
    }
}
