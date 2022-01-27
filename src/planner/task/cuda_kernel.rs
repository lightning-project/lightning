use crate::prelude::*;
use crate::types::{CudaArg, CudaKernelId, Dim, Executor, GenericAccessor, Point, Tasklet};
use crate::worker::executor::CudaExecutor;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct CudaLaunchTasklet {
    pub kernel_id: CudaKernelId,
    pub block_offset: Point,
    pub block_count: Dim,
    pub block_size: Dim,
    pub shared_mem: u32,
    pub args: Vec<CudaArg>,
}

impl Tasklet for CudaLaunchTasklet {
    type Output = ();

    fn execute(self, arrays: &[GenericAccessor], executor: &dyn Executor) -> Result {
        let executor = executor.downcast_ref::<CudaExecutor>()?;

        unsafe {
            executor.launch_kernel_async(
                self.kernel_id,
                self.block_count,
                self.block_size,
                self.block_offset,
                self.shared_mem,
                &self.args,
                arrays,
            )
        }
    }
}
