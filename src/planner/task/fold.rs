use crate::prelude::*;
use crate::types::{Affine, Dim, Executor, GenericAccessor, Reduction, Tasklet};
use crate::worker::executor::{CudaExecutor, HostExecutor};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct FoldTasklet {
    pub(crate) src_transform: Affine,
    pub(crate) dst_transform: Affine,
    pub(crate) extents: Dim,
    pub(crate) reduction: Reduction,
}

impl Tasklet for FoldTasklet {
    type Output = ();

    fn execute(self, arrays: &[GenericAccessor], executor: &dyn Executor) -> Result {
        let src_array = arrays[0].transform(&self.src_transform, self.extents);
        let dst_array = arrays[1].transform(&self.dst_transform, self.extents);

        if let Ok(executor) = executor.downcast_ref::<CudaExecutor>() {
            unsafe {
                executor.fold_async(
                    src_array.as_device(executor.device_id()).unwrap(),
                    dst_array.as_device_mut(executor.device_id()).unwrap(),
                    self.reduction,
                )?;
            }

            Ok(())
        } else if let Ok(executor) = executor.downcast_ref::<HostExecutor>() {
            unsafe {
                executor.fold(
                    src_array.as_host().unwrap(),
                    dst_array.as_host_mut().unwrap(),
                    self.reduction,
                )?;
            }

            Ok(())
        } else {
            bail!("invalid executor: {}", executor);
        }
    }
}
