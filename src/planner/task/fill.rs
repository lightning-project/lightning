use crate::prelude::*;
use crate::types::{Affine, DataValue, Dim, Executor, GenericAccessor, Tasklet};
use crate::worker::executor::{CudaExecutor, HostExecutor};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct FillTasklet {
    pub(crate) transform: Affine,
    pub(crate) domain: Dim,
    pub(crate) value: DataValue,
}

impl Tasklet for FillTasklet {
    type Output = ();

    fn execute(self, arrays: &[GenericAccessor], executor: &dyn Executor) -> Result {
        if let Ok(executor) = executor.downcast_ref::<HostExecutor>() {
            let dst = arrays[0]
                .as_host_mut()
                .unwrap()
                .transform(&self.transform, self.domain);

            unsafe {
                executor.fill(dst, self.value);
            }

            Ok(())
        } else if let Ok(executor) = executor.downcast_ref::<CudaExecutor>() {
            let dst = arrays[0]
                .as_device_mut(executor.device_id())
                .unwrap()
                .transform(&self.transform, self.domain);

            unsafe {
                executor.fill_async(dst, self.value)?;
            }

            Ok(())
        } else {
            Err(anyhow!("unknown executor: {}", executor))
        }
    }
}
