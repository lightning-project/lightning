use crate::prelude::*;
use crate::types::{DataType, Executor, GenericAccessor, HostAccessor, Rect, Tasklet};
use crate::worker::executor::HostExecutor;
use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;
use std::fmt::{self, Debug};

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct WriteDataTasklet {
    pub(crate) region: Rect,
    pub(crate) dtype: DataType,
    pub(crate) data: ByteBuf,
}

impl Tasklet for WriteDataTasklet {
    type Output = ();

    fn execute(self, arrays: &[GenericAccessor], executor: &dyn Executor) -> Result {
        let executor = executor.downcast_ref::<HostExecutor>()?;
        let dst_array = arrays[0].as_host_mut().unwrap().slice(self.region);

        let src_array = HostAccessor::from_buffer_raw(
            self.data.as_ptr(),
            self.data.len(),
            dst_array.extents(),
            self.dtype,
        );

        unsafe {
            executor.copy(src_array, dst_array)?;
        }

        Ok(())
    }
}

impl Debug for WriteDataTasklet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WriteDataTasklet")
            .field("region", &self.region)
            .field("dtype", &self.dtype)
            .field("data", &format_args!("<{} bytes>", self.data.len()))
            .finish()
    }
}
