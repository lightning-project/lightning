use crate::prelude::*;
use crate::types::{DataType, Executor, GenericAccessor, HostMutAccessor, Rect, Tasklet};
use crate::worker::executor::HostExecutor;
use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;
use std::fmt::Debug;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ReadDataTasklet {
    pub(crate) region: Rect,
    pub(crate) dtype: DataType,
}

impl Tasklet for ReadDataTasklet {
    type Output = ByteBuf;

    fn execute(self, arrays: &[GenericAccessor], executor: &dyn Executor) -> Result<ByteBuf> {
        let executor = executor.downcast_ref::<HostExecutor>()?;
        let src_array = arrays[0].as_host().unwrap().slice(self.region);

        // Ensure that dst data has correct size.
        let extent = src_array.extents();
        let elem_size = self.dtype.size_in_bytes() as usize;
        let mut data = ByteBuf::new();
        data.resize_with(extent.volume() as usize * elem_size, default);

        let dst_array = HostMutAccessor::from_buffer_raw(
            data.as_mut_ptr(),
            data.len(),
            src_array.extents(),
            self.dtype,
        );

        unsafe {
            executor.copy(src_array, dst_array)?;
        }

        Ok(data)
    }
}
