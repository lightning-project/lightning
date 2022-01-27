use crate::prelude::*;
use crate::types::{AffineNM, Dim3, Dim4, Executor, GenericAccessor, Reduction, Tasklet};
use crate::worker::executor::CudaExecutor;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ReduceTasklet {
    pub(crate) src_transform: AffineNM<4, 3>,
    pub(crate) dst_transform: AffineNM<3, 3>,
    pub(crate) axis: usize,
    pub(crate) reduction: Reduction,
    pub(crate) extents: Dim4,
}

impl Tasklet for ReduceTasklet {
    type Output = ();

    fn execute(self, arrays: &[GenericAccessor], executor: &dyn Executor) -> Result {
        let executor = executor.downcast_ref::<CudaExecutor>()?;
        let device_id = executor.device_id();

        let src_extent = self.extents;
        let dst_extent = match self.axis {
            0 => Dim3::new(src_extent[1], src_extent[2], src_extent[3]),
            1 => Dim3::new(src_extent[0], src_extent[2], src_extent[3]),
            2 => Dim3::new(src_extent[0], src_extent[1], src_extent[3]),
            3 => Dim3::new(src_extent[0], src_extent[1], src_extent[2]),
            _ => bail!("invalid axis: {}", self.axis),
        };

        let src = arrays[0]
            .as_device(device_id)
            .ok_or_else(|| anyhow!("invalid array type"))?
            .transform(&self.src_transform, src_extent);

        let dst = arrays[1]
            .as_device_mut(device_id)
            .ok_or_else(|| anyhow!("invalid array type"))?
            .transform(&self.dst_transform, dst_extent);

        unsafe { executor.reduce_async(src, dst, self.axis, self.reduction) }
    }
}
