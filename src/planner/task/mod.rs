mod cuda_kernel;
mod fill;
mod fold;
mod read_data;
mod reduce;
mod write_data;

pub(crate) use self::cuda_kernel::*;
pub(crate) use self::fill::*;
pub(crate) use self::fold::*;
pub(crate) use self::read_data::*;
pub(crate) use self::reduce::*;
pub(crate) use self::write_data::*;
use crate::types::register_tasklet;

pub(crate) fn register_tasklets() {
    register_tasklet::<ReadDataTasklet>();
    register_tasklet::<WriteDataTasklet>();
    register_tasklet::<CudaLaunchTasklet>();
    register_tasklet::<FillTasklet>();
    register_tasklet::<FoldTasklet>();
    register_tasklet::<ReduceTasklet>();
}
