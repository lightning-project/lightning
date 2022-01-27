//! Management of CUDA profiler.

use crate::{cuda_check, Result};
use cuda_driver_sys::*;

pub fn profiler_start() -> Result {
    unsafe { cuda_check(cuProfilerStart()) }
}

pub fn profiler_stop() -> Result {
    unsafe { cuda_check(cuProfilerStop()) }
}
