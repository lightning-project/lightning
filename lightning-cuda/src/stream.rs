//! Management of CUDA streams.

use crate::{cuda_call, cuda_check, Error, Event, Result};
use cuda_driver_sys::cudaError_enum::{CUDA_ERROR_NOT_READY, CUDA_SUCCESS};
use cuda_driver_sys::*;
use std::ffi::c_void;
use std::os::raw::c_uint;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::process::abort;
use std::{fmt, mem};

/// CUDA Compute stream.
///
/// CUDA uses the concept of streams to organize asynchronous operations. A stream is basically a
/// queue of work performed on host and the device (e.g., kernel launches, memory copies). This object
/// wraps a `CUstream`.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Stream(CUstream);

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    /// Construct a new stream.
    pub fn new() -> Result<Self> {
        Self::with_options(None, StreamFlags::NON_BLOCKING)
    }

    /// Construct a new stream with the given priority and flags. The given priority should
    /// be in the range provided by `Stream::priority_range()`. Lower numbers represent higher
    /// priorities.
    pub fn with_options(priority: Option<i32>, flags: StreamFlags) -> Result<Self> {
        unsafe {
            let raw = cuda_call(|v| match priority {
                Some(p) => cuStreamCreateWithPriority(v, flags.bits, p),
                None => cuStreamCreate(v, flags.bits),
            })?;

            Ok(Self(raw))
        }
    }

    /// Returns the CUDA default stream.
    pub fn default() -> Result<Self> {
        Ok(Self(0 as _))
    }

    /// Returns true if this stream represents the CUDA default stream.
    pub fn is_default(&self) -> bool {
        self.0.is_null()
    }

    /// Returns the numerical range for stream priorities as a tuple of (least-priority, greatest-priority).
    /// Note that lower priority number represent higher priorities, i.e., least-priority >= greatest-priority.
    pub fn priority_range() -> Result<(i32, i32)> {
        unsafe {
            let (mut least, mut greatest) = (-1, -1);
            cuda_check(cuCtxGetStreamPriorityRange(&mut least, &mut greatest))?;
            Ok((least, greatest))
        }
    }

    /// Returns the priority of this stream.
    pub fn priority(&self) -> Result<i32> {
        unsafe { cuda_call(|v| cuStreamGetPriority(self.0, v)) }
    }

    /// Makes all future work for this stream wait until the given event has fired.
    pub fn wait_for_event(&self, event: &Event) -> Result {
        unsafe { cuda_check(cuStreamWaitEvent(self.0, event.raw(), 0)) }
    }

    /// Add a callback onto this stream. The callback will be called after previously
    /// enqueued work has finished and will be block work added after it. See `cuStreamAddCallback`
    /// for further restrictions and guarantees.
    ///
    /// The callback will be passed either `Ok(())` or an error code, either because CUDA failed
    /// to enqueue the callback or because an asynchronous device error occurred.
    ///
    /// The callback must not panic. Any panic will abort the entire process as to prevent the
    /// panic from crossing from Rust into the CUDA runtime.
    pub fn add_callback<F>(&self, fun: F)
    where
        F: FnOnce(Result) + Send + 'static,
    {
        // Wrapped function using catch_unwind.
        let wrappped_fun = move |result| {
            let result = catch_unwind(AssertUnwindSafe(|| (fun)(result)));

            if result.is_err() {
                eprintln!("fatal error: {:?}", result);
                abort();
            }
        };

        // We use the following optimization:
        // - If `size_of::<F>()` <= `size_of::<*mut c_void>()`, then we can transmute `F` into
        //   `*mut c_void` and pass this directly to cuStreamAddCallback.
        // - Otherwise, we must box `F` and pass the boxed pointer to cuStreamAddCallback.
        use std::mem::{size_of, ManuallyDrop};
        union Transmute<F> {
            fun: ManuallyDrop<F>,
            user_data: *mut c_void,
        }

        if size_of::<Transmute<F>>() <= size_of::<*mut c_void>() {
            unsafe extern "C" fn __stream_callback_union<F>(
                _stream: CUstream,
                err: CUresult,
                user_data: *mut c_void,
            ) where
                F: FnOnce(Result),
            {
                let package = Transmute { user_data };
                let fun: F = ManuallyDrop::into_inner(package.fun);
                (fun)(Error::new(err));
            }

            let package = Transmute {
                fun: ManuallyDrop::new(wrappped_fun),
            };

            let result = unsafe {
                cuStreamAddCallback(
                    self.0,
                    Some(__stream_callback_union::<F>),
                    package.user_data,
                    0,
                )
            };

            // If result != SUCCESS, then cuStreamAddCallback did not take ownership of the
            // callback and we must call it here manually.
            if result != CUDA_SUCCESS {
                unsafe {
                    __stream_callback_union::<F>(self.raw(), result, package.user_data);
                }
            }
        } else {
            unsafe extern "C" fn __stream_callback_boxed<F>(
                _stream: CUstream,
                err: CUresult,
                user_data: *mut c_void,
            ) where
                F: FnOnce(Result),
            {
                let boxed: Box<F> = Box::from_raw(user_data as *mut F);
                let result = Error::new(err);
                (boxed)(result);
            }

            let boxed = Box::into_raw(Box::new(wrappped_fun));
            let result = unsafe {
                cuStreamAddCallback(
                    self.0,
                    Some(__stream_callback_boxed::<F>),
                    boxed as *mut c_void,
                    0,
                )
            };

            // If result != SUCCESS, then cuStreamAddCallback did not take ownership of the
            // callback and we must call it here manually.
            if result != CUDA_SUCCESS {
                unsafe {
                    __stream_callback_boxed::<F>(self.raw(), result, boxed as *mut c_void);
                }
            }
        }
    }

    /// Returns `true` if all previously enqueued work onto the stream has completed and `false` otherwise.
    pub fn query(&self) -> Result<bool> {
        unsafe {
            match cuStreamQuery(self.0) {
                CUDA_SUCCESS => Ok(true),
                CUDA_ERROR_NOT_READY => Ok(false),
                err => Err(Error::from_raw(err)),
            }
        }
    }

    /// Block until all previously enqueued work onto this stream has completed.
    pub fn synchronize(&self) -> Result {
        unsafe { cuda_check(cuStreamSynchronize(self.0)) }
    }

    /// Returns the underlying `CUstream` object.
    #[inline(always)]
    pub fn raw(&self) -> CUstream {
        self.0
    }

    /// Returns the underlying `CUstream` object and consume this object, preventing the destructor
    /// from being run.
    #[inline(always)]
    pub fn into_raw(self) -> CUstream {
        let out = self.0;
        mem::forget(self);
        out
    }

    /// Construct from a `CUstream` object.
    ///
    /// # Safety
    /// The given `CUstream` object should be a valid CUDA stream object.
    #[inline(always)]
    pub unsafe fn from_raw(stream: CUstream) -> Self {
        Self(stream)
    }
}

impl fmt::Debug for Stream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("CudaStream").field(&self.0).finish()
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            if !self.is_default() {
                cuStreamDestroy_v2(self.0);
            }
        }
    }
}

bitflags::bitflags! {
    /// Flags for configuring a CUDA stream.
    ///
    /// Represent the `CU_STREAM_*` flags from the CUDA API.
    pub struct StreamFlags: c_uint {
        const NON_BLOCKING = CUstream_flags_enum::CU_STREAM_NON_BLOCKING as c_uint;
        const DEFAULT = CUstream_flags_enum::CU_STREAM_DEFAULT as c_uint;
    }
}
