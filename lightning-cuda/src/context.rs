//! Management of CUDA contexts
//!
//!

use crate::{cuda_call, cuda_check, Device, Error, Result};
use cuda_driver_sys::*;
use std::fmt;
use std::marker::PhantomData;
use std::os::raw::c_uint;
use std::ptr::NonNull;

/// Create a new CUDA context for the given device.
///
/// The caller ***must*** destroy the context when done using it by calling [`destroy_context`],
/// otherwise the context will leak resources.
///
/// [`destroy_context`]: fn.destroy_context.html
pub fn create_context(device: Device, flags: ContextFlags) -> Result<ContextHandle> {
    unsafe {
        let handle = cuda_call(|c| cuCtxCreate_v2(c, flags.bits, device.raw()))?;
        let popped = cuda_call(|c| cuCtxPopCurrent_v2(c))?;
        assert_eq!(handle, popped);

        Ok(ContextHandle::from_raw(handle).unwrap())
    }
}

/// Access the primary CUDA context for the given device.
///
/// Each CUDA device is associated with one _primary_ context. This function returns the
/// primary context for the given the device and increments its usage count by one. This allows
/// multiple different libraries to safely interact without explicitly exchanging CUDA contexts.
///
/// The caller ***must*** call [`destroy_context`]  when done using the context to decrease
/// the usage count. Failing to do so will leak resources.
///
/// [`destroy_context`]: fn.destroy_context.html
pub fn retain_device_context(device: Device) -> Result<ContextHandle> {
    unsafe {
        let h = cuda_call(|c| cuDevicePrimaryCtxRetain(c, device.raw()))?;
        Ok(ContextHandle::from_raw(h).unwrap())
    }
}

/// Release the primary CUDA context for the given device.
///
/// Decrements the usage count of the primary context associated with a device. Once the usage count
/// drops to zero, the primary context is destroyed by the CUDA runtime.
///
/// # Safety
/// All CUDA resources (events, streams, memory, modules) associated with the given context will be
/// invalidated once this function returns. It is best to destroy all these resources before
/// destroying the context. For example, the buffer backing a [`PinnedMem`] will be deallocated,
/// meaning any attempt to access the buffer results in memory faults.
///
/// [`PinnedMem`]: ../mem/struct.PinnedMem.html
pub unsafe fn release_device_context(device: Device) -> Result {
    cuda_check(cuDevicePrimaryCtxRelease(device.raw()))
}

/// Destroys the CUDA context associated with the given handle.
///
/// The context must have been created using [`create_context`].
///
/// [`create_context`]: fn.create_context.html
///
/// # Safety
/// All CUDA resources (events, streams, memory, modules) associated with the given context will be
/// invalidated once this function returns. It is best to destroy all these resources before
/// destroying the context. For example, the buffer backing a [`PinnedMem`] will be deallocated,
/// meaning any attempt to access the buffer results in memory faults.
///
/// [`PinnedMem`]: ../mem/struct.PinnedMem.html
pub unsafe fn destroy_context(handle: ContextHandle) -> Result {
    cuda_check(cuCtxDestroy_v2(handle.raw()))
}

/// Handle to a CUDA context.
///
/// Represents a pointer to valid CUDA context. Use [`with`], [`try_with`], or [`activate`] to
/// push this CUDA context on to the thread-local context stack managed by the CUDA runtime.
///
/// [`with`]: #method.with
/// [`try_with`]: #method.try_with
/// [`activate`]: #method.activate
///
/// Note that this types does not _own_ the associated context, but is merely a handle the
/// context. This handle can be obtained using either [`create_context`] or [`retain_device_context`].
/// The associated context is _not_ destroyed automatically when the handle goes out of scope, but
/// it must be destroyed manually using either [`destroy_context`] or [`release_device_context`]
/// when done using it.
///
/// [`create_context`]: fn.create_context.html
/// [`retain_device_context`]: fn.retain_device_context.html
/// [`destroy_context`]: fn.destroy_context.html
/// [`release_device_context`]: fn.release_device_context.html
#[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
pub struct ContextHandle(NonNull<()>);

// CUDA driver API is thread-safe, so sending a context across threads should be safe.
unsafe impl Send for ContextHandle {}
unsafe impl Sync for ContextHandle {}

impl fmt::Debug for ContextHandle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("CudaContextHandle").field(&self.0).finish()
    }
}

impl ContextHandle {
    /// Returns the top CUDA context in the thread-local stack of contexts.
    ///
    /// Yields `None` if the stack is empty.
    #[inline]
    pub fn current() -> Result<Option<ContextHandle>> {
        unsafe { cuda_call(|c| cuCtxGetCurrent(c)).map(|c| Self::from_raw(c)) }
    }

    /// The device associated with this context.
    #[inline]
    pub fn device(self) -> Result<Device> {
        self.try_with(|| Device::current())
    }

    /// Call the the given closure while this context is active. This is equivalent to pushing this
    /// context, calling the closure, and popping it again.
    ///
    /// ```
    /// # use cuba::*;
    /// # fn fun() {}
    /// let context = create_context(Device::nth(0)?, ContextFlags::empty())?;
    ///
    /// // Call fun while the context is active.
    /// context.with(|| {
    ///     fun();
    /// })?;
    ///
    /// // The above is equivalent to the following, but less error-prone when the work could
    /// // return early.
    /// context.push()?;
    /// fun();
    /// context.pop()?;
    /// ```
    #[inline]
    pub fn with<F, T>(self, fun: F) -> Result<T>
    where
        F: FnOnce() -> T,
    {
        self.try_with(|| Ok((fun)()))
    }

    /// Similar to [`with`], but allows fallible operations.
    ///
    /// [`with`]: #method.with
    #[inline]
    pub fn try_with<F, T, E>(self, fun: F) -> Result<T, E>
    where
        F: FnOnce() -> Result<T, E>,
        E: From<Error>,
    {
        let guard = self.activate()?;
        let out = (fun)();
        drop(guard);

        out
    }

    /// Make this context the active context. A RAII guard is returned. When the guard is destroyed,
    /// the context is deactivated again.
    ///
    /// This is equivalent to pushing this context and popping it again once done.
    ///
    /// ```
    /// # use cuba::*;
    /// # fn fun() {}
    /// let context = create_context(Device::nth(0)?, ContextFlags::empty())?;
    ///
    /// // Call fun while the context is active.
    /// {
    ///     // Must assign guard to variable, otherwise it is immediately destroyed.
    ///     let guard = context.activate();
    ///
    ///     // Perform work.
    ///     ..;
    ///
    ///     // `guard` goes out of scope and the context is deactivated.
    /// }
    ///
    /// // The above is equivalent to the following, but less error-prone when the
    /// // work could return early (e.g., break, return, continue, panic).
    /// {
    ///     context.push()?;
    ///     ..;
    ///     context.pop()?;
    /// }
    /// ```
    #[inline]
    pub fn activate(&self) -> Result<ContextGuard<'_>> {
        self.push()?;
        Ok(ContextGuard(PhantomData))
    }

    /// Push this context on top of the thread-local stack of context, making it the active context
    /// for CUDA operations. The context must be deactivated using [`pop`] when done. Usage of
    /// this method is not recommended since it is easy to accidentally forget to pop the
    /// context. See [`with`], [`try_with`], and [`activate`]  for alternatives.
    ///
    /// [`pop`]: #method.pop
    /// [`with`]: #method.with
    /// [`try_with`]: #method.try_with
    /// [`activate`]: #method.activate
    // TODO: Should this be unsafe?
    #[inline]
    pub fn push(self) -> Result {
        unsafe { cuda_check(cuCtxPushCurrent_v2(self.raw())) }
    }

    /// Pops the current context of the thread-local stack and returns the handle. Returns `None`
    /// if the stack was empty.
    // TODO: Should this be unsafe?
    #[inline]
    pub fn pop() -> Result<Option<ContextHandle>> {
        unsafe { cuda_call(|c| cuCtxPopCurrent_v2(c)).map(|c| Self::from_raw(c)) }
    }

    /// Blocks until all requested work on this context have completed.
    pub fn synchronize(self) -> Result {
        self.try_with(|| unsafe { cuda_check(cuCtxSynchronize()) })
    }

    /// Returns the underlying `CUcontext` of this handle.
    #[inline(always)]
    pub fn raw(self) -> CUcontext {
        self.0.as_ptr() as CUcontext
    }

    /// Construct a `ContextHandle` for the given `CUcontext`. Returns `None` if the argument is
    /// `null`.
    ///
    /// # Safety
    /// The given argument must a valid `CUcontext` as this function performs no additional checks.
    #[inline(always)]
    pub unsafe fn from_raw(c: CUcontext) -> Option<Self> {
        NonNull::new(c as *mut ()).map(Self)
    }

    // Returns the free and total amount of memory for the device associated with this context.
    pub fn memory_free_and_total(self) -> Result<(usize, usize)> {
        self.try_with(|| {
            let (mut total, mut free) = (0, 0);
            cuda_check(unsafe { cuMemGetInfo_v2(&mut free, &mut total) })?;
            Ok((free, total))
        })
    }

    /// Enable peer access
    pub fn enable_peer_access(self) -> Result {
        self.try_with(|| cuda_check(unsafe { cuCtxEnablePeerAccess(self.raw(), 0) }))
    }

    /// Disable peer access
    pub fn disable_peer_access(self) -> Result {
        self.try_with(|| cuda_check(unsafe { cuCtxDisablePeerAccess(self.raw()) }))
    }

    pub fn can_access_peer(self, peer: Device) -> Result<bool> {
        self.device()?.can_access_peer(peer)
    }
}

/// RAII guard returned by [`ContextHandle#activate`].
///
/// Pops the current CUDA context when dropped. See the above method for details.
///
/// [`ContextHandle#activate`]: struct.ContextHandle.html#method.activate
#[derive(Debug)]
// PhantomData has two parts:
// * "&'a ContextHandle: To make sure the guard does not outlive the handle.
// * "*mut ()": To make sure the type is !Send.
pub struct ContextGuard<'a>(PhantomData<(&'a ContextHandle, *mut i32)>);

impl Drop for ContextGuard<'_> {
    fn drop(&mut self) {
        let _ = ContextHandle::pop();
    }
}

bitflags::bitflags! {
    /// Flags for configuring a CUDA context.
    ///
    /// Represents the `CU_CTX_*` flags from the CUDA driver API.
    #[allow(non_camel_case_types)]
    pub struct ContextFlags: c_uint {
        const SCHED_SPIN = CUctx_flags_enum::CU_CTX_SCHED_SPIN as c_uint;
        const SCHED_YIELD = CUctx_flags_enum::CU_CTX_SCHED_YIELD as c_uint;
        const BLOCING_SYNC = CUctx_flags_enum::CU_CTX_SCHED_BLOCKING_SYNC as c_uint;
        const SCHED_AUTO = CUctx_flags_enum::CU_CTX_SCHED_AUTO as c_uint;
        const MAP_HOST = CUctx_flags_enum::CU_CTX_MAP_HOST as c_uint;
        const LMEM_RESIZE_TO_MAX = CUctx_flags_enum::CU_CTX_LMEM_RESIZE_TO_MAX as c_uint;
    }
}
