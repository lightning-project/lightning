//! Error handling

use cuda_driver_sys::*;
use std::error::Error as StdError;
use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::num::NonZeroU32;
use std::result::Result as StdResult;
use std::{fmt, mem, str};

/// Check the result of a CUDA driver API function.
///
/// All functions in the CUDA driver API return a `CUresult` flag indicating whether the operation
/// was successful or not. This function takes such a flag and returns `Ok(())` on success
/// and `Err(Error)` otherwise, thus making it easy to wrap native CUDA functions.
///
/// # Example
/// ```
/// # use cuda_driver_sys::*;
/// # use cuba::*;
/// let result: Result<(), Error> = unsafe {
///     cuda_check(cuInit(0))
/// };
/// ```
#[inline(always)]
pub fn cuda_check(code: CUresult) -> Result {
    Error::new(code)
}

/// Returns the output value of a CUDA driver API function.
///
/// Many functions in the CUDA driver API follow the pattern where they take a pointer to an
/// uninitialized memory location as an argument and, on success, the output is written to the specified
/// location. This pattern is inconvenient in Rust since working with uninitialized values can be
/// tricky. This function helps to ease the pain: it creates an unitialized value, calls the
/// provided closure with a pointer to that value, and returns `Ok(value)` on success
/// and `Err(Error)` otherwise.
///
/// # Safety
/// This function assume that the provided closure initializes the memory location `*mut T`.
///
/// # Example
/// ```
/// # use cuda_driver_sys::*;
/// # use cuba::*;
/// # use std::os::raw::c_int;
/// // Get the number of CUDA-capable devices in the systems. `result` will be either `Ok(n)`
/// // on success or `Err(Error)` if an error occurs.
/// let result: Result<c_int, Error> = unsafe {
///     cuda_call(|count: *mut c_int| cuDeviceGetCount(count))
/// };
/// ```
#[inline(always)]
pub unsafe fn cuda_call<T, F>(fun: F) -> Result<T>
where
    F: FnOnce(*mut T) -> CUresult,
{
    let mut result = MaybeUninit::uninit();
    let code = (fun)(result.as_mut_ptr());
    match Error::new(code) {
        Ok(_) => Ok(result.assume_init()),
        Err(e) => Err(e),
    }
}

/// Alias for `std::result::Result<T, cuba::Error>`.
pub type Result<T = (), E = Error> = StdResult<T, E>;

/// Error returned by the CUDA driver API.
///
/// Nearly all functions in the CUDA driver API return an `CUresult` status code which is
/// either `CUDA_SUCCESS` on success or one of the `CUDA_ERROR_*` values on an error. This object
/// wraps such an an error status code.
#[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Error(NonZeroU32);

// Error wraps a CUresult which is just an integer, thus Error should be thread-safe.
unsafe impl Send for Error {}
unsafe impl Sync for Error {}

impl Error {
    /// Construct an error object. Returns `Ok(())` for `CUDA_SUCCESS` and `Err(Error)` for
    /// `CUDA_ERROR_*`.
    #[inline(always)]
    pub fn new(code: CUresult) -> Result<(), Self> {
        if code == CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            unsafe { Err(Error::from_raw(code)) }
        }
    }

    /// Get the name of this error. Returns an error if this error code is not recognized by CUDA.
    pub fn name(self) -> Result<&'static str> {
        unsafe {
            let result = cuda_call(|v| cuGetErrorName(self.raw(), v))?;
            let name = CStr::from_ptr(result);
            Ok(str::from_utf8_unchecked(name.to_bytes()))
        }
    }

    /// Get the description of this error. Returns an error if this error code is not recognized by CUDA.
    pub fn description(self) -> Result<&'static str> {
        unsafe {
            let result = cuda_call(|v| cuGetErrorString(self.raw(), v))?;
            let desc = CStr::from_ptr(result);
            Ok(str::from_utf8_unchecked(desc.to_bytes()))
        }
    }

    /// Construct an error object from a raw `CUresult`. Prefer to use [`new`] if possible.
    ///
    /// [`new`]: #method.new
    ///
    /// # Safety
    /// The given status code cannot be `CUDA_SUCCESS`.
    #[inline(always)]
    #[cold]
    pub unsafe fn from_raw(result: CUresult) -> Self {
        // CUDA_SUCCESS should be 0, so we can use a NonZeroU32 to store the remaining codes.
        assert_eq!(CUresult::CUDA_SUCCESS as usize, 0);

        Error(NonZeroU32::new_unchecked(result as u32))
    }

    /// Returns the underlying `CUresult`.
    #[inline(always)]
    pub fn raw(self) -> CUresult {
        unsafe { mem::transmute(self.0) }
    }
}

impl StdError for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (name, description) = (self.name(), (*self).description());

        write!(
            f,
            "{} (errno {}): {}",
            name.ok().as_deref().unwrap_or("<unknown error>"),
            self.0.get(),
            description.ok().as_deref().unwrap_or("<unknown error>"),
        )
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Ok(name) = self.name() {
            f.debug_tuple("CudaError").field(&name).finish()
        } else {
            f.debug_tuple("CudaError").field(&self.0).finish()
        }
    }
}

impl Default for Error {
    fn default() -> Self {
        // Not sure if this makes sense, but unknown error should be a good default value.
        unsafe { Error::from_raw(CUresult::CUDA_ERROR_UNKNOWN) }
    }
}
