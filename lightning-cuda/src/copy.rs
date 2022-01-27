//! Copying between device/host buffers..
use crate::mem::Scalar;
use crate::{cuda_check, DeviceMem, DeviceSlice, DeviceSliceMut, PinnedMem, Result, Stream};
use crate::{Contiguous, DevicePtr};
use cuda_driver_sys::*;
use std::mem::size_of;

/// Copies raw bytes .
///
/// Copies `n * size_of<T>()` raw bytes from `src` to `dst`.  If possible, prefer to use the
/// safer [`copy`] function instead. The operation is performed synchronously with respect to the host
/// and this function does not return until all data has been copied.
///
/// [`copy`]: fn.copy.html
///
/// # Safety
///  * `src` should be a unified virtual address space pointer which should allow for `n * size_of<T>` consecutive reads.
///  * `dst` should be a unified virtual address space pointer which should allow for `n * size_of<T>` consecutive writes.
///  * Elements of type `T` are bitwise copied which could lead to violating memory safety.
///
/// # Example
/// ```
/// # use cuba::*;
/// // Allocate device memory.
/// let x = DeviceMem::<i32>::zeroed(10)?;
/// let mut y = DeviceMem::<i32>::empty(10)?;
///
/// // Copy an array. In this case, we could have also use the regular `copy` function.
/// unsafe {
///     copy_raw(x.as_ptr(), y.as_ptr(), 10)?;
/// }
/// ```
pub unsafe fn copy_raw<T>(src: DevicePtr<T>, dst: DevicePtr<T>, n: usize) -> Result {
    cuda_check(cuMemcpy(dst.raw(), src.raw(), n * size_of::<T>()))
}

/// Copies raw bytes asynchronously.
///
/// Asynchronously copies `n * size_of<T>()` raw bytes from `src` to `dst` on the provided `stream`.
/// If possible, prefer to use the safer [`copy_async`] function instead.
///
/// [`copy_async`]: fn.copy_async.html
///
/// # Safety
///  * `src` should be a unified virtual address space pointer which should allow for `n * size_of<T>` consecutive reads.
///  * `dst` should be a unified virtual address space pointer which should allow for `n * size_of<T>` consecutive writes.
///  * Elements of type `T` are bitwise copied which could lead to violating memory safety.
///  * The function returns immediately while the copy is performed asynchronously. The memory locations
///    pointed to by `src` and `dst` should be not mutated while the operation is in progress.
///
/// # Example
/// ```
/// # use cuba::*;
/// // Allocate device memory.
/// let x = DeviceMem::<i32>::zeroed(10)?;
/// let mut y = DeviceMem::<i32>::empty(10)?;
/// let stream = Stream::new()?;
///
/// // Copy an array. In this case, we could have also use the regular `copy_async` function.
/// unsafe {
///     copy_raw_async(&stream, x.as_ptr(), y.as_ptr(), 10)?;
/// }
///
/// // Wait until copy completes.
/// stream.synchronize();
/// ```
pub unsafe fn copy_raw_async<T>(
    stream: &Stream,
    src: DevicePtr<T>,
    dst: DevicePtr<T>,
    n: usize,
) -> Result {
    cuda_check(cuMemcpyAsync(
        dst.raw(),
        src.raw(),
        n * size_of::<T>(),
        stream.raw(),
    ))
}

/// Copies `n` elements of type `T` from `src` to `dst`.
///
/// Parameters `src` and `dst` should be memory locations which can be read/written by CUDA
/// (see [`CopySource`] and [`CopyDestination`]). Type `T` should be [`Scalar`] to ensure the
/// copy operation is safe. The operation is performed synchronously with respect to the host
// and this function does not return until all data has been copied.
///
/// [`CopySource`]: trait.CopySource.html
/// [`CopyDestination`]: trait.CopyDestination.html
/// [`Scalar`]: trait.Scalar.html
///
/// # Panics
/// Panics if `src` and `dst` have different lengths.
///
/// # Example
/// ```
/// # use cuba::*;
/// // Allocate device memory.
/// let mut x = DeviceMem::empty(3)?;
///
/// // Copy an array.
/// unsafe {
///     copy(&[1, 2, 3], &mut x)?;
/// }
/// ```
pub fn copy<S, D, T>(src: S, mut dst: D) -> Result
where
    S: CopySource<Item = T>,
    D: CopyDestination<Item = T>,
    T: Scalar,
{
    let (src, n) = src.as_source();
    let (dst, m) = dst.as_destination();
    assert_eq!(n, m);

    unsafe { copy_raw(src, dst, n) }
}

/// Copies `n` elements of type `T` from `src` to `dst` asynchronously.
///
/// Parameters `src` and `dst` should be memory locations which can be read/written by CUDA
/// (see [`CopySource`] and [`CopyDestination`]). Type `T` should be [`Scalar`] to ensure the
/// copy operation is memory safe.
///
/// [`CopySource`]: trait.CopySource.html
/// [`CopyDestination`]: trait.CopyDestination.html
/// [`Scalar`]: trait.Scalar.html
///
/// # Panics
/// Panics if `src` and `dst` have different lengths.
///
/// # Safety
///  The function returns immediately while the copy is performed asynchronously. The memory locations
///    pointed to by `src` and `dst` should be not mutated while the operation is in progress.
///
/// # Example
/// ```
/// # use cuba::*;
/// // Allocate device memory and create stream.
/// let mut x = DeviceMem::empty(3)?;
/// let stream = Stream::new()?;
///
/// // Copy an array.
/// unsafe { copy_async(&stream, &[1, 2, 3], &mut x)?; }
///
/// // Wait until copy completes.
/// stream.synchronize();
/// ```
pub unsafe fn copy_async<S, D, T>(stream: &Stream, src: S, mut dst: D) -> Result
where
    S: CopySource<Item = T>,
    D: CopyDestination<Item = T>,
    T: Scalar,
{
    let (src_ptr, n) = src.as_source();
    let (dst_ptr, m) = dst.as_destination();
    assert_eq!(n, m);
    copy_raw_async(stream, src_ptr, dst_ptr, n)
}

/// Memory region suitable to be used as copy source in [`copy`] and [`copy_async`].
///
/// [`copy`]: fn.copy.html
/// [`copy_async`]: fn.copy_async.html
///
/// # Safety
/// This trait should return a memory region that is valid to be used as source in `cuMemcpy`.
pub unsafe trait CopySource {
    type Item: Scalar;

    /// Should return a `(ptr, len)` pair indicating the memory region starts at location `ptr` and
    /// is `len * size_of<Item>` bytes long.
    fn as_source(&self) -> (DevicePtr<Self::Item>, usize);
}

unsafe impl<S: ?Sized + CopySource> CopySource for &S {
    type Item = S::Item;

    fn as_source(&self) -> (DevicePtr<S::Item>, usize) {
        (&**self).as_source()
    }
}

unsafe impl<T: Scalar> CopySource for [T] {
    type Item = T;

    fn as_source(&self) -> (DevicePtr<T>, usize) {
        let ptr = DevicePtr::new(self.as_ptr() as CUdeviceptr);
        (ptr, self.len())
    }
}

unsafe impl<T: Scalar> CopySource for Vec<T> {
    type Item = T;

    fn as_source(&self) -> (DevicePtr<T>, usize) {
        self.as_slice().as_source()
    }
}

unsafe impl<T: Scalar> CopySource for Box<T> {
    type Item = T;

    fn as_source(&self) -> (DevicePtr<T>, usize) {
        let ptr = DevicePtr::new(&**self as *const T as CUdeviceptr);
        (ptr, 1)
    }
}

unsafe impl<T: Scalar> CopySource for DeviceSlice<'_, T> {
    type Item = T;

    fn as_source(&self) -> (DevicePtr<T>, usize) {
        (self.as_ptr(), self.len())
    }
}

unsafe impl<T: Scalar> CopySource for DeviceSliceMut<'_, T> {
    type Item = T;

    fn as_source(&self) -> (DevicePtr<T>, usize) {
        (self.as_ptr(), self.len())
    }
}

unsafe impl<T: Scalar> CopySource for DeviceMem<T> {
    type Item = T;

    fn as_source(&self) -> (DevicePtr<T>, usize) {
        (self.as_ptr(), self.len())
    }
}

unsafe impl<T: Scalar> CopySource for PinnedMem<T> {
    type Item = T;

    fn as_source(&self) -> (DevicePtr<T>, usize) {
        (self.as_ptr(), self.len())
    }
}

/// Memory region suitable to be used as copy destination in [`copy`] and [`copy_async`].
///
/// [`copy`]: fn.copy.html
/// [`copy_async`]: fn.copy_async.html
///
/// # Safety
/// This trait should return a memory region that is valid to be used as destination in `cuMemcpy`.
pub unsafe trait CopyDestination {
    type Item;

    /// Should return a `(ptr, len)` pair indicating the memory region starts at location `ptr` and
    /// is `len * size_of<Item>` bytes long.
    fn as_destination(&mut self) -> (DevicePtr<Self::Item>, usize);
}

unsafe impl<S: ?Sized + CopyDestination> CopyDestination for &mut S {
    type Item = S::Item;

    fn as_destination(&mut self) -> (DevicePtr<S::Item>, usize) {
        (**self).as_destination()
    }
}

unsafe impl<T: Scalar> CopyDestination for [T] {
    type Item = T;

    fn as_destination(&mut self) -> (DevicePtr<T>, usize) {
        let ptr = DevicePtr::new(self.as_ptr() as CUdeviceptr);
        (ptr, self.len())
    }
}

unsafe impl<T: Scalar> CopyDestination for Vec<T> {
    type Item = T;

    fn as_destination(&mut self) -> (DevicePtr<T>, usize) {
        self.as_mut_slice().as_destination()
    }
}

unsafe impl<T: Scalar> CopyDestination for Box<T> {
    type Item = T;

    fn as_destination(&mut self) -> (DevicePtr<T>, usize) {
        let ptr = DevicePtr::new(&**self as *const T as CUdeviceptr);
        (ptr, 1)
    }
}

unsafe impl<T: Scalar> CopyDestination for DeviceSliceMut<'_, T> {
    type Item = T;

    fn as_destination(&mut self) -> (DevicePtr<T>, usize) {
        (self.as_ptr(), self.len())
    }
}

unsafe impl<T: Scalar> CopyDestination for DeviceMem<T> {
    type Item = T;

    fn as_destination(&mut self) -> (DevicePtr<T>, usize) {
        (self.as_ptr(), self.len())
    }
}

unsafe impl<T: Scalar> CopyDestination for PinnedMem<T> {
    type Item = T;

    fn as_destination(&mut self) -> (DevicePtr<T>, usize) {
        (self.as_ptr(), self.len())
    }
}
