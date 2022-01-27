//! Memory allocation in CUDA.
use crate::{copy, copy_raw, cuda_call, cuda_check, CopyDestination, CopySource, Error, Result};
use cuda_driver_sys::*;
use std::any::type_name;
use std::convert::TryInto;
use std::ffi::c_void;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::ops::{self};
use std::{fmt, mem, slice};

/// Raw memory pointer which can be accessed on a CUDA device.
///
///
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct DevicePtr<T = ()> {
    pub(crate) ptr: CUdeviceptr,
    pub(crate) phantom: PhantomData<*const T>,
}

unsafe impl<T> Send for DevicePtr<T> {}
unsafe impl<T> Sync for DevicePtr<T> {}
impl<T> Copy for DevicePtr<T> {}
impl<T> Clone for DevicePtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Debug for DevicePtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "cuba::DevicePtr<{}>({:p})",
            type_name::<T>(),
            self.ptr as *const ()
        )
    }
}

impl<T> DevicePtr<T> {
    pub fn new(ptr: CUdeviceptr) -> Self {
        if align_of::<T>() > 0 {
            assert_eq!((ptr as usize) % align_of::<T>(), 0);
        }

        DevicePtr {
            ptr,
            phantom: PhantomData,
        }
    }

    pub fn add(&self, count: usize) -> Self {
        let nbytes = usize::checked_mul(count, size_of::<T>()).expect("integer overflow");
        self.add_bytes(nbytes)
    }

    pub fn add_bytes(&self, nbytes: usize) -> Self {
        let offset = nbytes.try_into().expect("integer overflow");
        let ptr = CUdeviceptr::checked_add(self.ptr, offset).expect("integer overflow");

        DevicePtr {
            ptr,
            phantom: PhantomData,
        }
    }

    pub fn offset(&self, count: isize) -> Self {
        let nbytes = isize::checked_mul(count, size_of::<T>() as isize).expect("integer overflow");
        self.offset_bytes(nbytes)
    }

    pub fn offset_bytes(&self, nbytes: isize) -> Self {
        let ptr = if nbytes > 0 {
            let offset = nbytes.try_into().expect("integer overflow");
            CUdeviceptr::checked_add(self.ptr, offset).expect("integer overflow")
        } else {
            let offset = (-nbytes).try_into().expect("integer overflow");
            CUdeviceptr::checked_sub(self.ptr, offset).expect("integer overflow")
        };

        DevicePtr {
            ptr,
            phantom: PhantomData,
        }
    }

    pub fn offset_from(&self, base: Self) -> usize {
        usize::checked_sub(self.ptr as usize, base.ptr as usize).expect("integer overflow")
    }

    pub fn cast<U>(&self) -> DevicePtr<U> {
        DevicePtr::new(self.raw())
    }

    pub fn raw(&self) -> CUdeviceptr {
        self.ptr
    }

    pub unsafe fn from_raw(ptr: CUdeviceptr) -> Self {
        DevicePtr {
            ptr,
            phantom: PhantomData,
        }
    }
}

/// Memory buffer allocated on a CUDA device.
///
/// Fixed-size buffer of elements of type `T` located on device memory. The buffer is allocated
/// using `cuMemAlloc` and freed using `cuMemFree`.
///
/// See [`Contiguous`] and [`ContiguousMut`] for memory operations on this type.
///
/// [`Contiguous`]: trait.Contiguous.html
/// [`ContiguousMut`]: trait.ContiguousMut.html
pub struct DeviceMem<T> {
    pub(crate) ptr: DevicePtr<T>,
    pub(crate) len: usize,
}

unsafe impl<T> Send for DeviceMem<T> {}
unsafe impl<T> Sync for DeviceMem<T> {}

impl<T> Debug for DeviceMem<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "DeviceMem<{}>({:p}, {})",
            type_name::<T>(),
            self.ptr.raw() as *const (),
            self.len
        )
    }
}

impl<T> Drop for DeviceMem<T> {
    fn drop(&mut self) {
        unsafe {
            cuMemFree_v2(self.ptr.ptr);
        }
    }
}

impl<T: Scalar> DeviceMem<T> {
    /// Allocate buffer of length `len` without initializing the memory.
    ///
    /// This is safe since any bit pattern is valid for `Scalar` types.
    pub fn empty(len: usize) -> Result<Self> {
        unsafe { Self::uninitialized(len) }
    }

    /// Allocate buffer of length `len` and set all elements to zero.
    ///
    /// This is safe since any bit pattern is valid for `Scalar` types.
    pub fn zeroed(len: usize) -> Result<Self> {
        let mut array = Self::empty(len)?;
        array.zero()?;
        Ok(array)
    }

    /// Allocate buffer of length `len` and set all elements to `value`.
    ///
    /// This is safe since raw copying elements is valid for `Scalar` types.
    pub fn filled(len: usize, value: T) -> Result<Self> {
        let mut array = Self::empty(len)?;
        array.fill(value)?;
        Ok(array)
    }

    // Allocate buffer and copy the elements from the given slice.
    ///
    /// This is safe since raw copying elements is valid for `Scalar` types.
    pub fn from_slice(array: &[T]) -> Result<Self> {
        let mut mem = Self::empty(array.len())?;
        copy(array, &mut mem)?;
        Ok(mem)
    }
}

impl<T> DeviceMem<T> {
    /// Allocate buffer of length `len` without initializing the memory.
    ///
    /// # Safety
    /// The buffer must be initialized before reading the contents.
    pub unsafe fn uninitialized(len: usize) -> Result<Self> {
        let size = len * size_of::<T>();
        let raw = cuda_call(|p| cuMemAlloc_v2(p, size))?;
        let ptr = DevicePtr::new(raw);

        Ok(DeviceMem { ptr, len })
    }

    /// Returns the underlying `DevicePtr<T>` and length of this buffer and  consume this object,
    /// preventing the destructor from being run.
    pub fn into_raw(self) -> (DevicePtr<T>, usize) {
        let out = (self.ptr, self.len);
        mem::forget(self);
        out
    }

    /// Create a `DeviceMem<T>` from a `DevicePtr<T>` and length.
    ///
    /// # Safety
    /// * The buffer must have been allocated using `cuMemAlloc`.
    /// * The capacity must be the correct size for the allocation.
    pub unsafe fn from_raw(ptr: DevicePtr<T>, len: usize) -> Self {
        Self { ptr, len }
    }
}

/// Memory buffer allocated on host as paged-locked memory.
///
/// Fixed-size buffer of elements of type `T` located on device memory. The buffer is allocated
/// using `cuMemHostAlloc` and freed using `cuMemFreeHost`.
///
/// See [`Contiguous`] and [`ContiguousMut`] for memory operations on this type.
///
/// [`Contiguous`]: trait.Contiguous.html
/// [`ContiguousMut`]: trait.ContiguousMut.html
pub struct PinnedMem<T> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
}

unsafe impl<T: Send> Send for PinnedMem<T> {}
unsafe impl<T: Sync> Sync for PinnedMem<T> {}

impl<T> Debug for PinnedMem<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("CudaPinnedMem")
            .field(&self.ptr)
            .field(&self.len)
            .finish()
    }
}

impl<T> Drop for PinnedMem<T> {
    fn drop(&mut self) {
        unsafe {
            cuMemFreeHost(self.ptr as *mut c_void);
        }
    }
}

impl<T: Scalar> PinnedMem<T> {
    /// Allocate buffer of length `len` without initializing the memory.
    ///
    /// This is safe since any bit pattern is valid for `Scalar` types.
    pub fn empty(len: usize) -> Result<Self> {
        unsafe { Self::uninitialized(len) }
    }

    /// Allocate buffer of length `len` and set all elements to zero.
    ///
    /// This is safe since any bit pattern is valid for `Scalar` types.
    pub fn zeroed(len: usize) -> Result<Self> {
        let mut mem = Self::empty(len)?;
        mem.zero()?;
        Ok(mem)
    }

    /// Allocate buffer of length `len` and set all elements to `value`.
    ///
    /// This is safe since raw copying elements is valid for `Scalar` types.
    pub fn filled(len: usize, value: T) -> Result<Self> {
        let mut array = Self::empty(len)?;
        array.fill(value)?;
        Ok(array)
    }

    // Allocate buffer and copy the elements from the given slice.
    ///
    /// This is safe since raw copying elements is valid for `Scalar` types.
    pub fn from_slice(array: &[T]) -> Result<Self> {
        let mut mem = Self::empty(array.len())?;
        mem.copy_from_slice(array)?;
        Ok(mem)
    }
}

impl<T> PinnedMem<T> {
    /// Allocate buffer of length `len` without initializing the memory.
    ///
    /// # Safety
    /// The buffer must be initialized before reading the contents.
    pub unsafe fn uninitialized(len: usize) -> Result<Self> {
        let flag = CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP;
        let size = len * size_of::<T>();
        let hptr = cuda_call(|p| cuMemHostAlloc(p, size, flag))?;

        let dptr = cuda_call(|dptr| cuMemHostGetDevicePointer_v2(dptr, hptr, 0))?;
        assert_eq!(hptr as CUdeviceptr, dptr);

        Ok(PinnedMem {
            ptr: hptr as *mut T,
            len,
        })
    }

    /// Returns a host-side memory slice.
    pub fn as_host(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Returns a mutable host-side memory slice.
    pub fn as_host_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// Fixed-size mutable slice of memory accessible to a CUDA device.
///
/// See [`Contiguous`] and [`ContiguousMut`] for memory operations on this type.
///
/// [`Contiguous`]: trait.Contiguous.html
/// [`ContiguousMut`]: trait.ContiguousMut.html
pub struct DeviceSliceMut<'a, T> {
    pub(crate) ptr: DevicePtr<T>,
    pub(crate) len: usize,
    pub(crate) phantom: PhantomData<&'a mut [T]>,
}

impl<T> Debug for DeviceSliceMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "DeviceSliceMut<{}>({:p}, {})",
            type_name::<T>(),
            self.ptr.raw() as *const (),
            self.len
        )
    }
}

impl<'a, T> DeviceSliceMut<'a, T> {
    pub unsafe fn from_raw(ptr: DevicePtr<T>, len: usize) -> DeviceSliceMut<'a, T> {
        DeviceSliceMut {
            ptr,
            len,
            phantom: PhantomData,
        }
    }
}

/// Fixed-size slice of memory accessible to a CUDA device.
///
/// See [`Contiguous`] for memory operations on this type.
///
/// [`Contiguous`]: trait.Contiguous.html
pub struct DeviceSlice<'a, T> {
    pub(crate) ptr: DevicePtr<T>,
    pub(crate) len: usize,
    phantom: PhantomData<&'a [T]>,
}

impl<T> Debug for DeviceSlice<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "cuba::DeviceSlice<{}>({:p}, {})",
            type_name::<T>(),
            self.ptr.raw() as *const (),
            self.len
        )
    }
}

impl<'a, T> DeviceSlice<'a, T> {
    pub unsafe fn from_raw(ptr: DevicePtr<T>, len: usize) -> DeviceSlice<'a, T> {
        DeviceSlice {
            ptr,
            len,
            phantom: PhantomData,
        }
    }
}

pub trait Contiguous<'a>: Sized {
    type Item: 'a;
    fn as_slice(&self) -> DeviceSlice<'a, Self::Item>;

    fn len(&self) -> usize {
        self.as_slice().len
    }

    fn as_ptr(&self) -> DevicePtr<Self::Item> {
        self.as_slice().ptr
    }

    fn size_in_bytes(&self) -> usize {
        self.as_slice().len * size_of::<Self::Item>()
    }

    fn is_empty(&self) -> bool {
        self.as_slice().len == 0
    }

    fn try_slice(&self, range: impl SliceRange) -> Option<DeviceSlice<'a, Self::Item>> {
        let (offset, length) = range.offset_length(self.len())?;

        Some(unsafe { DeviceSlice::from_raw(self.as_ptr().add(offset), length) })
    }

    fn slice(&self, range: impl SliceRange) -> DeviceSlice<'a, Self::Item> {
        if let Some(s) = self.try_slice(&range) {
            return s;
        }

        panic!(
            "range {:?} is out of bounds for slice of length {}",
            range,
            self.len()
        );
    }

    fn split_at(&self, index: usize) -> (DeviceSlice<'a, Self::Item>, DeviceSlice<'a, Self::Item>) {
        let (ptr, len) = (self.as_ptr(), self.len());
        if index > len {
            panic!(
                "index {} is out of bounds for slice of length {}",
                index, len
            );
        }

        unsafe {
            (
                DeviceSlice::from_raw(ptr, index),
                DeviceSlice::from_raw(ptr.add(index), len - index),
            )
        }
    }

    /// Copy data from this buffer into another buffer. Alias for [`copy(self, other)`].
    ///
    /// [`copy(other, self)`]: ../copy/fn.copy.html
    fn copy_to<D>(&self, other: D) -> Result
    where
        D: CopyDestination<Item = Self::Item>,
        Self::Item: Scalar,
    {
        copy(self.as_slice(), other)
    }

    /// Copy data from this buffer into a newly allocated `Vec`.
    ///
    /// [`copy(other, self)`]: ../copy/fn.copy.html
    fn copy_to_vec(&self) -> Result<Vec<Self::Item>>
    where
        Self::Item: Scalar,
    {
        let n = self.len();
        let mut v = Vec::with_capacity(n);

        unsafe {
            copy_raw(
                self.as_ptr(),
                DevicePtr::from_raw(v.as_mut_ptr() as CUdeviceptr),
                n,
            )?;
            v.set_len(n)
        }

        Ok(v)
    }
}

pub trait ContiguousMut<'a>: Contiguous<'a> {
    fn as_slice_mut(&mut self) -> DeviceSliceMut<'a, Self::Item>;

    /// Returns a
    ///
    /// # Panics
    /// if the given range is out of bounds.
    ///
    /// ```
    /// # use cuba::*;
    /// // Allocate an array of 150 integer.
    /// let mut mem: DeviceMem<u32> = DeviceMem::empty(150)?;
    ///
    /// // Create slice to first 50 integers and set their value to 1.
    /// mem.slice_mut(..50).fill(1);
    ///
    /// // Create slice to middle 50 integers and set their value to 0.
    /// mem.slice_mut(50..100).zero();
    /// ```
    fn slice_mut(&mut self, range: impl SliceRange) -> DeviceSliceMut<'a, Self::Item> {
        let len = self.len();
        if let Some(s) = self.try_slice_mut(&range) {
            return s;
        }

        panic!(
            "range {:?} is out of bounds for slice of length {}",
            range, len
        );
    }

    /// Similar to [`slice_mut`], expect it returns `None` if the range is out of bounds instead of
    /// panicking.
    ///
    /// [`slice_mut`]: #method.slice_mut
    fn try_slice_mut(&mut self, range: impl SliceRange) -> Option<DeviceSliceMut<'a, Self::Item>> {
        let (offset, length) = range.offset_length(self.len())?;

        Some(unsafe { DeviceSliceMut::from_raw(self.as_ptr().add(offset), length) })
    }

    ///
    fn split_at_mut(
        &mut self,
        index: usize,
    ) -> (
        DeviceSliceMut<'a, Self::Item>,
        DeviceSliceMut<'a, Self::Item>,
    ) {
        let (ptr, len) = (self.as_ptr(), self.len());
        if index > len {
            panic!(
                "index {} is out of bounds for slice of length {}",
                index, len
            );
        }

        unsafe {
            (
                DeviceSliceMut::from_raw(ptr, index),
                DeviceSliceMut::from_raw(ptr.add(index), len - index),
            )
        }
    }

    /// Copy data from another buffer into this buffer. Alias for [`copy(other, self)`].
    ///
    /// [`copy(other, self)`]: ../copy/fn.copy.html
    fn copy_from<S>(&mut self, other: S) -> Result
    where
        S: CopySource<Item = Self::Item>,
        Self::Item: Scalar,
    {
        copy(other, self.as_slice_mut())
    }

    /// Copy data from a slice into this buffer. Alias for [`copy(other, self)`].
    ///
    /// [`copy(other, self)`]: ../copy/fn.copy.html
    fn copy_from_slice(&mut self, other: &[Self::Item]) -> Result
    where
        Self::Item: Scalar,
    {
        copy(other, self.as_slice_mut())
    }

    /// Set all elements of this buffer to zero. This is safe since any bit pattern is valid for
    /// `Scalar` types.
    fn zero(&mut self) -> Result
    where
        Self::Item: Scalar,
    {
        unsafe { self.memset(0) }
    }

    /// Set all elements in this buffer to `value`. This is safe since raw copying elements is valid for Scalar types.
    fn fill(&mut self, value: Self::Item) -> Result
    where
        Self::Item: Scalar,
    {
        use std::ptr;
        let item_size = size_of::<Self::Item>();
        let item_ptr = &value as *const Self::Item;
        let (dest_ptr, size_bytes) = (self.as_ptr().raw(), self.size_in_bytes());

        // 1 byte fill
        if item_size == 1 {
            unsafe {
                let value = ptr::read_volatile(item_ptr as *const [u8; 1]);
                cuda_check(cuMemsetD8_v2(
                    dest_ptr,
                    u8::from_ne_bytes(value),
                    size_bytes,
                ))
            }
        }
        // 2 byte fill
        else if item_size == 2 {
            unsafe {
                let value = ptr::read_volatile(item_ptr as *const [u8; 2]);
                cuda_check(cuMemsetD16_v2(
                    dest_ptr,
                    u16::from_ne_bytes(value),
                    size_bytes,
                ))
            }
        }
        // 4 byte fill
        else if item_size == 4 {
            unsafe {
                let value = ptr::read_volatile(item_ptr as *const [u8; 4]);
                cuda_check(cuMemsetD32_v2(
                    dest_ptr,
                    u32::from_ne_bytes(value),
                    size_bytes,
                ))
            }
        }
        // fallback: 1 byte fill if all bytes are the same
        else {
            let mut all_same = false;
            let byte;

            unsafe {
                byte = ptr::read_volatile(item_ptr as *const u8);
                for i in 1..item_size {
                    all_same &= byte == ptr::read_volatile((item_ptr as *const u8).add(i));
                }
            }

            if all_same {
                unsafe { cuda_check(cuMemsetD8_v2(dest_ptr, byte, size_bytes)) }
            } else {
                // unfortunately there is no fill function for >4 byte values, return an error
                // TODO: may implement fallback by copying data from host?
                Error::new(CUresult::CUDA_ERROR_INVALID_VALUE)
            }
        }
    }

    /// Set all bytes in this memory region to the given byte value.
    ///
    /// # Safety
    /// The caller must ensure that writing raw bytes results in a valid value for `T`.
    unsafe fn memset(&mut self, value: u8) -> Result {
        let (ptr, size_bytes) = (self.as_ptr().raw(), self.size_in_bytes());
        cuda_check(cuMemsetD8_v2(ptr, value, size_bytes))
    }
}

impl<'a, T: 'a> Contiguous<'a> for DeviceSliceMut<'a, T> {
    type Item = T;

    fn as_slice(&self) -> DeviceSlice<'a, T> {
        unsafe { DeviceSlice::from_raw(self.ptr, self.len) }
    }
}

impl<'a, T: 'a> ContiguousMut<'a> for DeviceSliceMut<'a, T> {
    fn as_slice_mut(&mut self) -> DeviceSliceMut<'a, T> {
        unsafe { DeviceSliceMut::from_raw(self.ptr, self.len) }
    }
}

impl<'a, T: 'a> Contiguous<'a> for DeviceSlice<'a, T> {
    type Item = T;

    fn as_slice(&self) -> DeviceSlice<'a, T> {
        unsafe { DeviceSlice::from_raw(self.ptr, self.len) }
    }
}

impl<'a, T: 'a> Contiguous<'a> for DeviceMem<T>
where
    Self: 'a,
{
    type Item = T;

    fn as_slice(&self) -> DeviceSlice<'a, T> {
        unsafe { DeviceSlice::from_raw(self.ptr, self.len) }
    }
}

impl<'a, T: 'a> ContiguousMut<'a> for DeviceMem<T>
where
    Self: 'a,
{
    fn as_slice_mut(&mut self) -> DeviceSliceMut<'a, T> {
        unsafe { DeviceSliceMut::from_raw(self.ptr, self.len) }
    }
}

impl<'a, T: 'a> Contiguous<'a> for PinnedMem<T>
where
    Self: 'a,
{
    type Item = T;

    fn as_slice(&self) -> DeviceSlice<'a, T> {
        unsafe { DeviceSlice::from_raw(DevicePtr::from_raw(self.ptr as CUdeviceptr), self.len) }
    }
}

impl<'a, T: 'a> ContiguousMut<'a> for PinnedMem<T>
where
    Self: 'a,
{
    fn as_slice_mut(&mut self) -> DeviceSliceMut<'a, T> {
        unsafe { DeviceSliceMut::from_raw(DevicePtr::from_raw(self.ptr as CUdeviceptr), self.len) }
    }
}

impl<T> ops::Deref for PinnedMem<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_host()
    }
}

impl<T> ops::DerefMut for PinnedMem<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_host_mut()
    }
}

impl<T> AsRef<[T]> for PinnedMem<T> {
    fn as_ref(&self) -> &[T] {
        self.as_host()
    }
}

impl<T> AsMut<[T]> for PinnedMem<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_host_mut()
    }
}

/// Represents a type which can be safely copied back-and-forth between host and device memory
/// without leading to memory safety violations.
///
/// # Safety
/// * The type must be `Copy` (No Drop implementation).
/// * The type must `Send` and `Sync` (Possible to copy between threads via device memory).
/// * Any possible bit-pattern must be valid value. This allows methods such as `Device::empty`
///   to exist which allocate memory without initializing. It also means copy data from device
///   memory to host memory is always safe even for faulty kernels which corrupt memory. This
///   restriction is the reason why `bool` and `char` are not `Scalar`.
///
///
pub unsafe trait Scalar: Copy + Send + Sync + Sized + 'static {}

unsafe impl<T: Scalar> Scalar for DevicePtr<T> {}

macro_rules! impl_scalar_type {
    (@prim $($t:ty)*) => {
        $(
            unsafe impl Scalar for $t { }
        )*
    };
    (@array $($n:expr)*) => {
        $(
            unsafe impl <T: Scalar> Scalar for [T; $n] { }
        )*
    };
    (@tuple $($letters:ident)+) => {
        unsafe impl <$($letters : Scalar,)*> Scalar for ($($letters,)*) {

        }

        impl_scalar_type!(@tuple_next  $($letters)* | );
    };
    (@tuple) => {
        // no nop
    };
    (@tuple_next $head:ident $($rest:ident)+ | $($letters:ident)*) => {
        impl_scalar_type!(@tuple_next $($rest)* | $($letters)* $head);
    };
    (@tuple_next $head:ident | $($letters:ident)* ) => {
        impl_scalar_type!(@tuple $($letters)*);
    };
}

impl_scalar_type!(@prim i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64);
impl_scalar_type!(@prim float2 double2);
impl_scalar_type!(@array 0 1 2 3 4 5 6 7 8);
impl_scalar_type!(@tuple A B C D E F G H);

/// Valid slice range to be used in [`slice`], [`slice_mut`], [`try_slice`], and [`try_slice_mut`].
///
/// [`slice`]: trait.Contiguous.html#method.slice
/// [`try_slice`]: trait.Contiguous.html#method.try_slice
/// [`slice_mut`]: trait.ContiguousMut.html#method.slice_mut
/// [`try_slice_mut`]: trait.ContiguousMut.html#method.try_slice_mut
///
/// * Range: `from..to`
/// * RangeFrom: `from..`
/// * RangeTo: `..to`
/// * RangeFull: `..`
/// * RangeInclusive: `from..=to`
/// * RangeToInclusive: `..=to`
///
pub unsafe trait SliceRange: Debug {
    #[doc(hidden)]
    fn offset_length(&self, len: usize) -> Option<(usize, usize)>;
}

unsafe impl<R: SliceRange> SliceRange for &R {
    fn offset_length(&self, len: usize) -> Option<(usize, usize)> {
        (&**self).offset_length(len)
    }
}

unsafe impl SliceRange for ops::RangeFull {
    fn offset_length(&self, len: usize) -> Option<(usize, usize)> {
        Some((0, len))
    }
}

unsafe impl SliceRange for ops::RangeFrom<usize> {
    fn offset_length(&self, len: usize) -> Option<(usize, usize)> {
        if self.start <= len {
            Some((self.start, len - self.start))
        } else {
            None
        }
    }
}

unsafe impl SliceRange for ops::RangeTo<usize> {
    fn offset_length(&self, len: usize) -> Option<(usize, usize)> {
        if self.end <= len {
            Some((0, self.end))
        } else {
            None
        }
    }
}

unsafe impl SliceRange for ops::RangeInclusive<usize> {
    fn offset_length(&self, len: usize) -> Option<(usize, usize)> {
        // Why are start and end not public like the other ranges? Fix your game notch!
        let (&start, &end_inc) = (self.start(), self.end());
        let end = usize::checked_add(end_inc, 1)?;

        if start <= end && end <= len {
            Some((start, end - start))
        } else {
            None
        }
    }
}

unsafe impl SliceRange for ops::RangeToInclusive<usize> {
    fn offset_length(&self, len: usize) -> Option<(usize, usize)> {
        let end = usize::checked_add(self.end, 1)?;

        if end <= len {
            Some((0, end))
        } else {
            None
        }
    }
}

unsafe impl SliceRange for ops::Range<usize> {
    fn offset_length(&self, len: usize) -> Option<(usize, usize)> {
        if self.start <= self.end && self.end <= len {
            Some((self.start, self.end - self.start))
        } else {
            None
        }
    }
}
