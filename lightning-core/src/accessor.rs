use crate::info::DeviceId;
use crate::prelude::{reversed, TryInto};
use crate::{AffineNM, DataType, DimN, HasDataType, RectN, MAX_DIMS};
use lightning_cuda::prelude::*;
use lightning_cuda::DevicePtr;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StridesN<const N: usize> {
    #[serde(with = "serde_arrays")]
    strides: [i64; N],
}

impl<const N: usize> StridesN<N> {
    pub(crate) fn contiguous(size: DimN<u64, N>) -> Self {
        let mut strides = [0; N];

        if !size.is_empty() {
            let mut last_stride = 1;

            for i in reversed(0..N) {
                strides[i] = last_stride;

                let d = size[i].try_into().unwrap();
                last_stride = i64::checked_mul(last_stride, d).unwrap();
            }
        }

        Self { strides }
    }

    #[inline]
    pub fn order(&self) -> [usize; MAX_DIMS] {
        let mut axes = [0; MAX_DIMS];
        for i in 0..N {
            axes[i] = i;
        }

        // Selection sort
        for i in 0..(N - 1) {
            let mut best = i;

            for j in (i + 1)..N {
                if self.strides[axes[j]].abs() < self.strides[axes[best]].abs() {
                    best = j;
                }
            }

            axes.swap(i, best);
        }

        axes
    }

    pub fn to_byte_strides(&self, dtype: DataType) -> ByteStridesN<N> {
        let mut byte_strides = [0; N];
        let elem_size = dtype.size_in_bytes() as i64;

        for i in 0..N {
            byte_strides[i] = elem_size * self.strides[i];
        }

        ByteStridesN::from(byte_strides)
    }
}

pub type Strides = StridesN<MAX_DIMS>;

impl<const N: usize> From<[i64; N]> for StridesN<N> {
    fn from(strides: [i64; N]) -> Self {
        Self { strides }
    }
}

impl<const N: usize> Deref for StridesN<N> {
    type Target = [i64; N];

    fn deref(&self) -> &[i64; N] {
        &self.strides
    }
}

impl<const N: usize> DerefMut for StridesN<N> {
    fn deref_mut(&mut self) -> &mut [i64; N] {
        &mut self.strides
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteStridesN<const N: usize> {
    #[serde(with = "serde_arrays")]
    strides: [i64; N],
}

impl<const N: usize> ByteStridesN<N> {
    #[inline(always)]
    pub fn offset_in_bytes(&self, p: [i64; N]) -> isize {
        let mut offset = 0;
        for i in 0..N {
            offset += p[i] * self.strides[i];
        }
        offset as isize
    }

    pub fn to_usize(&self) -> [usize; MAX_DIMS] {
        let mut result = [0; MAX_DIMS];

        for i in 0..N {
            result[i] = self.strides[i].try_into().unwrap();
        }

        result
    }
}

impl<const N: usize> From<[i64; N]> for ByteStridesN<N> {
    fn from(strides: [i64; N]) -> Self {
        Self { strides }
    }
}

impl<const N: usize> Deref for ByteStridesN<N> {
    type Target = [i64; N];

    fn deref(&self) -> &[i64; N] {
        &self.strides
    }
}

impl<const N: usize> DerefMut for ByteStridesN<N> {
    fn deref_mut(&mut self) -> &mut [i64; N] {
        &mut self.strides
    }
}

pub type ByteStrides = ByteStridesN<MAX_DIMS>;

// These types below may need to move to their own module at some point.
#[derive(Debug, Copy, Clone)]
pub enum UnifiedPtr {
    Host(*const u8),
    HostMut(*mut u8),
    Device(CudaDevicePtr, DeviceId),
    DeviceMut(CudaDevicePtr, DeviceId),
}

impl UnifiedPtr {
    pub fn to_const(&self) -> Self {
        use UnifiedPtr::*;

        match *self {
            Host(ptr) => Host(ptr),
            HostMut(ptr) => Host(ptr),
            Device(ptr, id) => Device(ptr, id),
            DeviceMut(ptr, id) => Device(ptr, id),
        }
    }
}

unsafe impl Send for UnifiedPtr {}
unsafe impl Sync for UnifiedPtr {}

pub trait Data {
    type Ptr;

    fn offset_by_bytes(self, nbytes: i64) -> Self;
    fn as_ptr(&self) -> Self::Ptr;
}

pub trait DataMut: Data {
    type PtrMut;

    fn as_ptr_mut(&self) -> Self::PtrMut;
}

impl Data for *mut u8 {
    type Ptr = *const u8;

    fn offset_by_bytes(self, nbytes: i64) -> Self {
        self.wrapping_offset(nbytes as isize)
    }

    fn as_ptr(&self) -> Self::Ptr {
        *self
    }
}

impl DataMut for *mut u8 {
    type PtrMut = *mut u8;

    fn as_ptr_mut(&self) -> Self::PtrMut {
        *self
    }
}

impl Data for *const u8 {
    type Ptr = *const u8;

    fn offset_by_bytes(self, nbytes: i64) -> Self {
        self.wrapping_offset(nbytes as isize)
    }

    fn as_ptr(&self) -> Self::Ptr {
        *self
    }
}

impl Data for CudaDevicePtr {
    type Ptr = CudaDevicePtr;

    fn offset_by_bytes(self, nbytes: i64) -> Self {
        self.offset_bytes(nbytes as isize)
    }

    fn as_ptr(&self) -> Self::Ptr {
        *self
    }
}

impl DataMut for CudaDevicePtr {
    type PtrMut = CudaDevicePtr;

    fn as_ptr_mut(&self) -> Self::PtrMut {
        *self
    }
}

impl Data for UnifiedPtr {
    type Ptr = UnifiedPtr;

    fn offset_by_bytes(self, nbytes: i64) -> Self {
        use UnifiedPtr::*;
        match self {
            Host(ptr) => Host(ptr.offset_by_bytes(nbytes)),
            HostMut(ptr) => HostMut(ptr.offset_by_bytes(nbytes)),
            Device(ptr, id) => Device(ptr.offset_by_bytes(nbytes), id),
            DeviceMut(ptr, id) => DeviceMut(ptr.offset_by_bytes(nbytes), id),
        }
    }

    fn as_ptr(&self) -> Self::Ptr {
        use UnifiedPtr::*;
        match *self {
            Host(ptr) => Host(ptr),
            HostMut(ptr) => Host(ptr as *const u8),
            Device(ptr, id) => Device(ptr, id),
            DeviceMut(ptr, id) => Device(ptr, id),
        }
    }
}

impl DataMut for UnifiedPtr {
    type PtrMut = UnifiedPtr;

    fn as_ptr_mut(&self) -> Self::PtrMut {
        use UnifiedPtr::*;
        match *self {
            HostMut(ptr) => HostMut(ptr),
            DeviceMut(ptr, id) => DeviceMut(ptr, id),
            _ => panic!("pointer type is not mutable"),
        }
    }
}

type GenericAccessorN<const N: usize> = Accessor<UnifiedPtr, N>;
type HostAccessorN<const N: usize> = Accessor<*const u8, N>;
type HostMutAccessorN<const N: usize> = Accessor<*mut u8, N>;
type CudaAccessorN<const N: usize> = Accessor<CudaDevicePtr, N>;
type CudaMutAccessorN<const N: usize> = Accessor<CudaDevicePtr, N>;

pub type GenericAccessor = GenericAccessorN<MAX_DIMS>;
pub type HostAccessor = HostAccessorN<MAX_DIMS>;
pub type HostMutAccessor = HostMutAccessorN<MAX_DIMS>;
pub type CudaAccessor = CudaAccessorN<MAX_DIMS>;
pub type CudaMutAccessor = CudaMutAccessorN<MAX_DIMS>;

pub type GenericAccessor1 = GenericAccessorN<1>;
pub type HostAccessor1 = HostAccessorN<1>;
pub type HostAccessorMut1 = HostMutAccessorN<1>;
pub type CudaAccessor1 = CudaAccessorN<1>;
pub type CudaMutAccessor1 = CudaMutAccessorN<1>;

pub type GenericAccessor2 = GenericAccessorN<2>;
pub type HostAccessor2 = HostAccessorN<2>;
pub type HostAccessorMut2 = HostMutAccessorN<2>;
pub type CudaAccessor2 = CudaAccessorN<2>;
pub type CudaMutAccessor2 = CudaMutAccessorN<2>;

pub type GenericAccessor3 = GenericAccessorN<3>;
pub type HostAccessor3 = HostAccessorN<3>;
pub type HostAccessorMut3 = HostMutAccessorN<3>;
pub type CudaAccessor3 = CudaAccessorN<3>;
pub type CudaMutAccessor3 = CudaMutAccessorN<3>;

pub type GenericAccessor4 = GenericAccessorN<4>;
pub type HostAccessor4 = HostAccessorN<4>;
pub type HostAccessorMut4 = HostMutAccessorN<4>;
pub type CudaAccessor4 = CudaAccessorN<4>;
pub type CudaMutAccessor4 = CudaMutAccessorN<4>;

unsafe impl<const N: usize> Send for HostAccessorN<N> {}
unsafe impl<const N: usize> Send for HostMutAccessorN<N> {}

#[derive(Debug, Copy, Clone)]
pub struct Accessor<T, const N: usize> {
    pub(crate) strides: StridesN<N>,
    pub(crate) size: DimN<u64, N>,
    pub(crate) data_type: DataType,
    pub(crate) data: T,
}

impl<T, const N: usize> Accessor<T, N> {
    pub unsafe fn new(
        data: T,
        strides: StridesN<N>,
        size: DimN<u64, N>,
        data_type: DataType,
    ) -> Self {
        Self {
            strides,
            size,
            data_type,
            data,
        }
    }

    pub fn extents(&self) -> DimN<u64, N> {
        self.size
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn strides(&self) -> StridesN<N> {
        self.strides
    }

    pub fn strides_in_bytes(&self) -> ByteStridesN<N> {
        self.strides.to_byte_strides(self.data_type)
    }

    pub fn size_in_bytes(&self) -> usize {
        let elem_size = self.data_type.layout().pad_to_align().size();
        self.size.volume() as usize * elem_size
    }

    pub fn to_dim<const M: usize>(self) -> Accessor<T, M> {
        let mut new_size = [1; M];
        let mut new_strides = [0; M];

        for i in 0..N {
            if i < M {
                new_size[i] = self.size[i];
                new_strides[i] = self.strides[i];
            } else {
                assert_eq!(self.size[i], 1);
            }
        }

        Accessor {
            strides: StridesN::from(new_strides),
            size: DimN::from(new_size),
            data_type: self.data_type,
            data: self.data,
        }
    }
}

impl<T: Data + Clone, const N: usize> Accessor<T, N> {
    pub fn split_at(self, axis: usize, index: u64) -> (Self, Self) {
        assert!(index < self.size[axis]);
        let mut left_size = self.size;
        let mut right_size = self.size;
        left_size[axis] = index;
        right_size[axis] -= index;

        let byte_strides = self.strides_in_bytes();
        let byte_offset = index as i64 * byte_strides[axis];

        (
            Self {
                strides: self.strides,
                data_type: self.data_type,
                data: self.data.clone(),
                size: left_size,
            },
            Self {
                strides: self.strides,
                data_type: self.data_type,
                data: self.data.offset_by_bytes(byte_offset),
                size: right_size,
            },
        )
    }

    pub fn slice(self, region: RectN<N>) -> Self {
        assert!(self.size.to_bounds().contains(region));
        let offset = region.low();
        let new_size = region.extents();

        let byte_strides = self.strides_in_bytes();
        let mut byte_offset = 0;
        for i in 0..N {
            byte_offset += offset[i] as i64 * byte_strides[i];
        }

        Self {
            strides: self.strides,
            data_type: self.data_type,
            data: self.data.offset_by_bytes(byte_offset),
            size: new_size,
        }
    }

    pub fn transform<const M: usize>(
        self,
        transform: &AffineNM<M, N>,
        new_size: DimN<u64, M>,
    ) -> Accessor<T, M> {
        let translate = transform.translate();
        let old_strides = self.strides;
        let mut new_strides = [0i64; M];

        if let Some(matrix) = transform.transform() {
            for i in 0..N {
                let mut lo = translate[i];
                let mut hi = lo;

                for j in 0..M {
                    let scale = matrix[i][j];

                    if new_size[j] == 0 {
                        // no-op
                    } else if scale > 0 {
                        hi += scale * (new_size[j] as i64 - 1);
                    } else {
                        lo += scale * new_size[j] as i64;
                    }
                }

                assert!(
                    0 <= lo && lo <= hi && hi as u64 <= self.size[i],
                    "failed: 0 <= {} <= {} <= {} for axis {}",
                    lo,
                    hi,
                    self.size[i],
                    i
                );
            }

            for j in 0..M {
                let mut s = 0;

                for i in 0..N {
                    s += matrix[i][j] * old_strides[i];
                }

                new_strides[j] = s;
            }
        } else {
            for i in 0..N {
                let lo = translate[i];
                let hi = lo + *new_size.get(i).unwrap_or(&0) as i64;

                assert!(
                    0 <= lo && lo <= hi && hi as u64 <= self.size[i],
                    "failed: 0 <= {} <= {} <= {} for axis {}",
                    lo,
                    hi,
                    self.size[i],
                    i
                );
            }

            for i in 0..usize::min(N, M) {
                new_strides[i] = old_strides[i];
            }
        }

        let mut byte_offset = 0;
        let elem_size = self.data_type.size_in_bytes() as i64;
        for i in 0..N {
            byte_offset += translate[i] * old_strides[i] * elem_size;
        }

        Accessor {
            strides: StridesN::from(new_strides),
            data_type: self.data_type,
            data: self.data.offset_by_bytes(byte_offset),
            size: new_size,
        }
    }

    pub fn swap_axes(&self, a: usize, b: usize) -> Self {
        let mut strides = self.strides;
        let mut size = self.size;

        strides.swap(a, b);
        size.swap(a, b);

        Accessor {
            strides,
            size,
            data_type: self.data_type,
            data: self.data.clone(),
        }
    }

    pub fn as_ptr(&self) -> T::Ptr {
        self.data.as_ptr()
    }
}

impl<T: DataMut, const N: usize> Accessor<T, N> {
    pub fn as_ptr_mut(&self) -> T::PtrMut {
        self.data.as_ptr_mut()
    }
}

fn compute_strides<const N: usize>(
    capacity_in_bytes: usize,
    size: DimN<u64, N>,
    data_type: DataType,
) -> StridesN<N> {
    let dsize = data_type.layout().pad_to_align().size();
    assert_eq!(capacity_in_bytes, dsize * size.volume() as usize);

    StridesN::contiguous(size)
}

impl<const N: usize> HostAccessorN<N> {
    pub fn from_buffer<T>(ptr: *const T, capacity: usize, extents: DimN<u64, N>) -> Self
    where
        T: HasDataType,
    {
        let data_type = T::data_type();
        let layout = data_type.layout().pad_to_align();

        Self::from_buffer_raw(
            ptr as *const u8,
            layout.size() * capacity,
            extents,
            data_type,
        )
    }

    pub fn from_buffer_raw(
        data: *const u8,
        capacity_in_bytes: usize,
        size: DimN<u64, N>,
        data_type: DataType,
    ) -> Self {
        Self {
            strides: compute_strides(capacity_in_bytes, size, data_type),
            size,
            data_type,
            data,
        }
    }
}

impl<const N: usize> HostMutAccessorN<N> {
    pub fn from_buffer<T>(ptr: *mut T, capacity: usize, extents: DimN<u64, N>) -> Self
    where
        T: HasDataType,
    {
        let data_type = T::data_type();
        let layout = data_type.layout().pad_to_align();

        Self::from_buffer_raw(ptr as *mut u8, layout.size() * capacity, extents, data_type)
    }

    pub fn from_buffer_raw(
        data: *mut u8,
        capacity_in_bytes: usize,
        size: DimN<u64, N>,
        data_type: DataType,
    ) -> Self {
        Self {
            strides: compute_strides(capacity_in_bytes, size, data_type),
            size,
            data_type,
            data,
        }
    }
}

impl<const N: usize> GenericAccessorN<N> {
    #[inline]
    fn map_ptr<F, P>(&self, map: F) -> Option<Accessor<P, N>>
    where
        F: FnOnce(UnifiedPtr) -> Option<P>,
    {
        if let Some(data) = map(self.data) {
            Some(Accessor {
                data,
                strides: self.strides,
                size: self.size,
                data_type: self.data_type,
            })
        } else {
            None
        }
    }

    pub fn as_host(&self) -> Option<HostAccessorN<N>> {
        self.map_ptr(|ptr| match ptr {
            UnifiedPtr::Host(ptr) => Some(ptr),
            UnifiedPtr::HostMut(ptr) => Some(ptr as *const _),
            _ => None,
        })
    }

    pub fn as_host_mut(&self) -> Option<HostMutAccessorN<N>> {
        self.map_ptr(|ptr| match ptr {
            UnifiedPtr::HostMut(ptr) => Some(ptr),
            _ => None,
        })
    }

    pub fn as_device(&self, id: DeviceId) -> Option<CudaAccessorN<N>> {
        self.map_ptr(|ptr| match ptr {
            UnifiedPtr::Device(ptr, x) if x == id => Some(ptr),
            UnifiedPtr::DeviceMut(ptr, x) if x == id => Some(ptr),
            UnifiedPtr::Host(ptr) => Some(DevicePtr::new(ptr as _)), // CUDA unified mem
            UnifiedPtr::HostMut(ptr) => Some(DevicePtr::new(ptr as _)),
            _ => None,
        })
    }

    pub fn as_device_mut(&self, id: DeviceId) -> Option<CudaAccessorN<N>> {
        self.map_ptr(|ptr| match ptr {
            UnifiedPtr::DeviceMut(ptr, x) if x == id => Some(ptr),
            UnifiedPtr::HostMut(ptr) => Some(DevicePtr::new(ptr as _)),
            _ => None,
        })
    }
}
