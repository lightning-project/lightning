use crate::api::{ArrayIndex, ArraySlice, ArrayView, Context, Event, Shape, SliceDescriptor};
use crate::prelude::*;
use crate::types::{
    Dim, HasDataType, HostAccessor, HostMutAccessor, MemoryId, One, Permutation, Rect, Zero,
    MAX_DIMS,
};
use std::collections::Bound;
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::ops::RangeBounds;
use std::vec::IntoIter;

#[derive(Clone)]
pub struct Array<T> {
    pub(crate) inner: ArrayView,
    pub(crate) phantom: PhantomData<T>,
}

impl<T> Debug for Array<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = &self.inner.handle;
        f.debug_struct("Array")
            .field("id", &inner.id)
            .field("dtype", &inner.dtype)
            .field("domain", &self.inner.domain)
            .field("transform", &self.inner.transform)
            .finish()
    }
}

unsafe impl<T> Send for Array<T> {}
unsafe impl<T> Sync for Array<T> {}

impl<T> Array<T> {
    pub fn context(&self) -> &Context {
        &self.inner.handle.context
    }

    pub fn is_empty(&self) -> bool {
        self.inner.domain.is_empty()
    }
}

impl<T> Array<T>
where
    T: HasDataType,
{
    pub fn copy(&self) -> Result<Self> {
        let x = self.context().empty_like(&self)?;
        self.assign_to(&x)?;
        Ok(x)
    }

    pub fn assign_to(&self, target: &Array<T>) -> Result<Event> {
        self.inner.assign_to(&target.inner)
    }

    pub fn assign_from(&self, source: &Array<T>) -> Result<Event> {
        source.inner.assign_to(&self.inner)
    }

    pub fn fill(&self, value: T) -> Result<Event> {
        self.inner.fill(T::into(value))
    }

    pub fn fill_zeros(&self) -> Result<Event>
    where
        T: Zero,
    {
        self.fill(T::zero())
    }

    pub fn fill_ones(&self) -> Result<Event>
    where
        T: One,
    {
        self.fill(T::one())
    }

    pub fn set<I: ArrayIndex>(&self, indices: I, value: T) -> Result<Event> {
        let p = indices.to_point();
        self.inner.set(p, value.into())
    }

    pub fn get<I: ArrayIndex>(&self, indices: I) -> Result<T> {
        let p = indices.to_point();
        let value = self.inner.get(p)?;
        Ok(value.try_into().unwrap())
    }

    pub fn copy_from(&self, data: &[T]) -> Result<Event> {
        assert_eq!(data.len() as u64, self.len());
        let region = self.inner.transform.apply_bounds(self.inner.domain);
        let buffer = HostAccessor::from_buffer(data.as_ptr(), data.len(), region.extents());

        let future = self.context().with_planner(|driver, planner| unsafe {
            planner.write_array(driver, self.inner.handle.id, region, buffer)
        })?;

        Ok(Event { future })
    }

    pub fn copy_to(&self, data: &mut [T]) -> Result {
        assert_eq!(data.len() as u64, self.len());
        assert_eq!(T::data_type(), self.inner.handle.dtype);
        let region = self.inner.transform.apply_bounds(self.inner.domain);

        let data = HostMutAccessor::from_buffer(data.as_mut_ptr(), data.len(), region.extents());

        self.context().with_planner(|driver, planner| unsafe {
            planner.read_array(driver, self.inner.handle.id, region, data)
        })
    }

    pub fn to_vec(&self) -> Result<Vec<T>> {
        let mut data = vec![default(); self.len() as usize];
        self.copy_to(&mut data)?;
        Ok(data)
    }
}

impl<T> Array<T> {
    pub fn domain(&self) -> Rect {
        self.inner.domain
    }

    pub fn extents(&self) -> Dim {
        self.domain().extents()
    }

    pub fn shape<S: Shape>(&self) -> S {
        S::from_dim(self.extents())
    }

    pub fn synchronize(&self) -> Result {
        self.inner.barrier()?.wait()?;
        Ok(())
    }

    pub fn affinity<I: ArrayIndex>(&self, indices: I) -> Option<MemoryId> {
        self.inner
            .affinity(indices.to_point())
            .map(|e| e.best_affinity_memory())
    }

    pub fn element<I: ArrayIndex>(&self, indices: I) -> Array<T> {
        let p = indices.to_point();
        let mut inner = self.inner.clone();

        for axis in 0..MAX_DIMS {
            inner = inner.slice_axis(axis, p[axis], p[axis] + 1);
        }

        Array {
            inner,
            phantom: PhantomData,
        }
    }

    pub fn swap_axes(&self, a: usize, b: usize) -> Self {
        assert!(a < MAX_DIMS && b < MAX_DIMS);

        Self {
            inner: self.inner.swap_axes(a, b),
            phantom: PhantomData,
        }
    }

    pub fn permutate_axes(&self, permutation: Permutation) -> Self {
        Self {
            inner: self.inner.permutate_axes(permutation),
            phantom: PhantomData,
        }
    }

    pub fn insert_axis(&self, axis: usize) -> Array<T> {
        assert!(axis < MAX_DIMS);

        Array {
            inner: self.inner.swap_axes(axis, MAX_DIMS - 1),
            phantom: PhantomData,
        }
    }

    pub fn remove_axis(&self, axis: usize) -> Array<T> {
        self.collapse_axis(axis, 0)
    }

    fn normalize_range<R: RangeBounds<u64>>(&self, axis: usize, range: R) -> (u64, u64) {
        assert!(axis < MAX_DIMS);
        let domain = &self.inner.domain;

        let start = match range.start_bound() {
            Bound::Included(v) => *v,
            Bound::Excluded(v) => *v + 1,
            Bound::Unbounded => domain.low()[axis],
        };

        let end = match range.end_bound() {
            Bound::Included(v) => *v - 1,
            Bound::Excluded(v) => *v,
            Bound::Unbounded => domain.high()[axis],
        };

        (start, end)
    }

    pub fn slice_axis<R>(&self, axis: usize, range: R) -> Array<T>
    where
        R: RangeBounds<u64>,
    {
        let (start, end) = self.normalize_range(axis, range);

        Self {
            inner: self.inner.slice_axis(axis, start, end),
            phantom: PhantomData,
        }
    }

    pub fn restrict_axis<R>(&self, axis: usize, range: R) -> Array<T>
    where
        R: RangeBounds<u64>,
    {
        let (start, end) = self.normalize_range(axis, range);

        Self {
            inner: self.inner.restrict_axis(axis, start, end),
            phantom: PhantomData,
        }
    }

    pub fn collapse_axis(&self, axis: usize, index: u64) -> Array<T> {
        Array {
            inner: self.inner.collapse_axis(axis, index),
            phantom: PhantomData,
        }
    }

    pub fn slice<I: ArraySlice>(&self, indices: I) -> Array<T> {
        let slices = indices.into_slices();
        let slices = slices.as_ref();

        let mut inner = self.inner.clone();

        for (axis, slice) in reversed(enumerate(slices)) {
            inner = match slice {
                &SliceDescriptor::Index(i) => inner.collapse_axis(axis, i),
                &SliceDescriptor::Range(start, end) => {
                    let (start, end) = self.normalize_range(axis, (start, end));
                    inner.slice_axis(axis, start, end)
                }
                SliceDescriptor::NewAxis => {
                    unimplemented!()
                }
            }
        }

        Array {
            inner,
            phantom: PhantomData,
        }
    }

    pub fn restrict<I: ArraySlice>(&self, indices: I) -> Array<T> {
        let slices = indices.into_slices();
        let slices = slices.as_ref();

        let mut inner = self.inner.clone();

        for (axis, slice) in reversed(enumerate(slices)) {
            inner = match slice {
                &SliceDescriptor::Index(i) => inner.collapse_axis(axis, i),
                &SliceDescriptor::Range(start, end) => {
                    let (start, end) = self.normalize_range(axis, (start, end));
                    inner.restrict_axis(axis, start, end)
                }
                SliceDescriptor::NewAxis => {
                    unimplemented!()
                }
            }
        }

        Array {
            inner,
            phantom: PhantomData,
        }
    }

    pub fn regions(&self) -> Result<RegionIterator> {
        Ok(RegionIterator {
            subregions: self.inner.regions()?.into_iter(),
        })
    }

    pub fn len(&self) -> u64 {
        self.inner.domain.extents().volume()
    }

    pub fn nrows(&self) -> u64 {
        self.extents()[0]
    }

    pub fn row(&self, index: u64) -> Self {
        self.collapse_axis(0, index)
    }

    pub fn ncols(&self) -> u64 {
        self.extents()[1]
    }

    pub fn column(&self, index: u64) -> Self {
        self.collapse_axis(1, index)
    }

    pub fn nlayers(&self) -> u64 {
        self.extents()[2]
    }

    pub fn layer(&self, index: u64) -> Self {
        self.collapse_axis(2, index)
    }
}

#[derive(Debug)]
pub struct RegionIterator {
    subregions: IntoIter<(MemoryId, Rect)>,
}

impl Iterator for RegionIterator {
    type Item = (MemoryId, Rect);

    fn next(&mut self) -> Option<Self::Item> {
        self.subregions.next()
    }
}
