use super::domain::Shape;
use super::{Array, ArrayView, CudaKernel, CudaKernelBuilder};
use crate::api::{ArrayMeta, Event};
use crate::driver::DriverHandle;
use crate::planner::distribution::{IntoDataDistribution, ReplicateDist};
use crate::planner::{Planner, PlannerStage};
use crate::prelude::*;
use crate::types::{Affine, ChunkLayoutBuilder, HasDataType, One, SystemInfo, Zero};
use std::marker::PhantomData;
use std::sync::{Arc, Weak};

#[derive(Debug, Default)]
pub(crate) struct State {}

/// Lightning runtime context.
#[derive(Clone, Debug)]
pub struct Context {
    pub(crate) driver: DriverHandle,
    pub(crate) planner: Weak<Mutex<Planner>>,
}

impl Context {
    pub(crate) fn with_planner<F, T>(&self, fun: F) -> Result<T>
    where
        F: FnOnce(&DriverHandle, &mut Planner) -> Result<T>,
    {
        let planner = self
            .planner
            .upgrade()
            .ok_or_else(|| anyhow!("planner was lost"))?;
        let mut guard = planner.lock();

        fun(&self.driver, &mut guard)
    }

    pub(crate) fn submit<F>(&self, fun: F) -> Result<Event>
    where
        F: FnOnce(&mut PlannerStage) -> Result,
    {
        self.with_planner(|driver, planner| {
            let future = planner.submit_stage(driver, fun)?;

            Ok(Event { future })
        })
    }

    pub fn system(&self) -> &SystemInfo {
        self.driver.system()
    }

    pub fn inner(&self) -> &DriverHandle {
        &self.driver
    }

    /// Synchronize all with workers. This method will block until all work submitted to the
    /// workers has completed.
    pub fn synchronize(&self) -> Result {
        self.driver.synchronize()
    }

    pub fn compile_kernel(&self, mut def: CudaKernelBuilder) -> Result<CudaKernel> {
        def.compile(self)
    }

    pub fn array<T, D>(&self, data: &[T], distribution: D) -> Result<Array<T>>
    where
        T: HasDataType,
        D: IntoDataDistribution,
    {
        let array = self.empty(data.len() as u64, distribution)?;
        array.copy_from(&data)?;
        Ok(array)
    }

    pub fn scalar<T>(&self, data: T) -> Result<Array<T>>
    where
        T: HasDataType,
    {
        let array = self.empty((), ReplicateDist::new())?;
        array.fill(data)?;
        Ok(array)
    }

    /// Allocates a new array with uninitialized contents (i.e., an 'empty' array).
    ///
    /// The dimensionality is given by `shape` while `distribution` specifies how the contents of
    /// the array will be distributed among the workers.
    pub fn empty_with_layout<T, S, D>(
        &self,
        shape: S,
        distribution: D,
        layout: ChunkLayoutBuilder,
    ) -> Result<Array<T>>
    where
        T: HasDataType,
        S: Shape,
        D: IntoDataDistribution,
    {
        let size = shape.into_dim();
        let (dist, chunks) = distribution.into_data_distribution(self.system(), size)?;

        let id = self.with_planner(|driver, planner| {
            planner.create_array(
                driver,
                size,
                T::data_type(),
                &chunks,
                Arc::clone(&dist),
                layout.clone(),
            )
        })?;

        let handle = Arc::new(ArrayMeta {
            id,
            dtype: T::data_type(),
            size,
            distribution: dist,
            context: Context::clone(self),
            layout,
        });

        Ok(Array {
            inner: ArrayView {
                handle,
                domain: size.to_bounds(),
                transform: Affine::identity(),
            },
            phantom: PhantomData,
        })
    }

    /// Allocates a new array with uninitialized contents (i.e., an 'empty' array).
    ///
    /// The dimensionality is given by `shape` while `distribution` specifies how the contents of
    /// the array will be distributed among the workers.
    pub fn empty<T, S, D>(&self, shape: S, distribution: D) -> Result<Array<T>>
    where
        T: HasDataType,
        S: Shape,
        D: IntoDataDistribution,
    {
        self.empty_with_layout(shape, distribution, ChunkLayoutBuilder::row_major())
    }

    /// Allocates a new array initialized with zeros.
    ///
    /// The dimensionality is given by `shape` while `distribution` specifies how the contents of
    /// the array will be distributed among the workers.
    pub fn zeros<T, S, D>(&self, dims: S, dist: D) -> Result<Array<T>>
    where
        T: HasDataType + Zero,
        S: Shape,
        D: IntoDataDistribution,
    {
        self.full(dims, T::zero(), dist)
    }

    /// Allocates a new array initialized with ones.
    ///
    /// The dimensionality is given by `shape` while `distribution` specifies how the contents of
    /// the array will be distributed among the workers.
    pub fn ones<T, S, D>(&self, dims: S, dist: D) -> Result<Array<T>>
    where
        T: HasDataType + One,
        S: Shape,
        D: IntoDataDistribution,
    {
        self.full(dims, T::one(), dist)
    }

    /// Allocates a new array initialized with the given value.
    ///
    /// The dimensionality is given by `shape` while `distribution` specifies how the contents of
    /// the array will be distributed among the workers.
    pub fn full<T, S, D>(&self, dims: S, value: T, dist: D) -> Result<Array<T>>
    where
        T: HasDataType,
        S: Shape,
        D: IntoDataDistribution,
    {
        let array = self.empty(dims, dist)?;
        array.fill(value)?;
        Ok(array)
    }

    /// Allocates a new array with the same shape and distribution as the given array.
    ///
    /// Note that the new array will be uninitialized (i.e., no data will be copied). Use
    /// [`Array::copy`] to also copy the data from the given array.
    pub fn empty_like<T, T2>(&self, array: &Array<T>) -> Result<Array<T2>>
    where
        T2: HasDataType,
    {
        let inner = &array.inner;
        let domain = inner.domain;
        let old_transform = &inner.transform;
        let layout = &inner.handle.layout;

        let region = old_transform.apply_bounds(domain);
        let size = region.extents();

        let (dist, chunks) = inner
            .handle
            .distribution
            .clone_region(&self.system(), region)?;

        let id = self.with_planner(|driver, planner| {
            planner.create_array(
                driver,
                size,
                T2::data_type(),
                &chunks,
                Arc::clone(&dist),
                layout.clone(),
            )
        })?;

        let handle = Arc::new(ArrayMeta {
            id,
            dtype: T2::data_type(),
            size,
            distribution: dist,
            context: Context::clone(self),
            layout: layout.clone(),
        });

        Ok(Array {
            inner: ArrayView {
                handle,
                domain: size.to_bounds(),
                transform: Affine::identity(),
            },
            phantom: PhantomData,
        })
    }

    /// Allocates a new array initialized with the given value. The array will have the same shape
    /// and distribution as the given array.
    pub fn full_like<T>(&self, value: T, array: &Array<T>) -> Result<Array<T>>
    where
        T: HasDataType,
    {
        let result = self.empty_like(array)?;
        result.fill(value)?;
        Ok(result)
    }

    /// Allocates a new zeroed array with the same shape and distribution as the given array.
    pub fn zeros_like<T>(&self, array: &Array<T>) -> Result<Array<T>>
    where
        T: HasDataType + Zero,
    {
        self.full_like(T::zero(), array)
    }

    /// Allocates a new array with the same shape and distribution as the given array.
    pub fn ones_like<T>(&self, array: &Array<T>) -> Result<Array<T>>
    where
        T: HasDataType + One,
    {
        self.full_like(T::one(), array)
    }

    /// Identical to [`empty`] but memory is laid out in column-major order (ie, Fortran order).
    pub fn empty_f<T, S, D>(&self, shape: S, distribution: D) -> Result<Array<T>>
    where
        T: HasDataType,
        S: Shape,
        D: IntoDataDistribution,
    {
        self.empty_with_layout(shape, distribution, ChunkLayoutBuilder::column_major())
    }

    /// Identical to [`full`] but memory is laid out in column-major order (ie, Fortran order).
    pub fn full_f<T, S, D>(&self, dims: S, value: T, dist: D) -> Result<Array<T>>
    where
        T: HasDataType,
        S: Shape,
        D: IntoDataDistribution,
    {
        let array = self.empty_f(dims, dist)?;
        array.fill(value)?;
        Ok(array)
    }

    /// Identical to [`zeros`] but memory is laid out in column-major order (ie, Fortran order).
    pub fn zeros_f<T, S, D>(&self, dims: S, dist: D) -> Result<Array<T>>
    where
        T: HasDataType + Zero,
        S: Shape,
        D: IntoDataDistribution,
    {
        self.full_f(dims, T::zero(), dist)
    }

    /// Identical to [`ones`] but memory is laid out in column-major order (ie, Fortran order).
    pub fn ones_f<T, S, D>(&self, dims: S, dist: D) -> Result<Array<T>>
    where
        T: HasDataType + One,
        S: Shape,
        D: IntoDataDistribution,
    {
        self.full_f(dims, T::one(), dist)
    }
}
