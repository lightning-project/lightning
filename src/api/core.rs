use super::{Context, Event};
use crate::planner::{distribution::DataDistribution, ArrayId};
use crate::prelude::*;
use crate::types::{
    Affine, ChunkLayoutBuilder, DataType, DataValue, Dim, ExecutorId, HostMutAccessor, MemoryId,
    Permutation, Point, Rect, MAX_DIMS,
};
use std::fmt::{self, Debug};
use std::sync::Arc;

pub(crate) struct ArrayMeta {
    pub(crate) id: ArrayId,
    pub(crate) dtype: DataType,
    pub(crate) size: Dim,
    pub(crate) distribution: Arc<dyn DataDistribution>,
    pub(crate) context: Context,
    pub(crate) layout: ChunkLayoutBuilder,
}

impl Drop for ArrayMeta {
    fn drop(&mut self) {
        let result = self
            .context
            .with_planner(|driver, planner| planner.delete_array(driver, self.id));

        if let Err(e) = result {
            warn!("error while deleting array: {}", e);
        }
    }
}

#[derive(Clone)]
pub struct ArrayView {
    pub(crate) handle: Arc<ArrayMeta>,
    pub(crate) domain: Rect,
    pub(crate) transform: Affine,
}

impl Debug for ArrayView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = &self.handle;
        f.debug_struct("ArrayView")
            .field("id", &inner.id)
            .field("dtype", &inner.dtype)
            .field("domain", &self.domain)
            .field("transform", &self.transform)
            .finish()
    }
}

impl ArrayView {
    pub fn id(&self) -> ArrayId {
        self.handle.id
    }

    pub fn data_type(&self) -> DataType {
        self.handle.dtype
    }

    pub fn permutate_axes(&self, perm: Permutation) -> Self {
        Self {
            handle: Arc::clone(&self.handle),
            domain: perm.apply_bounds(self.domain),
            transform: Affine::combine(&Affine::from_permutation(perm.invert()), &self.transform),
        }
    }

    pub fn shift_axis_to_end(&self, axis: usize) -> Self {
        assert!(axis < MAX_DIMS);
        let p = Permutation::with_axis_removed(axis);
        self.permutate_axes(p)
    }

    pub fn swap_axes(&self, a: usize, b: usize) -> Self {
        assert!(a < MAX_DIMS && b < MAX_DIMS);
        let p = Permutation::with_axes_swapped(a, b);
        self.permutate_axes(p)
    }

    pub fn restrict_axis(&self, axis: usize, start: u64, end: u64) -> Self {
        self.limit_axis(axis, start, end, false)
    }

    pub fn slice_axis(&self, axis: usize, start: u64, end: u64) -> Self {
        self.limit_axis(axis, start, end, true)
    }

    fn limit_axis(&self, axis: usize, start: u64, end: u64, shift_to_zero: bool) -> Self {
        let (mut lo, mut hi) = (self.domain.low(), self.domain.high());
        assert!(axis < MAX_DIMS);
        assert!(
            lo[axis] <= start && start <= end && end <= hi[axis],
            "slice range {}..{} not within array domain {}..{}",
            start,
            end,
            lo[axis],
            hi[axis]
        );

        let transform = if shift_to_zero {
            let mut offset = Point::zeros();
            offset[axis] = start;

            lo[axis] = 0;
            hi[axis] = end - start;

            Affine::combine(&Affine::add_offset(offset), &self.transform)
        } else {
            lo[axis] = start;
            hi[axis] = end;

            self.transform.clone()
        };

        Self {
            handle: Arc::clone(&self.handle),
            domain: Rect::from_bounds(lo, hi),
            transform,
        }
    }

    pub fn collapse_axis(&self, axis: usize, index: u64) -> Self {
        self.slice_axis(axis, index, index + 1)
            .shift_axis_to_end(axis)
    }

    pub fn affinity(&self, p: Point) -> Option<ExecutorId> {
        if let Some(d) = self.handle.distribution.as_work_distribution() {
            let q = self.transform.apply_point(p);
            Some(d.query_point(q))
        } else {
            None
        }
    }

    pub fn assign_to(&self, target: &ArrayView) -> Result<Event> {
        if self.data_type() != target.data_type() {
            bail!(
                "data type mismatch: {:?} != {:?}",
                self.data_type(),
                target.data_type()
            );
        }

        let src_transform = self
            .transform
            .to_regular()
            .ok_or_else(|| anyhow!("array transform is not regular"))?;
        let dst_transform = target
            .transform
            .to_regular()
            .ok_or_else(|| anyhow!("array transform is not regular"))?;

        if src_transform.axes() != dst_transform.axes() {
            bail!("axes of transform do not match");
        }

        let src_region = src_transform.apply_bounds(self.domain);
        let dst_region = dst_transform.apply_bounds(target.domain);

        self.context().submit(|stage| {
            stage.add_copy(
                self.handle.id,
                src_region.low(),
                target.handle.id,
                dst_region.low(),
                src_region.extents(),
            )
        })
    }

    pub fn context(&self) -> Context {
        Context::clone(&self.handle.context)
    }

    pub fn assign_from(&self, source: &ArrayView) -> Result<Event> {
        source.assign_to(self)
    }

    pub fn element(&self, p: Point) -> ArrayView {
        assert!(self.domain.contains_point(p));
        let offset = self.transform.apply_point(p);

        Self {
            handle: Arc::clone(&self.handle),
            domain: Dim::one().to_bounds(),
            transform: Affine::add_offset(offset),
        }
    }

    pub fn set(&self, p: Point, value: DataValue) -> Result<Event> {
        assert_eq!(value.data_type(), self.data_type());
        self.element(p).fill(value)
    }

    pub fn get(&self, p: Point) -> Result<DataValue> {
        let dtype = self.data_type();
        let mut data = vec![0; dtype.size_in_bytes()];
        let output =
            HostMutAccessor::from_buffer_raw(data.as_mut_ptr(), data.len(), Dim::one(), dtype);
        let region = Rect::new(p, Dim::one());

        self.context().with_planner(|driver, planner| unsafe {
            planner.read_array(driver, self.handle.id, region, output)
        })?;

        Ok(DataValue::from_raw_data(&data, dtype))
    }

    pub fn barrier(&self) -> Result<Event> {
        let region = self.transform.apply_bounds(self.domain);

        self.context()
            .submit(|stage| stage.add_sync(self.handle.id, region))
    }

    pub fn regions(&self) -> Result<Vec<(MemoryId, Rect)>> {
        let dist = self
            .handle
            .distribution
            .as_work_distribution()
            .ok_or_else(|| anyhow!("array has no preferred work distribution"))?;

        let m = self
            .transform
            .to_regular()
            .ok_or_else(|| anyhow!("array transform is not invertible"))?;

        let bounds = m.apply_bounds(self.domain);
        let subregions = dist
            .query_region(bounds)
            .into_iter()
            .map(|(affinity, subregion)| {
                let affinity = affinity.best_affinity_memory();
                let subregion = m.inverse_bounds(subregion, self.domain);
                (affinity, subregion)
            })
            .collect();

        Ok(subregions)
    }

    pub fn fill(&self, value: DataValue) -> Result<Event> {
        if value.data_type() != self.data_type() {
            bail!(
                "data type mismatch: {:?} != {:?}",
                value.data_type(),
                self.data_type()
            );
        }

        let transform = self
            .transform
            .to_regular()
            .ok_or_else(|| anyhow!("array transform is not regular"))?;
        let region = transform.apply_bounds(self.domain);

        self.context()
            .submit(|stage| stage.add_fill(self.handle.id, region, value))
    }
}
