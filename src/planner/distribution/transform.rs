use super::IntoWorkDistribution;
use super::{
    ChunkDescriptor, ChunkQueryResult, DataDistribution, IntoDataDistribution, WorkDistribution,
};
use crate::prelude::*;
use crate::types::{Dim, ExecutorId, MemoryId, Permutation, Point, Rect, SystemInfo};
use std::fmt;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Copy, Clone, Debug)]
pub struct PermutationDist<D> {
    permutation: Permutation,
    inner: D,
}

impl<D> PermutationDist<D> {
    pub fn new(permutation: Permutation, inner: D) -> Self {
        Self { permutation, inner }
    }

    pub fn swap_xy(inner: D) -> Self {
        Self::new(Permutation::with_axes_swapped(0, 1), inner)
    }

    pub fn swap_xz(inner: D) -> Self {
        Self::new(Permutation::with_axes_swapped(0, 2), inner)
    }

    pub fn swap_yz(inner: D) -> Self {
        Self::new(Permutation::with_axes_swapped(1, 2), inner)
    }
}

impl<D: IntoDataDistribution> IntoDataDistribution for PermutationDist<D> {
    fn into_data_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)> {
        let p = self.permutation;
        let (inner, mut chunks) = self
            .inner
            .into_data_distribution(system, p.apply_extents(size))?;

        for chunk in &mut chunks {
            chunk.size = p.inverse_extents(chunk.size);
        }

        let dist: Arc<dyn DataDistribution> = if p.is_identity() {
            inner
        } else if p == Permutation::with_axes_swapped(0, 1) {
            Arc::new(PermutationDistribution {
                permutation: move || Permutation::with_axes_swapped(0, 1),
                inner,
            })
        } else if p == Permutation::with_axes_swapped(0, 2) {
            Arc::new(PermutationDistribution {
                permutation: move || Permutation::with_axes_swapped(0, 2),
                inner,
            })
        } else {
            Arc::new(PermutationDistribution {
                permutation: move || p,
                inner,
            })
        };

        Ok((dist, chunks))
    }
}

impl<D: IntoWorkDistribution> IntoWorkDistribution for PermutationDist<D> {
    fn into_work_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<Arc<dyn WorkDistribution>> {
        let p = self.permutation;
        let inner = self
            .inner
            .into_work_distribution(system, p.apply_extents(size))?;

        Ok(Arc::new(PermutationDistribution {
            permutation: move || p,
            inner,
        }))
    }
}

struct PermutationDistribution<P, D: ?Sized> {
    permutation: P,
    inner: Arc<D>,
}

impl<P, D: ?Sized + Debug> Debug for PermutationDistribution<P, D>
where
    P: Fn() -> Permutation,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PermutationDistribution")
            .field("permutation", &(self.permutation)())
            .field("inner", &self.inner)
            .finish()
    }
}

impl<P> WorkDistribution for PermutationDistribution<P, dyn WorkDistribution>
where
    P: Fn() -> Permutation,
{
    fn query_point(&self, x: Point) -> ExecutorId {
        let p = (self.permutation)();
        self.inner.query_point(p.apply_point(x))
    }

    fn query_region(&self, region: Rect) -> Vec<(ExecutorId, Rect)> {
        let p = (self.permutation)();
        let mut results = self.inner.query_region(p.apply_bounds(region));

        for (_, subregion) in &mut results {
            *subregion = p.inverse_bounds(*subregion);
        }

        results
    }
}

impl<P> DataDistribution for PermutationDistribution<P, dyn DataDistribution>
where
    P: Fn() -> Permutation,
    P: Send + Sync + Copy + 'static,
{
    fn clone_region(
        &self,
        system: &SystemInfo,
        region: Rect,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)> {
        let p = (self.permutation)();
        let (dist, mut chunks) = self.inner.clone_region(system, p.apply_bounds(region))?;

        for chunk in &mut chunks {
            chunk.size = p.inverse_extents(chunk.size);
        }

        let dist = Arc::new(PermutationDistribution {
            permutation: self.permutation,
            inner: dist,
        });

        Ok((dist, chunks))
    }

    fn visit_unique(
        &self,
        region: Rect,
        affinity: Option<MemoryId>,
        callback: &mut dyn FnMut(ChunkQueryResult),
    ) {
        let p = (self.permutation)();
        self.inner
            .visit_unique(p.apply_bounds(region), affinity, &mut move |r| {
                callback(ChunkQueryResult {
                    chunk_index: r.chunk_index,
                    chunk_offset: p.inverse_point(r.chunk_offset),
                    region_offset: p.inverse_point(r.region_offset),
                    extents: p.inverse_extents(r.extents),
                })
            })
    }

    fn visit_replicated(&self, region: Rect, callback: &mut dyn FnMut(ChunkQueryResult)) {
        let p = (self.permutation)();
        self.inner
            .visit_replicated(p.apply_bounds(region), &mut move |r| {
                callback(ChunkQueryResult {
                    chunk_index: r.chunk_index,
                    chunk_offset: p.inverse_point(r.chunk_offset),
                    region_offset: p.inverse_point(r.region_offset),
                    extents: p.inverse_extents(r.extents),
                })
            })
    }
}
