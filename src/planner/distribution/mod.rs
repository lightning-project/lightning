//! Traits and utilities for dealing with data distributions.

pub mod centralize;
pub mod columns;
pub mod random;
pub mod replicate;
pub mod rows;
pub mod stencil;
pub mod stencil2d;
pub mod stencil3d;
pub mod tile;
pub mod transform;

use lightning_core::util::AsAny;
use std::fmt::Debug;
use std::sync::Arc;

use crate::prelude::*;
use crate::types::{Dim, ExecutorId, MemoryId, MemoryKind, Point, Rect, SystemInfo, WorkerId};

#[doc(inline)]
pub use centralize::CentralizeDist;
#[doc(inline)]
pub use columns::{ColumnBlockCyclic, ColumnBlockDist};
#[doc(inline)]
pub use random::RandomDist;
#[doc(inline)]
pub use replicate::ReplicateDist;
#[doc(inline)]
pub use rows::{RowBlockCyclic, RowBlockDist};
#[doc(inline)]
pub use stencil::StencilDist;
#[doc(inline)]
pub use stencil2d::Stencil2DDist;
#[doc(inline)]
pub use tile::TileDist;

pub type BlockDist = RowBlockDist;
pub type BlockCyclic = RowBlockCyclic;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ChunkQueryResult {
    pub chunk_index: usize,   // The chunk id.
    pub chunk_offset: Point,  // The offset within the chunk.
    pub region_offset: Point, // The offset relative to the region that was queried.
    pub extents: Dim,         // The no. of elements along each axis.
}

pub trait IntoWorkDistribution {
    fn into_work_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<Arc<dyn WorkDistribution>>;
}

impl<T: IntoWorkDistribution> IntoWorkDistribution for &T
where
    T: Clone,
{
    fn into_work_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<Arc<dyn WorkDistribution>> {
        self.clone().into_work_distribution(system, size)
    }
}

pub trait WorkDistribution {
    fn query_point(&self, p: Point) -> ExecutorId;
    fn query_region(&self, region: Rect) -> Vec<(ExecutorId, Rect)>;
}

#[derive(Debug)]
pub struct ChunkDescriptor {
    pub owner: WorkerId,
    pub affinity: MemoryKind,
    pub size: Dim,
}

/// Types that can be converted into an [`DataDistribution`].
///
/// Anything implementing [`IntoDataDistribution`] can be converted into a concrete [`DataDistribution`]
/// given an array size and the list of available workers. The trait could thus be seen as a
/// "factory" object which can produce [`DataDistribution`]s.
///
/// The difference between [`DataDistribution`] and [`IntoDataDistribution`] is that the first specifies
/// a concrete data distribution for physical array, while the latter specifies an abstract
/// description of a data distribution (for example: row-wise, column-wise, tiled, replicated, etc.)
pub trait IntoDataDistribution {
    fn into_data_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)>;
}

impl<T: IntoDataDistribution> IntoDataDistribution for &T
where
    T: Clone,
{
    fn into_data_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)> {
        self.clone().into_data_distribution(system, size)
    }
}

/// Interface that specifies how the data of an array should be distributed across the workers.
pub trait DataDistribution: Debug + AsAny + Send + Sync + 'static {
    fn as_work_distribution(&self) -> Option<&dyn WorkDistribution> {
        None
    }

    fn clone_region(
        &self,
        _system: &SystemInfo,
        _region: Rect,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)> {
        bail!("cloning of distribution is not supported");
    }

    /// Query how the data within the given region is distributed.
    ///
    /// The provided callback will be called for each returned subregion. The union of the
    /// subregions represents the entire region. Subregions are disjoint (do not overlap), meaning
    /// each point in the given region will represented by one _unique_ subregion.
    ///
    /// The affinity parameter is used to signal that it is preferred if the chunks have affinity
    /// to the given memory space.
    fn visit_unique(
        &self,
        region: Rect,
        affinity: Option<MemoryId>,
        callback: &mut dyn FnMut(ChunkQueryResult),
    );

    /// Query how the data within the given region is distributed.
    ///
    /// The provided callback will be called for each returned subregion within the given region.
    /// The union of the  subregions represents the entire region. All chunks that lie within the
    /// given region will be returned, meaning that the same point may be presented in multiple
    /// subregions if it is replicated.
    fn visit_replicated(&self, region: Rect, callback: &mut dyn FnMut(ChunkQueryResult));
}

pub trait MemoryDistribution {
    fn generate(self, system: &SystemInfo, num_hint: Option<usize>) -> Result<Vec<MemoryId>>;
}

#[derive(Copy, Clone, Debug, Default)]
pub struct AllNodes;
impl MemoryDistribution for AllNodes {
    fn generate(self, system: &SystemInfo, num_hint: Option<usize>) -> Result<Vec<MemoryId>> {
        let places = system
            .workers()
            .iter()
            .map(|w| w.memory_id)
            .collect::<Vec<_>>();

        MemoryDistribution::generate(places, system, num_hint)
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct AllGPUs;
impl MemoryDistribution for AllGPUs {
    fn generate(self, system: &SystemInfo, num_hint: Option<usize>) -> Result<Vec<MemoryId>> {
        let places = system
            .workers()
            .iter()
            .flat_map(|w| &w.devices)
            .map(|w| w.memory_id)
            .collect::<Vec<_>>();

        MemoryDistribution::generate(places, system, num_hint)
    }
}

impl<T: MemoryDistribution + Clone> MemoryDistribution for &T {
    fn generate(self, system: &SystemInfo, num_hint: Option<usize>) -> Result<Vec<MemoryId>> {
        self.clone().generate(system, num_hint)
    }
}

impl MemoryDistribution for MemoryId {
    fn generate(self, _system: &SystemInfo, num_hint: Option<usize>) -> Result<Vec<MemoryId>> {
        let n = num_hint.unwrap_or(1);
        Ok(vec![self; n])
    }
}

impl MemoryDistribution for Vec<MemoryId> {
    fn generate(mut self, _system: &SystemInfo, num_hint: Option<usize>) -> Result<Vec<MemoryId>> {
        if let Some(n) = num_hint {
            if self.len() < n {
                for i in 0..(n - self.len()) {
                    self.push(self[i]);
                }
            } else {
                while self.len() > n {
                    let _ = self.pop();
                }
            }
        }

        Ok(self)
    }
}

impl MemoryDistribution for &[MemoryId] {
    fn generate(self, system: &SystemInfo, num_hint: Option<usize>) -> Result<Vec<MemoryId>> {
        self.to_vec().generate(system, num_hint)
    }
}

impl<const N: usize> MemoryDistribution for [MemoryId; N] {
    fn generate(self, system: &SystemInfo, num_hint: Option<usize>) -> Result<Vec<MemoryId>> {
        self.to_vec().generate(system, num_hint)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    #[derive(Debug, Copy, Clone)]
    pub(super) struct ReferenceChunk {
        pub(super) subregion: Rect,
        pub(super) owned_subregion: Rect,
    }

    pub(super) fn validate_distribution(
        size: Dim,
        chunks: &[ReferenceChunk],
        distribution: &dyn DataDistribution,
    ) {
        for (index, &chunk) in enumerate(chunks) {
            let mut output = vec![];
            distribution.visit_unique(chunk.owned_subregion, None, &mut |v| output.push(v));

            assert_eq!(
                output,
                [ChunkQueryResult {
                    chunk_index: index,
                    chunk_offset: chunk.owned_subregion.low() - chunk.subregion.low(),
                    region_offset: Point::zeros(),
                    extents: chunk.owned_subregion.extents(),
                }]
            );
        }

        let mut rng = SmallRng::seed_from_u64(0);

        for _ in 0..100 {
            let replicated: bool = rng.gen();
            let (mut lo, mut hi) = (Point::zeros(), Point::zeros());
            for i in 0..3 {
                let a = rng.gen_range(0..size[i]);
                let mut b = a;
                while a == b {
                    b = rng.gen_range(0..=size[i]);
                }

                lo[i] = a.min(b);
                hi[i] = a.max(b);
            }

            let query = Rect::from_bounds(lo, hi);
            let mut gotten = vec![];
            if replicated {
                distribution.visit_replicated(query, &mut |r| gotten.push(r));
            } else {
                distribution.visit_unique(query, None, &mut |r| gotten.push(r));
            }

            // Check if subregions do not overlap
            if !replicated {
                for (i, a) in gotten.iter().enumerate() {
                    for (j, b) in gotten.iter().enumerate() {
                        let a_region = Rect::new(a.region_offset + query.low(), a.extents);
                        let b_region = Rect::new(b.region_offset + query.low(), b.extents);

                        if i != j && Rect::intersects(a_region, b_region) {
                            panic!("visit_unique returned intersecting regions: {:#?}", gotten);
                        }
                    }
                }
            }

            // Check if all chunks are matched
            if replicated {
                for (index, chunk) in enumerate(chunks) {
                    if let Some(intersect) = Rect::intersection(query, chunk.subregion) {
                        let a = gotten.iter().find(|e| e.chunk_index == index).unwrap();

                        assert_eq!(
                            intersect,
                            Rect::new(a.region_offset + query.low(), a.extents)
                        );
                    }
                }
            }

            // Check if chunks are actually valid
            for a in &gotten {
                let chunk = &chunks[a.chunk_index];

                let a_region = Rect::new(a.region_offset + query.low(), a.extents);
                assert!(chunk.subregion.contains(a_region));

                let chunk_region = Rect::new(a.chunk_offset, a.extents);
                assert!(chunk.subregion.extents().to_bounds().contains(chunk_region));
            }

            // Check if union matches
            let mut union = Rect::default();
            for a in &gotten {
                let a_region = Rect::new(a.region_offset + query.low(), a.extents);
                union = Rect::union(union, a_region)
            }
            //dbg!(query, &gotten);
            assert_eq!(
                union, query,
                "union of subregions does not match query region"
            );
        }
    }
}
