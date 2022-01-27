//! Internals for [`CentralizeDist`].
use super::*;

/// Distribution that centralizes data at one memory.
#[derive(Copy, Clone, Debug)]
pub struct CentralizeDist {
    place: Option<MemoryId>,
}

impl CentralizeDist {
    pub fn new(place: MemoryId) -> Self {
        Self { place: Some(place) }
    }

    pub fn root() -> Self {
        Self { place: None }
    }
}

impl IntoDataDistribution for CentralizeDist {
    fn into_data_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)> {
        let place = match self.place {
            Some(p) => p,
            None => system.workers()[0].devices[0].memory_id,
        };

        let chunks = vec![ChunkDescriptor {
            size,
            owner: place.node_id(),
            affinity: place.kind(),
        }];

        let dist = Arc::new(CentralizeDistribution { place, size });

        Ok((dist, chunks))
    }
}

impl IntoWorkDistribution for CentralizeDist {
    fn into_work_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<Arc<dyn WorkDistribution>> {
        let place = match self.place {
            Some(p) => p,
            None => system.workers()[0].devices[0].memory_id,
        };

        let dist = Arc::new(CentralizeDistribution { place, size });
        Ok(dist)
    }
}

#[derive(Debug)]
struct CentralizeDistribution {
    place: MemoryId,
    size: Dim,
}

impl WorkDistribution for CentralizeDistribution {
    fn query_point(&self, _p: Point) -> ExecutorId {
        self.place.best_affinity_executor()
    }

    fn query_region(&self, region: Rect) -> Vec<(ExecutorId, Rect)> {
        vec![(self.place.best_affinity_executor(), region)]
    }
}

impl DataDistribution for CentralizeDistribution {
    fn as_work_distribution(&self) -> Option<&dyn WorkDistribution> {
        Some(self)
    }

    fn clone_region(
        &self,
        system: &SystemInfo,
        region: Rect,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)> {
        CentralizeDist::new(self.place).into_data_distribution(system, region.extents())
    }

    fn visit_unique(
        &self,
        region: Rect,
        _affinity: Option<MemoryId>,
        callback: &mut dyn FnMut(ChunkQueryResult),
    ) {
        self.visit_replicated(region, callback)
    }

    fn visit_replicated(&self, region: Rect, callback: &mut dyn FnMut(ChunkQueryResult)) {
        (callback)(ChunkQueryResult {
            chunk_index: 0,
            region_offset: Point::zeros(),
            chunk_offset: region.low(),
            extents: region.extents(),
        });
    }
}
