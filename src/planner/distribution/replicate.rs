//! Internals for [`ReplicateDist`].
use super::*;
use itertools::Itertools;

/// Distribution that replicates data across all devices.
#[derive(Copy, Clone, Debug)]
pub struct ReplicateDist<P = AllNodes> {
    places: P,
}

impl ReplicateDist {
    pub fn new() -> Self {
        Self::with_memories(AllNodes)
    }
}

impl<P> ReplicateDist<P> {
    pub fn with_memories(places: P) -> Self {
        Self { places }
    }
}

impl<P: MemoryDistribution> IntoDataDistribution for ReplicateDist<P> {
    fn into_data_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)> {
        let devices = self.places.generate(system, None)?;

        let chunks = devices
            .iter()
            .sorted()
            .unique()
            .map(|&place| ChunkDescriptor {
                size,
                owner: place.node_id(),
                affinity: place.kind(),
            })
            .collect();

        let dist = Arc::new(ReplicateDistribution {
            devices: devices.into_boxed_slice(),
            size,
        });

        Ok((dist, chunks))
    }
}

#[derive(Debug)]
struct ReplicateDistribution {
    devices: Box<[MemoryId]>,
    size: Dim,
}

impl DataDistribution for ReplicateDistribution {
    fn clone_region(
        &self,
        system: &SystemInfo,
        region: Rect,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)> {
        ReplicateDist::new().into_data_distribution(system, region.extents())
    }

    fn visit_unique(
        &self,
        region: Rect,
        affinity: Option<MemoryId>,
        callback: &mut dyn FnMut(ChunkQueryResult),
    ) {
        let chunk_index = match affinity {
            Some(x) => {
                if let Ok(i) = self.devices.binary_search(&x) {
                    i
                } else if let Ok(i) = self
                    .devices
                    .binary_search(&MemoryId::new(x.node_id(), MemoryKind::Host))
                {
                    i
                } else {
                    0
                }
            }
            None => 0,
        };

        (callback)(ChunkQueryResult {
            chunk_index,
            region_offset: Point::zeros(),
            chunk_offset: region.low(),
            extents: region.extents(),
        });
    }

    fn visit_replicated(&self, region: Rect, callback: &mut dyn FnMut(ChunkQueryResult)) {
        for chunk_index in 0..self.devices.len() {
            (callback)(ChunkQueryResult {
                chunk_index,
                region_offset: Point::zeros(),
                chunk_offset: region.low(),
                extents: region.extents(),
            });
        }
    }
}
