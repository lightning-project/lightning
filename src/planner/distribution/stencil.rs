//! Internals for [`StencilDist`].
use super::*;

/// Distributes 1D array in block-cylic fashion while also maintaining halo cells.
#[derive(Copy, Clone, Debug)]
pub struct StencilDist<P = AllGPUs> {
    chunk_size: u64,
    halo_size: u64,
    places: P,
}

impl StencilDist<AllGPUs> {
    pub fn new(chunk_size: u64, halo_size: u64) -> Self {
        Self::with_memories(chunk_size, halo_size, default())
    }
}

impl<P> StencilDist<P> {
    pub fn with_memories(chunk_size: u64, halo_size: u64, places: P) -> Self {
        Self {
            chunk_size,
            halo_size,
            places,
        }
    }
}

impl<P: MemoryDistribution> StencilDist<P> {
    fn into_dist(self, system: &SystemInfo, size: Dim) -> Result<Arc<StencilDistribution>> {
        let n = size[0];
        if Dim::from(n) != size {
            bail!("array must be one-dimensional");
        }

        if self.chunk_size == 0 {
            bail!("block size cannot be zero");
        }

        let num_chunks = div_ceil(n, self.chunk_size);
        let places = self.places.generate(system, Some(num_chunks as usize))?;

        Ok(Arc::new(StencilDistribution {
            chunk_size: self.chunk_size,
            halo_before: self.halo_size,
            halo_after: self.halo_size,
            size: n,
            places,
        }))
    }
}

impl<P: MemoryDistribution> IntoDataDistribution for StencilDist<P> {
    fn into_data_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)> {
        let dist = self.into_dist(system, size)?;

        let mut chunks = vec![];
        let size = dist.size;
        let chunk_size = dist.chunk_size;
        let halo_before = dist.halo_before;
        let halo_after = dist.halo_after;

        for (i, &place) in enumerate(&dist.places) {
            let i = i as u64;
            let a = (i * chunk_size).saturating_sub(halo_before);
            let b = min(size, (i + 1) * chunk_size + halo_after);

            chunks.push(ChunkDescriptor {
                owner: place.node_id(),
                affinity: place.kind(),
                size: Dim::from(b - a),
            });
        }

        Ok((dist, chunks))
    }
}

impl<P: MemoryDistribution> IntoWorkDistribution for StencilDist<P> {
    fn into_work_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<Arc<dyn WorkDistribution>> {
        Ok(self.into_dist(system, size)?)
    }
}

#[derive(Debug, Clone)]
pub struct StencilDistribution {
    chunk_size: u64,  // size of each chunk (exclusive halos)
    halo_before: u64, // no. of halos at the start of each chunk
    halo_after: u64,  // no. of halos at the end of each chunk
    size: u64,        // full number of elements
    places: Vec<MemoryId>,
}

impl WorkDistribution for StencilDistribution {
    fn query_point(&self, p: Point) -> ExecutorId {
        let i = p[0] / self.chunk_size;
        self.places[i as usize].best_affinity_executor()
    }

    fn query_region(&self, region: Rect) -> Vec<(ExecutorId, Rect)> {
        let mut subregions = vec![];
        self.visit_unique(region, None, &mut |r| {
            let place = self.places[r.chunk_index].best_affinity_executor();
            let subregion = Rect::new(r.region_offset + region.low(), r.extents);
            subregions.push((place, subregion));
        });
        subregions
    }
}

impl DataDistribution for StencilDistribution {
    fn as_work_distribution(&self) -> Option<&dyn WorkDistribution> {
        Some(self)
    }

    fn clone_region(
        &self,
        _system: &SystemInfo,
        _region: Rect,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)> {
        bail!("cloning not supported");
    }

    fn visit_unique(
        &self,
        region: Rect,
        _affinity: Option<MemoryId>,
        callback: &mut dyn FnMut(ChunkQueryResult),
    ) {
        let chunk_size = self.chunk_size;
        let halo_before = self.halo_before;
        let halo_after = self.halo_after;

        let lo = region.low()[0];
        let hi = region.high()[0];

        // Early exit if the region fits into one chunk.
        let start = min(lo + halo_before, self.size) / chunk_size;
        let end = div_ceil(max(hi.saturating_sub(halo_after), lo), chunk_size);

        if start + 1 == end {
            let chunk_start = (start * chunk_size).saturating_sub(halo_before);

            callback(ChunkQueryResult {
                chunk_index: start as usize,
                chunk_offset: region.low() - Point::from(chunk_start),
                region_offset: Point::zeros(),
                extents: region.extents(),
            });

            return;
        }

        let start = lo / chunk_size;
        let end = div_ceil(hi, chunk_size);

        for i in start..end {
            let chunk_start = (i * chunk_size).saturating_sub(halo_before);
            let a = max(lo, i * chunk_size);
            let b = min(hi, (i + 1) * chunk_size);

            (callback)(ChunkQueryResult {
                chunk_index: i as usize,
                chunk_offset: Point::from(a - chunk_start),
                region_offset: Point::from(a) - region.low(),
                extents: Dim::from(b - a),
            });
        }
    }

    fn visit_replicated(&self, region: Rect, callback: &mut dyn FnMut(ChunkQueryResult)) {
        let chunk_size = self.chunk_size;
        let halo_before = self.halo_before;
        let halo_after = self.halo_after;

        let lo = region.low()[0];
        let hi = region.high()[0];

        let start = (lo.saturating_sub(halo_before)) / chunk_size;
        let end = div_ceil(min(hi + halo_after, self.size), chunk_size);

        for i in start..end {
            let chunk_start = (i * chunk_size).saturating_sub(halo_before);
            let a = max(lo, chunk_start);
            let b = min(hi, (i + 1) * chunk_size + halo_after);

            (callback)(ChunkQueryResult {
                chunk_index: i as usize,
                chunk_offset: Point::from(a - chunk_start),
                region_offset: Point::from(a) - region.low(),
                extents: Dim::from(b - a),
            });
        }
    }
}

/*
#[cfg(test)]
mod test {
    use super::{super::test::*, *};

    fn validate_stencil_distribution(block_size: u64, halo_size: u64, size: u64) {
        let num_chunks = div_ceil(size, block_size);
        let places = (0..num_chunks)
            .map(|_| GlobalMemoryId {
                node: NodeId(0),
                mem: Memory::Host,
            })
            .collect_vec();

        let dist = StencilDistribution {
            chunk_size: block_size,
            halo_before: halo_size,
            halo_after: halo_size,
            size,
            places,
        };

        let mut chunks = vec![];

        for i in 0..num_chunks {
            let i = i as u64;
            let start = (i * block_size);
            let end = min((i + 1) * block_size, size);

            let start_with_halo = start.saturating_sub(halo_size);
            let end_with_halo = min(end + halo_size, size);

            chunks.push(ReferenceChunk {
                shared_region: (start_with_halo, end_with_halo).into(),
                owned_region: (start, end).into(),
            });
        }

        validate_distribution(Extent::from(size), &chunks, &dist);
    }

    #[test]
    fn test_stencil_distribution() {
        validate_stencil_distribution(25, 1, 1000);

        validate_stencil_distribution(26, 2, 1000);

        validate_stencil_distribution(24, 3, 1000);
    }
}
*/
