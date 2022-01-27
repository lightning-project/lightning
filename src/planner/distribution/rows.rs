//! Internals for [`RowBlockCyclic`] and [`RowBlockDist`].
use super::*;

/// Distributes rows among devices in block-cyclic fashion.
#[derive(Copy, Clone, Debug)]
pub struct RowBlockCyclic<P = AllGPUs> {
    block_size: u64,
    places: P,
}

impl RowBlockCyclic<AllGPUs> {
    pub fn new(block_size: u64) -> Self {
        Self::with_memories(block_size, AllGPUs)
    }
}

impl<P> RowBlockCyclic<P> {
    pub fn with_memories(block_size: u64, places: P) -> Self {
        Self { block_size, places }
    }

    fn to_tile_dist(self, size: Dim) -> TileDist<P> {
        let tile_size = Dim::new(self.block_size, size[1], size[2]);
        let places = self.places;

        TileDist::with_memories(tile_size, places)
    }
}

impl<P: MemoryDistribution> IntoDataDistribution for RowBlockCyclic<P> {
    fn into_data_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)> {
        self.to_tile_dist(size).into_data_distribution(system, size)
    }
}

impl<P: MemoryDistribution> IntoWorkDistribution for RowBlockCyclic<P> {
    fn into_work_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<Arc<dyn WorkDistribution>> {
        self.to_tile_dist(size).into_work_distribution(system, size)
    }
}

/// Distributes rows among devices in blocked fashion.
#[derive(Copy, Clone, Debug)]
pub struct RowBlockDist {
    alignment: u64,
}

impl RowBlockDist {
    pub fn new() -> Self {
        Self::with_alignment(1)
    }

    pub fn with_alignment(alignment: u64) -> Self {
        Self { alignment }
    }

    fn to_tile_dist(self, system: &SystemInfo, size: Dim) -> TileDist<Vec<MemoryId>> {
        let places = AllGPUs.generate(system, None).unwrap();
        let nworkers = places.len() as u64;
        let block_size = div_ceil(size[0], self.alignment * nworkers) * self.alignment;

        RowBlockCyclic::with_memories(block_size, places).to_tile_dist(size)
    }
}

impl IntoDataDistribution for RowBlockDist {
    fn into_data_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)> {
        self.to_tile_dist(system, size)
            .into_data_distribution(system, size)
    }
}

impl IntoWorkDistribution for RowBlockDist {
    fn into_work_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<Arc<dyn WorkDistribution>> {
        self.to_tile_dist(system, size)
            .into_work_distribution(system, size)
    }
}
