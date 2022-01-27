//! Internals for [`ColumnBlockCyclic`] and [`ColumnBlockDist`].
use super::*;

/// Distributes columns among devices in block-cyclic fashion.
#[derive(Copy, Clone, Debug)]
pub struct ColumnBlockCyclic<P = AllGPUs> {
    block_size: u64,
    places: P,
}

impl ColumnBlockCyclic<AllGPUs> {
    pub fn new(block_size: u64) -> Self {
        Self::with_memories(block_size, AllGPUs)
    }
}

impl<P> ColumnBlockCyclic<P> {
    pub fn with_memories(block_size: u64, places: P) -> Self {
        Self { block_size, places }
    }

    fn to_tile_dist(self, size: Dim) -> TileDist<P> {
        let tile_size = Dim::new(size[0], self.block_size, size[2]);
        let places = self.places;

        TileDist::with_memories(tile_size, places)
    }
}

impl<P: MemoryDistribution> IntoDataDistribution for ColumnBlockCyclic<P> {
    fn into_data_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)> {
        self.to_tile_dist(size).into_data_distribution(system, size)
    }
}

impl<P: MemoryDistribution> IntoWorkDistribution for ColumnBlockCyclic<P> {
    fn into_work_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<Arc<dyn WorkDistribution>> {
        self.to_tile_dist(size).into_work_distribution(system, size)
    }
}

/// Distributes columns among devices in blocked fashion.
#[derive(Copy, Clone, Debug)]
pub struct ColumnBlockDist {
    alignment: u64,
}

impl ColumnBlockDist {
    pub fn new() -> Self {
        Self::with_alignment(1)
    }

    pub fn with_alignment(alignment: u64) -> Self {
        Self { alignment }
    }

    fn to_tile_dist(self, system: &SystemInfo, size: Dim) -> TileDist<Vec<MemoryId>> {
        let places = AllGPUs.generate(system, None).unwrap();
        let nworkers = places.len() as u64;
        let block_size = div_ceil(size[1], self.alignment * nworkers) * self.alignment;

        ColumnBlockCyclic::with_memories(block_size, places).to_tile_dist(size)
    }
}

impl IntoDataDistribution for ColumnBlockDist {
    fn into_data_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)> {
        self.to_tile_dist(system, size)
            .into_data_distribution(system, size)
    }
}

impl IntoWorkDistribution for ColumnBlockDist {
    fn into_work_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<Arc<dyn WorkDistribution>> {
        self.to_tile_dist(system, size)
            .into_work_distribution(system, size)
    }
}
