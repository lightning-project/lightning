//! Internals for [`TileDist`].
use super::IntoWorkDistribution;
use super::{
    AllGPUs, ChunkDescriptor, ChunkQueryResult, DataDistribution, IntoDataDistribution,
    MemoryDistribution, WorkDistribution,
};
use crate::prelude::*;
use crate::types::{Dim, ExecutorId, MemoryId, Point, Rect, SystemInfo, MAX_DIMS};
use std::sync::Arc;

/// Distributes 2D array using tiles.
#[derive(Copy, Clone, Debug)]
pub struct TileDist<P = AllGPUs> {
    pub(crate) tile_size: Dim,
    pub(crate) places: P,
}

impl TileDist {
    pub fn new(tile_size: impl Into<Dim>) -> Self {
        Self::with_memories(tile_size.into(), AllGPUs)
    }
}

impl<P> TileDist<P> {
    pub fn with_memories(tile_size: impl Into<Dim>, places: P) -> Self {
        Self {
            tile_size: tile_size.into(),
            places,
        }
    }
}

impl<P: Clone> TileDist<P> {
    pub fn stride_by(&self, step: Dim) -> Result<Self> {
        if step.volume() == 0 || self.tile_size % step != Dim::empty() {
            bail!(
                "tile size {:?} is not divisible by stride {:?}",
                self.tile_size,
                step
            );
        }

        Ok(Self {
            tile_size: self.tile_size / step,
            places: self.places.clone(),
        })
    }
}

impl<P: MemoryDistribution> IntoDataDistribution for TileDist<P> {
    fn into_data_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)> {
        let tile_size = self.tile_size;
        if tile_size.volume() == 0 {
            bail!("tile size cannot be zero");
        }

        let num_tiles = Dim::div_ceil(size, tile_size);
        let places = self
            .places
            .generate(system, Some(num_tiles.volume() as usize))?
            .into_boxed_slice();

        let mut chunks = vec![];
        let mut index = 0;

        for x in 0..num_tiles[0] {
            for y in 0..num_tiles[1] {
                for z in 0..num_tiles[2] {
                    let chunk_size = Dim::new(
                        min(tile_size[0], size[0] - x * tile_size[0]),
                        min(tile_size[1], size[1] - y * tile_size[1]),
                        min(tile_size[2], size[2] - z * tile_size[2]),
                    );

                    chunks.push(ChunkDescriptor {
                        size: chunk_size,
                        owner: places[index].node_id(),
                        affinity: places[index].kind(),
                    });

                    index += 1;
                }
            }
        }

        macro_rules! boxed {
            ($x:expr, $y:expr, $z: expr) => {
                Arc::new(TileDistribution::<$x, $y, $z> {
                    tile_size,
                    num_tiles,
                    places,
                }) as Arc<dyn DataDistribution>
            };
        }

        let dist = match *num_tiles {
            [_, 1, 1] => boxed!(true, false, false), // row-wise
            [1, _, 1] => boxed!(false, true, false), // column-wise
            [_, _, 1] => boxed!(true, true, false),  // 2d-tile
            [_, _, _] => boxed!(true, true, true),   // 3d-tile
        };

        Ok((dist, chunks))
    }
}

impl<P: MemoryDistribution> IntoWorkDistribution for TileDist<P> {
    fn into_work_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<Arc<dyn WorkDistribution>> {
        let tile_size = self.tile_size;
        if tile_size.volume() == 0 {
            bail!("tile size cannot be zero");
        }

        let num_tiles = Dim::div_ceil(size, tile_size);
        let places = self
            .places
            .generate(system, Some(num_tiles.volume() as usize))?
            .into_boxed_slice();

        let dist = Arc::new(TileDistribution::<true, true, true> {
            tile_size,
            num_tiles,
            places,
        });

        Ok(dist)
    }
}

#[derive(Debug, Clone)]
struct TileDistribution<const X: bool, const Y: bool, const Z: bool> {
    tile_size: Dim,
    num_tiles: Dim,
    places: Box<[MemoryId]>,
}

impl<const X: bool, const Y: bool, const Z: bool> WorkDistribution for TileDistribution<X, Y, Z> {
    fn query_point(&self, p: Point) -> ExecutorId {
        let mut index = 0;
        for i in 0..MAX_DIMS {
            index = index * self.num_tiles[i] + (p[i] / self.tile_size[i]);
        }

        self.places[index as usize].best_affinity_executor()
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

impl<const X: bool, const Y: bool, const Z: bool> DataDistribution for TileDistribution<X, Y, Z> {
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
        if region.is_empty() {
            return;
        }

        let mut start = [0; MAX_DIMS];
        let mut end = [1; MAX_DIMS];
        for i in 0..MAX_DIMS {
            if [X, Y, Z][i] {
                start[i] = region.low()[i] / self.tile_size[i];
                end[i] = div_ceil(region.high()[i], self.tile_size[i]);
            }
        }

        for x in start[0]..end[0] {
            for y in start[1]..end[1] {
                for z in start[2]..end[2] {
                    let tile_size = self.tile_size;
                    let num_tiles = self.num_tiles;

                    let p = [x, y, z];
                    let mut chunk_offset = Point::zeros();
                    let mut region_offset = Point::zeros();
                    let mut extents = Dim::one();
                    let mut index = 0;

                    for i in 0..MAX_DIMS {
                        let region_lo = region.low()[i];
                        let region_hi = region.high()[i];

                        if [X, Y, Z][i] {
                            index = index * num_tiles[i] + p[i];

                            let tile_lo = p[i] * tile_size[i];
                            let tile_hi = tile_lo + tile_size[i];

                            chunk_offset[i] = u64::saturating_sub(region_lo, tile_lo);
                            region_offset[i] = u64::saturating_sub(tile_lo, region_lo);
                            extents[i] =
                                u64::min(tile_hi, region_hi) - u64::max(tile_lo, region_lo);
                        } else {
                            chunk_offset[i] = region_lo;
                            region_offset[i] = 0;
                            extents[i] = region_hi - region_lo;
                        }
                    }

                    (callback)(ChunkQueryResult {
                        chunk_index: index as usize,
                        chunk_offset,
                        region_offset,
                        extents,
                    });
                }
            }
        }
    }

    fn visit_replicated(&self, region: Rect, callback: &mut dyn FnMut(ChunkQueryResult)) {
        self.visit_unique(region, None, callback)
    }
}

/*
#[cfg(test)]
mod test {
    use super::{super::test::*, *};

    fn validate_tile_distribution(size: Extent, tile_size: Extent) {
        let num_tiles = Extent::div_ceil(size, tile_size).xy();
        let places = (0..num_tiles.volume())
            .map(|_| GlobalMemoryId {
                node: NodeId(0),
                mem: Memory::Host,
            })
            .collect_vec();

        let dist = TileDistribution {
            offset: Point2::zeros(),
            size,
            tile_size: tile_size.xy(),
            num_tiles,
            places,
        };

        let mut chunks = vec![];

        for x in 0..num_tiles[0] {
            for y in 0..num_tiles[1] {
                let tile = Rect::new(Point::new(x, y, 0) * tile_size, tile_size);
                let region = Rect::intersection(tile, size.to_bounds()).unwrap();

                chunks.push(ReferenceChunk {
                    shared_region: region,
                    owned_region: region,
                });
            }
        }

        validate_distribution(size, &chunks, &dist);
    }

    #[test]
    fn test_tile_distribution() {
        validate_tile_distribution(Extent::from(100), Extent::from(11));

        validate_tile_distribution(Extent::from((100, 120)), Extent::from((27, 24)));

        validate_tile_distribution(Extent::from((100, 120, 140)), Extent::from((27, 24, 70)));
    }
}
*/
