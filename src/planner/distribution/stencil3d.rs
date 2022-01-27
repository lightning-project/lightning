//! Internals for [`StencilDist`].
use super::*;
use crate::types::{Dim3, Point3};

/// Distributes 1D array in block-cylic fashion while also maintaining halo cells.
#[derive(Copy, Clone, Debug)]
pub struct Stencil3DDist<P = AllGPUs> {
    tile_size: Dim3,
    halo_lo: Dim3,
    halo_hi: Dim3,
    shift: Point3,
    places: P,
}

impl Stencil3DDist<AllGPUs> {
    pub fn new(tile_size: impl Into<Dim3>, halo: u64) -> Self {
        Self::with_memories(tile_size, halo, AllGPUs)
    }
}

impl<P> Stencil3DDist<P> {
    pub fn with_memories(tile_size: impl Into<Dim3>, halo: u64, places: P) -> Self {
        Self {
            tile_size: tile_size.into(),
            halo_lo: Dim3::repeat(halo),
            halo_hi: Dim3::repeat(halo),
            shift: Point3::zeros(),
            places,
        }
    }

    pub fn halo(mut self, halo_lo: impl Into<Dim3>, halo_hi: impl Into<Dim3>) -> Self {
        self.halo_lo = halo_lo.into();
        self.halo_hi = halo_hi.into();
        self
    }

    pub fn shift(mut self, shift: impl Into<Point3>) -> Self {
        self.shift = shift.into();
        self
    }
}

impl<P: MemoryDistribution> Stencil3DDist<P> {
    fn into_dist(self, system: &SystemInfo, size: Dim) -> Result<Arc<Stencil3DDistribution>> {
        let mut tile = self.tile_size;
        let mut halo_lo = self.halo_lo;
        let mut halo_hi = self.halo_hi;

        if tile.contains(&0) {
            bail!("invalid tile size {:?}: cannot be zero", tile);
        }

        for i in 0..3 {
            if tile[i] >= size[i] {
                tile[i] = size[i];
                halo_lo[i] = 0;
                halo_hi[i] = 0;
            }
        }

        let shift = self.shift.to_dim() % tile;
        let offset = (tile - shift) % tile;
        let tile_halo = tile + halo_lo + halo_hi;
        let offset_halo = offset + halo_lo;
        let num_blocks = Dim3::div_ceil(size + offset, Dim3::from(tile));

        let places = self
            .places
            .generate(system, Some(num_blocks.volume() as usize))?;

        let dist = Stencil3DDistribution {
            num_blocks,
            tile,
            offset: offset.to_point(),
            tile_halo,
            offset_halo: offset_halo.to_point(),
            size,
            places,
        };

        Ok(Arc::new(dist))
    }
}

impl<P: MemoryDistribution> IntoDataDistribution for Stencil3DDist<P> {
    fn into_data_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)> {
        let dist = self.into_dist(system, size)?;

        let [xn, yn, zn] = *dist.num_blocks;
        let places = &dist.places;
        let mut chunks = vec![];

        for i in 0..xn {
            for j in 0..yn {
                for k in 0..zn {
                    let region = dist.chunk_region_with_halo(Point::new(i, j, k));

                    let place = places[chunks.len()];
                    chunks.push(ChunkDescriptor {
                        owner: place.node_id(),
                        affinity: place.kind(),
                        size: region.extents(),
                    });
                }
            }
        }

        Ok((dist, chunks))
    }
}

#[derive(Debug, Clone)]
pub struct Stencil3DDistribution {
    num_blocks: Dim3,
    tile: Dim3,
    offset: Point3,
    tile_halo: Dim3,
    offset_halo: Point3,
    size: Dim3,
    places: Vec<MemoryId>,
}

impl Stencil3DDistribution {
    fn chunk_index(&self, p: Point3) -> usize {
        let [_, yn, zn] = *self.num_blocks;
        ((p[0] * yn + p[1]) * zn + p[2]) as usize
    }

    fn chunk_region_with_halo(&self, p: Point3) -> Rect {
        let lo = (p * self.tile).saturating_sub(self.offset_halo);
        let hi = Point3::element_min(
            p * self.tile + self.tile_halo - self.offset_halo,
            self.size.to_point(),
        );

        Rect::from_bounds(lo, hi)
    }

    fn chunk_region_without_halo(&self, p: Point3) -> Rect {
        let lo = (p * self.tile).saturating_sub(self.offset);
        let hi = Point3::element_min(
            p * self.tile + self.tile - self.offset,
            self.size.to_point(),
        );

        Rect::from_bounds(lo, hi)
    }

    fn query_local(&self, region: Rect) -> Option<(usize, Point)> {
        let [xstart, ystart, zstart] = *((region.high() + self.offset_halo)
            .saturating_sub(self.tile_halo - self.tile + Point::ones())
            .to_dim()
            / self.tile);

        let [xend, yend, zend] = *((region.low() + self.offset_halo + Point::ones())
            .to_dim()
            .div_ceil(self.tile));

        if xstart < xend && ystart < yend && zstart < zend {
            let p = Point::new(xstart, ystart, zstart);
            let index = self.chunk_index(p);
            let chunk_lo = self.chunk_region_with_halo(p).low();

            Some((index, chunk_lo))
        } else {
            None
        }
    }

    fn query_tiles<F, const INCLUDE_HALOS: bool>(&self, region: Rect, mut callback: F)
    where
        F: FnMut(usize, Point, Rect),
    {
        let start;
        let end;

        if INCLUDE_HALOS {
            start = (region.low() + self.offset_halo)
                .to_dim()
                .saturating_sub(self.tile_halo - self.tile)
                / self.tile;
            end = (region.high() + self.offset_halo)
                .element_min(self.size.to_point() + self.offset)
                .to_dim()
                .div_ceil(self.tile);
        } else {
            start = ((region.low() + self.offset) / self.tile).to_dim();
            end = (region.high() + self.offset).to_dim().div_ceil(self.tile);
        }

        debug_assert!(start[0] < self.num_blocks[0]);
        debug_assert!(start[1] < self.num_blocks[1]);
        debug_assert!(start[2] < self.num_blocks[2]);

        debug_assert!(end[0] <= self.num_blocks[0]);
        debug_assert!(end[1] <= self.num_blocks[1]);
        debug_assert!(end[2] <= self.num_blocks[2]);

        for i in start[0]..end[0] {
            for j in start[1]..end[1] {
                for k in start[2]..end[2] {
                    let index = self.chunk_index(Point::new(i, j, k));
                    let chunk = self.chunk_region_without_halo(Point::new(i, j, k));
                    let chunk_halo = self.chunk_region_with_halo(Point::new(i, j, k));

                    if INCLUDE_HALOS {
                        debug_assert!(
                            chunk_halo.intersects(region),
                            "chunk {:?} does not intersect {:?}",
                            chunk_halo,
                            region
                        );
                    } else {
                        debug_assert!(chunk.intersects(region));
                    }

                    let p0;
                    let p1;

                    if INCLUDE_HALOS {
                        p0 = chunk_halo.low();
                        p1 = chunk_halo.high();
                    } else {
                        p0 = chunk.low();
                        p1 = chunk.high();
                    }

                    let p0 = Point::element_max(p0, region.low());
                    let p1 = Point::element_min(p1, region.high());

                    (callback)(index, chunk_halo.low(), Rect::from_bounds(p0, p1));
                }
            }
        }
    }
}

impl DataDistribution for Stencil3DDistribution {
    fn visit_unique(
        &self,
        region: Rect,
        _affinity: Option<MemoryId>,
        callback: &mut dyn FnMut(ChunkQueryResult),
    ) {
        if let Some((chunk_index, chunk_start)) = self.query_local(region) {
            (callback)(ChunkQueryResult {
                chunk_index,
                chunk_offset: region.low() - chunk_start,
                region_offset: Point::zeros(),
                extents: region.extents(),
            });

            return;
        }

        self.query_tiles::<_, false>(region, move |chunk_index, chunk_start, overlap| {
            (callback)(ChunkQueryResult {
                chunk_index,
                chunk_offset: overlap.low() - chunk_start,
                region_offset: overlap.low() - region.low(),
                extents: overlap.extents(),
            })
        })
    }

    fn visit_replicated(&self, region: Rect, callback: &mut dyn FnMut(ChunkQueryResult)) {
        self.query_tiles::<_, true>(region, move |chunk_index, chunk_start, overlap| {
            (callback)(ChunkQueryResult {
                chunk_index,
                chunk_offset: overlap.low() - chunk_start,
                region_offset: overlap.low() - region.low(),
                extents: overlap.extents(),
            })
        })
    }
}

#[cfg(test)]
mod test {
    use super::{super::test::*, *};
    use crate::types::{MemoryKind, WorkerId};

    fn validate_stencil3d_distribution_ext(
        tile_size: [u64; 3],
        halo_lo: [u64; 3],
        halo_hi: [u64; 3],
        shift: [u64; 3],
        size: [u64; 3],
    ) {
        let mut chunks = vec![];

        let a = (tile_size[0] - shift[0] % tile_size[0]) % tile_size[0];
        let b = (tile_size[1] - shift[1] % tile_size[1]) % tile_size[1];
        let c = (tile_size[2] - shift[2] % tile_size[2]) % tile_size[2];

        for xstart in (-(a as i64)..(size[0] as i64)).step_by(tile_size[0] as usize) {
            for ystart in (-(b as i64)..(size[1] as i64)).step_by(tile_size[1] as usize) {
                for zstart in (-(c as i64)..(size[2] as i64)).step_by(tile_size[2] as usize) {
                    let xend = min(xstart + tile_size[0] as i64, size[0] as i64) as u64;
                    let xstart = max(xstart, 0) as u64;

                    let yend = min(ystart + tile_size[1] as i64, size[1] as i64) as u64;
                    let ystart = max(ystart, 0) as u64;

                    let zend = min(zstart + tile_size[2] as i64, size[2] as i64) as u64;
                    let zstart = max(zstart, 0) as u64;

                    let xstart_with_halo = xstart.saturating_sub(halo_lo[0]);
                    let xend_with_halo = min(xend + halo_hi[0], size[0]);

                    let ystart_with_halo = ystart.saturating_sub(halo_lo[1]);
                    let yend_with_halo = min(yend + halo_hi[1], size[1]);

                    let zstart_with_halo = zstart.saturating_sub(halo_lo[2]);
                    let zend_with_halo = min(zend + halo_hi[2], size[2]);

                    chunks.push(ReferenceChunk {
                        subregion: dbg!((
                            xstart_with_halo..xend_with_halo,
                            ystart_with_halo..yend_with_halo,
                            zstart_with_halo..zend_with_halo,
                        ))
                        .into(),
                        owned_subregion: (xstart..xend, ystart..yend, zstart..zend).into(),
                    });
                }
            }
        }

        let dist = Stencil3DDist {
            tile_size: Dim3::from(tile_size),
            halo_lo: Dim3::from(halo_lo),
            halo_hi: Dim3::from(halo_hi),
            shift: Point3::from(shift),
            places: vec![MemoryId::new(WorkerId::new(0), MemoryKind::Host)],
        };
        let dist = dist
            .into_dist(&SystemInfo::new(vec![]), Dim3::from(size))
            .unwrap();

        validate_distribution(Dim3::from(size), &chunks, &*dist);
    }

    fn validate_stencil3d_distribution(tile_size: [u64; 3], shift: [u64; 3], size: [u64; 3]) {
        let halos = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [1, 1, 2]];

        for &halo_lo in &halos {
            for &halo_hi in &halos {
                validate_stencil3d_distribution_ext(tile_size, halo_lo, halo_hi, shift, size);
            }
        }
    }

    #[test]
    fn test_stencil3d_distribution() {
        validate_stencil3d_distribution([25, 25, 25], [0, 0, 0], [140, 150, 150]);
        validate_stencil3d_distribution([26, 26, 26], [0, 0, 0], [150, 150, 150]);
        validate_stencil3d_distribution([24, 24, 24], [0, 0, 0], [150, 150, 150]);

        validate_stencil3d_distribution([25, 25, 25], [0, 0, 0], [125, 150, 150]);
        validate_stencil3d_distribution([25, 26, 25], [1, 0, 0], [125, 150, 120]);
        validate_stencil3d_distribution([25, 26, 20], [1, 2, 1], [125, 150, 125]);
        validate_stencil3d_distribution([25, 26, 13], [0, 2, 0], [125, 150, 100]);
        validate_stencil3d_distribution([25, 26, 20], [2, 4, 2], [120, 150, 120]);
    }
}
