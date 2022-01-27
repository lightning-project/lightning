//! Internals for [`StencilDist`].
use super::*;

/// Distributes 1D array in block-cylic fashion while also maintaining halo cells.
#[derive(Copy, Clone, Debug)]
pub struct Stencil2DDist<P = AllGPUs> {
    block_size: [u64; 2],
    halo_size: [u64; 2],
    places: P,
}

impl Stencil2DDist<AllGPUs> {
    pub fn new(block_size: [u64; 2], halo_size: u64) -> Self {
        Self::with_memories(block_size, halo_size, AllGPUs)
    }
}

impl<P> Stencil2DDist<P> {
    pub fn with_memories(block_size: [u64; 2], halo_size: u64, places: P) -> Self {
        Self {
            block_size,
            halo_size: [halo_size, halo_size],
            places,
        }
    }
}

impl<P: MemoryDistribution> Stencil2DDist<P> {
    fn into_dist(self, system: &SystemInfo, size: Dim) -> Result<Arc<Stencil2DDistribution>> {
        let [xsize, ysize, _] = *size;
        if Dim::from((xsize, ysize)) != size {
            bail!("array must be two-dimensional");
        }

        if self.block_size.contains(&0) {
            bail!("block size cannot be zero");
        }

        let xn = div_ceil(xsize, self.block_size[0]);
        let yn = div_ceil(ysize, self.block_size[1]);
        let places = self.places.generate(system, Some((xn * yn) as usize))?;

        let dist = Stencil2DDistribution {
            block_size: self.block_size,
            halo_size: self.halo_size,
            size: [xsize, ysize],
            num_blocks: [xn, yn],
            places,
        };

        Ok(Arc::new(dist))
    }
}

impl<P: MemoryDistribution> IntoDataDistribution for Stencil2DDist<P> {
    fn into_data_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<(Arc<dyn DataDistribution>, Vec<ChunkDescriptor>)> {
        let dist = self.into_dist(system, size)?;

        let [xsize, ysize] = dist.size;
        let [xblock, yblock] = dist.block_size;
        let [xhalo, yhalo] = dist.halo_size;
        let [xn, yn] = dist.num_blocks;
        let places = &dist.places;

        let mut chunks = vec![];

        for i in 0..xn {
            for j in 0..yn {
                let x0 = (i * xblock).saturating_sub(xhalo);
                let x1 = min(xsize, (i + 1) * xblock + xhalo);

                let y0 = (j * yblock).saturating_sub(yhalo);
                let y1 = min(ysize, (j + 1) * yblock + yhalo);

                let place = places[chunks.len()];
                chunks.push(ChunkDescriptor {
                    owner: place.node_id(),
                    affinity: place.kind(),
                    size: Dim::from((x1 - x0, y1 - y0)),
                });
            }
        }

        Ok((dist, chunks))
    }
}

impl<P: MemoryDistribution> IntoWorkDistribution for Stencil2DDist<P> {
    fn into_work_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<Arc<dyn WorkDistribution>> {
        Ok(self.into_dist(system, size)?)
    }
}

#[derive(Debug, Clone)]
pub struct Stencil2DDistribution {
    block_size: [u64; 2],
    halo_size: [u64; 2],
    size: [u64; 2],
    num_blocks: [u64; 2],
    places: Vec<MemoryId>,
}

impl Stencil2DDistribution {
    fn query_local(&self, region: Rect) -> Option<(usize, Point)> {
        let [nrows, ncols] = self.size;
        let [_, yn] = self.num_blocks;
        let [xblock, yblock] = self.block_size;
        let [xhalo, yhalo] = self.halo_size;

        let xlo = region.low()[0];
        let xhi = region.high()[0];
        let ylo = region.low()[1];
        let yhi = region.high()[1];

        let xstart = min(xlo + xhalo, nrows) / xblock;
        let xend = div_ceil(xhi.saturating_sub(xhalo), xblock);

        let ystart = min(ylo + yhalo, ncols) / yblock;
        let yend = div_ceil(yhi.saturating_sub(yhalo), yblock);

        if xstart + 1 == xend && ystart + 1 == yend {
            let index = xstart * yn + ystart;

            let xchunk = (xstart * xblock).saturating_sub(xhalo);
            let ychunk = (ystart * yblock).saturating_sub(yhalo);
            let chunk_start = Point::from((xchunk, ychunk));

            Some((index as usize, chunk_start))
        } else {
            None
        }
    }

    fn query_tiles<F, const INCLUDE_HALOS: bool>(&self, region: Rect, mut callback: F)
    where
        F: FnMut(usize, Point, Rect),
    {
        let [xsize, ysize] = self.size;
        let [_, yn] = self.num_blocks;
        let [xblock, yblock] = self.block_size;
        let [xhalo, yhalo] = self.halo_size;

        let xlo = region.low()[0];
        let xhi = region.high()[0];
        let ylo = region.low()[1];
        let yhi = region.high()[1];

        let xstart;
        let xend;
        let ystart;
        let yend;

        if INCLUDE_HALOS {
            xstart = (xlo.saturating_sub(xhalo)) / xblock;
            xend = div_ceil(min(xhi + xhalo, xsize), xblock);

            ystart = (ylo.saturating_sub(yhalo)) / yblock;
            yend = div_ceil(min(yhi + yhalo, ysize), yblock);
        } else {
            xstart = xlo / xblock;
            xend = div_ceil(xhi, xblock);

            ystart = ylo / yblock;
            yend = div_ceil(yhi, yblock);
        }

        for i in xstart..xend {
            for j in ystart..yend {
                let index = (i * yn + j) as usize;

                let xstart = (i * xblock).saturating_sub(xhalo);
                let ystart = (j * yblock).saturating_sub(yhalo);

                let x0;
                let x1;
                let y0;
                let y1;

                if INCLUDE_HALOS {
                    x0 = max(xlo, xstart);
                    x1 = min(xhi, (i + 1) * xblock + xhalo);

                    y0 = max(ylo, ystart);
                    y1 = min(yhi, (j + 1) * yblock + yhalo);
                } else {
                    x0 = max(xlo, i * xblock);
                    x1 = min(xhi, (i + 1) * xblock);

                    y0 = max(ylo, j * yblock);
                    y1 = min(yhi, (j + 1) * yblock);
                }

                (callback)(
                    index,
                    Point::new(xstart, ystart, 0),
                    Rect::from_bounds(Point::new(x0, y0, 0), Point::new(x1, y1, 1)),
                );
            }
        }
    }
}

impl WorkDistribution for Stencil2DDistribution {
    fn query_point(&self, p: Point) -> ExecutorId {
        let [_, yn] = self.num_blocks;
        let [xblock, yblock] = self.block_size;

        let x = p[0] / xblock;
        let y = p[1] / yblock;

        let index = x * yn + y;
        self.places[index as usize].best_affinity_executor()
    }

    fn query_region(&self, region: Rect) -> Vec<(ExecutorId, Rect)> {
        let mut results = vec![];
        self.query_tiles::<_, false>(region, |index, _chunk_start, overlap| {
            results.push((self.places[index].best_affinity_executor(), overlap));
        });

        results
    }
}

impl DataDistribution for Stencil2DDistribution {
    fn as_work_distribution(&self) -> Option<&dyn WorkDistribution> {
        Some(self)
    }

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

    fn validate_stencil2d_distribution(block_size: [u64; 2], halo_size: [u64; 2], size: [u64; 2]) {
        let num_blocks = [
            div_ceil(size[0], block_size[0]),
            div_ceil(size[1], block_size[1]),
        ];

        let places = (0..(num_blocks[0] * num_blocks[1]))
            .map(|_| MemoryId::new(WorkerId(0), MemoryKind::Host))
            .collect_vec();

        let dist = Stencil2DDistribution {
            num_blocks,
            block_size,
            halo_size,
            size,
            places,
        };

        let mut chunks = vec![];

        for i in 0..num_blocks[0] {
            for j in 0..num_blocks[1] {
                let xstart = i * block_size[0];
                let xend = min((i + 1) * block_size[0], size[0]);

                let ystart = j * block_size[1];
                let yend = min((j + 1) * block_size[1], size[1]);

                let xstart_with_halo = xstart.saturating_sub(halo_size[0]);
                let xend_with_halo = min(xend + halo_size[0], size[0]);

                let ystart_with_halo = ystart.saturating_sub(halo_size[1]);
                let yend_with_halo = min(yend + halo_size[1], size[1]);

                chunks.push(ReferenceChunk {
                    subregion: (
                        xstart_with_halo..ystart_with_halo,
                        xend_with_halo..yend_with_halo,
                    )
                        .into(),
                    owned_subregion: (xstart..xend, ystart..yend).into(),
                });
            }
        }

        validate_distribution(Dim::from(size), &chunks, &dist);
    }

    #[test]
    fn test_stencil2d_distribution() {
        validate_stencil2d_distribution([25, 25], [1, 1], [150, 150]);
        validate_stencil2d_distribution([26, 26], [2, 2], [150, 150]);
        validate_stencil2d_distribution([24, 24], [3, 3], [150, 150]);

        validate_stencil2d_distribution([25, 25], [1, 1], [125, 150]);
        validate_stencil2d_distribution([25, 26], [1, 1], [125, 150]);
        validate_stencil2d_distribution([25, 26], [1, 2], [125, 150]);
        validate_stencil2d_distribution([25, 26], [0, 2], [125, 150]);
        validate_stencil2d_distribution([25, 26], [1, 3], [120, 150]);
    }
}
