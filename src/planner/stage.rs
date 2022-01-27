use bumpalo::collections::Vec as BumpVec;
use bumpalo::Bump;
use lightning_core::util::GroupByExt;
use smallvec::SmallVec;

use super::distribution::ChunkQueryResult;
use crate::driver::Plan;
use crate::planner::task::{FillTasklet, FoldTasklet, ReduceTasklet};
use crate::planner::{ArrayChunkId, ArrayId, ArrayMeta, ChunkAccess, ChunkMeta, Planner};
use crate::prelude::*;
use crate::types::dag::{EventList, OperationChunk};
use crate::types::{
    Affine, ChunkId, ChunkLayout, DataValue, Dim, EventId, MemoryId, MemoryKind, Point, Rect,
    Reduction, ReductionFunction, WorkerId, MAX_DIMS,
};

#[derive(PartialEq, Eq, Hash, Debug, PartialOrd, Ord, Clone, Copy)]
pub(super) struct RectId(u16);

#[derive(Default)]
struct RectCache(IndexSet<Rect>);

impl RectCache {
    fn put(&mut self, r: Rect) -> RectId {
        let (index, _) = self.0.insert_full(r);
        RectId(index.try_into().expect("too many rects"))
    }

    fn get(&self, index: RectId) -> Rect {
        (self.0)[index.0 as usize]
    }
}

#[derive(PartialEq, Eq, Hash)]
struct CacheKey {
    array_id: ArrayId,
    region: RectId,
    node_id: WorkerId,
}

struct TemporaryChunk<'b> {
    id: ChunkId,
    dep: EventId,
    uses: BumpVec<'b, EventId>,
}

#[derive(PartialEq, Eq, Hash)]
struct ReductionKey {
    array_id: ArrayId,
    region: RectId,
    reduction: Reduction,
}

struct ReductionFragment<'b> {
    pieces: &'b [ChunkQueryResult],
    replicas: HashMap<MemoryId, TemporaryChunk<'b>>,
}

#[derive(Debug)]
pub(super) struct ScatterPiece {
    chunk_index: usize,
    src_region: RectId,
    dst_offset: Point,
}

pub(super) enum UnmapAction<'a> {
    Read {
        array_id: ArrayId,
        chunk_idx: usize,
    },
    CacheRead {
        cache_idx: usize,
    },
    Write {
        array_id: ArrayId,
        chunk_idx: usize,
    },
    ScatterWrite {
        array_id: ArrayId,
        id: ChunkId,
        chunk_idx: Option<usize>,
        layout: ChunkLayout,
        pieces: BumpVec<'a, ScatterPiece>,
    },
    Reduction {
        reduction_idx: usize,
        place: MemoryId,
    },
    ReplicatedReduction {
        place: MemoryId,
        array_id: ArrayId,
        region: RectId,
        reduction: Reduction,
        num_replicas: u64,
        chunk_id: ChunkId,
    },
}

pub(crate) struct PlannerStage<'planner, 'alloc, 'plan> {
    pub(super) plan: &'plan mut Plan,
    pub(super) planner: &'planner Planner,
    chunks_accessed: BumpVec<'alloc, ChunkAccess>,
    cached_regions: IndexMap<CacheKey, TemporaryChunk<'alloc>>,
    reductions: IndexMap<ReductionKey, ReductionFragment<'alloc>>,
    rects: RectCache,
    alloc: &'alloc Bump,
}

impl<'planner, 'alloc, 'plan> PlannerStage<'planner, 'alloc, 'plan> {
    pub(super) fn new(
        plan: &'plan mut Plan,
        planner: &'planner Planner,
        alloc: &'alloc Bump,
    ) -> Self {
        Self {
            plan,
            planner,
            chunks_accessed: BumpVec::new_in(&alloc),
            cached_regions: default(),
            reductions: default(),
            rects: default(),
            alloc: &alloc,
        }
    }

    pub(super) fn commit_cached_chunks(&mut self) {
        for (_, fragment) in &self.cached_regions {
            self.plan.destroy_chunk(fragment.id, &*fragment.uses);
        }

        self.cached_regions.clear();
    }

    pub(super) fn add_access(&mut self, id: ArrayChunkId, task_id: EventId, is_write: bool) {
        self.plan.add_terminal(task_id);

        self.chunks_accessed.push(ChunkAccess {
            array_id: id.0,
            chunk_idx: id.1,
            task_id,
            is_write,
        });
    }

    pub(super) fn commit_accesses(&mut self) -> BumpVec<'alloc, ChunkAccess> {
        replace(&mut self.chunks_accessed, BumpVec::new_in(&self.alloc))
    }

    pub fn add_sync(&mut self, id: ArrayId, region: Rect) -> Result {
        let array = self.planner.array(id)?;
        let chunks = &*array.chunks;

        array.distribution.visit_replicated(region, &mut |r| {
            let chunk = &chunks[r.chunk_index];

            let dep = self.plan.add_terminal(chunk.last_readwrite);
            self.add_access(chunk.key, dep, false);
        });

        Ok(())
    }

    pub fn add_copy(
        &mut self,
        src_id: ArrayId,
        src_offset: Point,
        dst_id: ArrayId,
        dst_offset: Point,
        extents: Dim,
    ) -> Result {
        let src_array = self.planner.array(src_id)?;
        let dst_array = self.planner.array(dst_id)?;

        let chunks = &*dst_array.chunks;
        let dst_region = Rect::new(dst_offset, extents);

        dst_array
            .distribution
            .visit_replicated(dst_region, &mut |dst| {
                let dst_chunk = &chunks[dst.chunk_index];

                let region = Rect::new(src_offset + dst.region_offset, dst.extents);
                let place = MemoryId::new(dst_chunk.owner, dst_chunk.affinity);
                let chunks = &*src_array.chunks;

                src_array
                    .distribution
                    .visit_unique(region, Some(place), &mut |src| {
                        let src_chunk = &chunks[src.chunk_index];

                        let copy_op = self.copy_from_chunk_cached(
                            src_array,
                            src_offset + src.region_offset + dst.region_offset,
                            src_chunk,
                            src.chunk_offset,
                            dst_chunk.id,
                            dst.chunk_offset + src.region_offset,
                            &dst_chunk.layout,
                            dst_chunk.last_readwrite,
                            src.extents,
                        );

                        self.plan.add_terminal(copy_op);
                        self.add_access(dst_chunk.key, copy_op, true);
                    });
            });

        Ok(())
    }

    pub fn add_fill(&mut self, id: ArrayId, region: Rect, value: DataValue) -> Result {
        let handle = self.planner.array(id)?;
        let chunks = &*handle.chunks;

        if handle.dtype != value.data_type() {
            bail!(
                "data type mismatch: {:?} != {:?}",
                handle.dtype,
                value.data_type()
            );
        }

        let mut result = Ok(());

        handle.distribution.visit_replicated(region, &mut |r| {
            let chunk = &chunks[r.chunk_index];

            let fill_op = self.plan.add_tasklet(
                chunk.owner,
                chunk.affinity.best_affinity_executor(),
                &FillTasklet {
                    transform: Affine::add_offset(r.chunk_offset),
                    domain: r.extents,
                    value: value.clone(),
                },
                [OperationChunk {
                    id: chunk.id,
                    exclusive: true,
                    dependency: Some(chunk.last_readwrite),
                }],
            );

            match fill_op {
                Ok(op) => self.add_access(chunk.key, op, true),
                Err(e) => result = Err(e),
            }
        });

        result
    }

    pub(super) fn unmap_array(&mut self, exe_op: EventId, action: UnmapAction<'_>) -> Result {
        use UnmapAction::*;

        match action {
            Read {
                array_id,
                chunk_idx,
            } => {
                return self.unmap_read(array_id, exe_op, chunk_idx);
            }
            Write {
                array_id,
                chunk_idx,
            } => {
                return self.unmap_write(array_id, exe_op, chunk_idx);
            }
            CacheRead { cache_idx } => {
                return self.unmap_cached_read(exe_op, cache_idx);
            }
            ScatterWrite {
                array_id,
                id,
                chunk_idx,
                layout,
                pieces,
            } => {
                return self.unmap_scatter(array_id, exe_op, id, chunk_idx, layout, pieces);
            }
            Reduction {
                reduction_idx,
                place,
            } => {
                return self.unmap_reduction(exe_op, reduction_idx, place);
            }
            ReplicatedReduction {
                place,
                array_id,
                region,
                reduction,
                num_replicas,
                chunk_id,
            } => {
                return self.unmap_replicated_reduce(
                    exe_op,
                    place,
                    array_id,
                    self.rects.get(region),
                    reduction,
                    num_replicas,
                    chunk_id,
                );
            }
        }
    }

    fn create_assemble_chunk(
        &mut self,
        node_id: WorkerId,
        array: &ArrayMeta,
        region: Rect,
        pieces: &[ChunkQueryResult],
    ) -> (ChunkId, ChunkLayout, EventId) {
        let layout = array
            .layout
            .build(array.dtype, region.extents(), MemoryKind::Host);
        let (tmp_id, create_op) = self.plan.create_chunk(node_id, layout.clone());
        if pieces.is_empty() {
            return (tmp_id, layout, create_op);
        }

        let chunks = &*array.chunks;
        let mut deps = SmallVec::with_capacity(pieces.len());

        for piece in pieces {
            let src_chunk = &chunks[piece.chunk_index];

            let dep = self.copy_from_chunk_cached(
                array,
                region.low() + piece.region_offset,
                src_chunk,
                piece.chunk_offset,
                tmp_id,
                piece.region_offset,
                &layout,
                create_op,
                piece.extents,
            );

            deps.push(dep);
        }

        let dep = self.plan.join(node_id, deps);
        (tmp_id, layout, dep)
    }

    fn scatter_chunk(
        &mut self,
        src_id: ChunkId,
        src_layout: &ChunkLayout,
        src_dep: EventId,
        dst_array: &ArrayMeta,
        dst_pieces: &mut [ScatterPiece],
    ) -> EventList {
        let mut deps = SmallVec::with_capacity(dst_pieces.len());
        let dst_chunks = &*dst_array.chunks;

        for (src_region, dst_pieces) in dst_pieces.sort_and_group_by_key(|e| e.src_region) {
            let src_region = self.rects.get(src_region);
            let src_owner = src_id.owner();
            let src_offset = src_region.low();
            let extents = src_region.extents();
            let (local_pieces, remote_pieces) =
                partition::partition(dst_pieces, |e| dst_chunks[e.chunk_index].owner == src_owner);

            // Local pieces is easy, simply copy them from src_chunk to dst_chunk directly.
            for local_piece in local_pieces {
                let dst_chunk = &dst_chunks[local_piece.chunk_index];

                let copy_op = self.plan.add_copy(
                    src_id,
                    Affine::add_offset(src_offset),
                    src_dep,
                    dst_chunk.id,
                    Affine::add_offset(local_piece.dst_offset),
                    dst_chunk.last_readwrite,
                    extents,
                );

                self.add_access(dst_chunk.key, copy_op, true);
                deps.push(copy_op);
            }

            // Remote chunks are more complicated.
            if remote_pieces.len() > 0 {
                deps.extend(self.scatter_chunk_remote(
                    src_id,
                    src_layout,
                    src_offset,
                    extents,
                    src_dep,
                    dst_array,
                    remote_pieces,
                ));
            }
        }

        deps
    }

    fn scatter_chunk_remote(
        &mut self,
        src_id: ChunkId,
        src_layout: &ChunkLayout,
        src_offset: Point,
        extents: Dim,
        src_dep: EventId,
        dst_array: &ArrayMeta,
        dst_pieces: &mut [ScatterPiece],
    ) -> EventList {
        let src_owner = src_id.owner();

        // If copy region does not cover the entire chunk, then we must copy it to temporary chunk.
        if src_layout.size != extents {
            let tmp_layout = dst_array
                .layout
                .build(dst_array.dtype, extents, MemoryKind::Host);
            let (tmp_id, create_op) = self.plan.create_chunk(src_owner, tmp_layout.clone());

            let copy_op = self.plan.add_copy(
                src_id,
                Affine::add_offset(src_offset),
                src_dep,
                tmp_id,
                Affine::identity(),
                create_op,
                extents,
            );

            let deps = self.scatter_chunk_remote(
                tmp_id,
                &tmp_layout,
                Point::zeros(),
                extents,
                copy_op,
                dst_array,
                dst_pieces,
            );

            self.plan.destroy_chunk(tmp_id, deps);
            return EventList::from(&[copy_op][..]);
        }

        assert_eq!(src_offset, Point::zeros());
        let dst_chunks = &*dst_array.chunks;
        let mut deps = EventList::with_capacity(dst_pieces.len());

        // Group pieces by owner and call "scatter_chunk_remote_single" for each worker
        for (dst_owner, dst_pieces) in
            dst_pieces.sort_and_group_by_key(|e| dst_chunks[e.chunk_index].owner)
        {
            deps.push(self.scatter_chunk_remote_single(
                src_id, extents, src_dep, dst_array, dst_owner, dst_pieces,
            ));
        }

        deps
    }

    fn scatter_chunk_remote_single(
        &mut self,
        src_id: ChunkId,
        extents: Dim,
        src_dep: EventId,
        dst_array: &ArrayMeta,
        dst_owner: WorkerId,
        dst_pieces: &mut [ScatterPiece],
    ) -> EventId {
        let dst_chunks = &*dst_array.chunks;

        // If there is only 1 piece and this piece covers the entire chunk, then we can
        // sendrecv the piece directly.
        if dst_pieces.len() == 1 {
            let piece = &dst_pieces[0];
            let dst_chunk = &dst_chunks[piece.chunk_index];

            if dst_chunk.layout.size == extents {
                assert_eq!(piece.dst_offset, Point::zeros());
                let (send_op, recv_op) =
                    self.plan
                        .add_sendrecv(src_id, src_dep, dst_chunk.id, dst_chunk.last_readwrite);
                self.add_access(dst_chunk.key, recv_op, true);
                return send_op;
            }
        }

        // Otherwise, we create a temporary chunk to receive the data and copy it to each piece.
        let tmp_layout = dst_array
            .layout
            .build(dst_array.dtype, extents, MemoryKind::Host);
        let (tmp_id, create_op) = self.plan.create_chunk(dst_owner, tmp_layout.clone());
        let (send_op, recv_op) = self.plan.add_sendrecv(src_id, src_dep, tmp_id, create_op);

        let mut deps = SmallVec::with_capacity(dst_pieces.len());

        for piece in dst_pieces {
            let dst_chunk = &dst_chunks[piece.chunk_index];

            let copy_op = self.plan.add_copy(
                tmp_id,
                Affine::identity(),
                recv_op,
                dst_chunk.id,
                Affine::add_offset(piece.dst_offset),
                dst_chunk.last_readwrite,
                extents,
            );

            deps.push(copy_op);
            self.add_access(dst_chunk.key, copy_op, true);
        }

        self.plan.destroy_chunk(tmp_id, deps);

        return send_op;
    }

    fn copy_from_chunk_cached(
        &mut self,
        src_array: &ArrayMeta,
        src_array_offset: Point,
        src_chunk: &ChunkMeta,
        src_offset: Point,
        dst_id: ChunkId,
        dst_offset: Point,
        dst_layout: &ChunkLayout,
        dst_dep: EventId,
        extents: Dim,
    ) -> EventId {
        let src_id = src_chunk.id;
        let src_owner = src_chunk.owner;
        let dst_owner = dst_id.owner();

        if src_owner == dst_owner {
            let copy_id = self.plan.add_copy(
                src_id,
                Affine::add_offset(src_offset),
                src_chunk.last_write,
                dst_id,
                Affine::add_offset(dst_offset),
                dst_dep,
                extents,
            );

            self.add_access(src_chunk.key, copy_id, false);
            return copy_id;
        }

        if extents != dst_layout.size {
            let key = CacheKey {
                array_id: src_array.id,
                region: self.rects.put(Rect::new(src_array_offset, extents)),
                node_id: dst_id.owner(),
            };

            let fragment = if let Some(fragment) = self.cached_regions.get_mut(&key) {
                fragment
            } else {
                let tmp_layout = src_array
                    .layout
                    .build(src_array.dtype, extents, MemoryKind::Host);
                let (tmp_id, tmp_op) = self.plan.create_chunk(dst_id.owner(), tmp_layout.clone());
                let copy_op = self.copy_from_chunk_cached(
                    src_array,
                    src_array_offset,
                    src_chunk,
                    src_offset,
                    tmp_id,
                    Point::zeros(),
                    &tmp_layout,
                    tmp_op,
                    extents,
                );

                self.cached_regions.entry(key).or_insert(TemporaryChunk {
                    id: tmp_id,
                    dep: copy_op,
                    uses: BumpVec::new_in(&self.alloc),
                })
            };

            let copy_op = self.plan.add_copy(
                fragment.id,
                Affine::identity(),
                fragment.dep,
                dst_id,
                Affine::add_offset(dst_offset),
                dst_dep,
                extents,
            );

            fragment.uses.push(copy_op);
            return copy_op;
        }

        if extents != src_chunk.layout.size {
            let key = CacheKey {
                array_id: src_array.id,
                region: self.rects.put(Rect::new(src_array_offset, extents)),
                node_id: src_owner,
            };

            let fragment = if let Some(fragment) = self.cached_regions.get_mut(&key) {
                fragment
            } else {
                let tmp_layout = src_array
                    .layout
                    .build(src_array.dtype, extents, MemoryKind::Host);
                let (tmp_id, tmp_op) = self.plan.create_chunk(src_owner, tmp_layout.clone());
                let copy_op = self.plan.add_copy(
                    src_id,
                    Affine::add_offset(src_offset),
                    src_chunk.last_write,
                    tmp_id,
                    Affine::identity(),
                    tmp_op,
                    extents,
                );

                self.add_access(src_chunk.key, copy_op, false);

                self.cached_regions.entry(key).or_insert(TemporaryChunk {
                    id: tmp_id,
                    dep: copy_op,
                    uses: BumpVec::new_in(&self.alloc),
                })
            };

            let (send_op, recv_op) =
                self.plan
                    .add_sendrecv(fragment.id, fragment.dep, dst_id, dst_dep);

            fragment.uses.push(send_op);
            return recv_op;
        }

        assert_eq!(src_offset, Point::zeros());
        assert_eq!(dst_offset, Point::zeros());

        let (send_op, recv_op) =
            self.plan
                .add_sendrecv(src_id, src_chunk.last_write, dst_id, dst_dep);

        self.add_access(src_chunk.key, send_op, false);
        recv_op
    }

    pub(super) fn map_read(
        &mut self,
        place: MemoryId,
        array_id: ArrayId,
        region: Rect,
    ) -> Result<(OperationChunk, Point, UnmapAction<'alloc>)> {
        let array = self.planner.array(array_id)?;
        let node_id = place.node_id();
        let mut pieces = BumpVec::with_capacity_in(1, &self.alloc);

        array
            .distribution
            .visit_unique(region, Some(place), &mut |r| {
                pieces.push(r);
            });

        let chunks = &*array.chunks;
        if pieces.len() == 1 {
            let piece = pieces[0];
            let chunk = &chunks[piece.chunk_index];

            if chunk.owner == node_id {
                return Ok((
                    OperationChunk {
                        id: chunk.id,
                        exclusive: false,
                        dependency: Some(chunk.last_write),
                    },
                    piece.chunk_offset,
                    UnmapAction::Read {
                        array_id,
                        chunk_idx: piece.chunk_index,
                    },
                ));
            }
        }

        let key = CacheKey {
            node_id,
            array_id: array.id,
            region: self.rects.put(region),
        };

        let (cache_idx, fragment) =
            if let Some((index, _, fragment)) = self.cached_regions.get_full(&key) {
                (index, fragment)
            } else {
                let (tmp_id, _, dep) = self.create_assemble_chunk(node_id, array, region, &pieces);

                let entry = self.cached_regions.entry(key);
                let index = entry.index();
                let fragment = entry.or_insert(TemporaryChunk {
                    id: tmp_id,
                    dep,
                    uses: BumpVec::new_in(&self.alloc),
                });

                (index, &*fragment)
            };

        Ok((
            OperationChunk {
                id: fragment.id,
                exclusive: false,
                dependency: Some(fragment.dep),
            },
            Point::zeros(),
            UnmapAction::CacheRead { cache_idx },
        ))
    }

    fn unmap_read(&mut self, array_id: ArrayId, exe_op: EventId, chunk_idx: usize) -> Result {
        self.add_access(ArrayChunkId(array_id, chunk_idx), exe_op, false);
        Ok(())
    }

    fn unmap_cached_read(&mut self, exe_op: EventId, cache_idx: usize) -> Result {
        self.cached_regions[cache_idx].uses.push(exe_op);
        Ok(())
    }

    pub(super) fn map_readwrite(
        &mut self,
        place: MemoryId,
        array_id: ArrayId,
        region: Rect,
    ) -> Result<(OperationChunk, Point, UnmapAction<'alloc>)> {
        self.map_readwrite_or_write(place, array_id, region, false)
    }

    pub(super) fn map_write(
        &mut self,
        place: MemoryId,
        array_id: ArrayId,
        region: Rect,
    ) -> Result<(OperationChunk, Point, UnmapAction<'alloc>)> {
        self.map_readwrite_or_write(place, array_id, region, true)
    }

    fn map_readwrite_or_write(
        &mut self,
        place: MemoryId,
        array_id: ArrayId,
        region: Rect,
        write_only: bool,
    ) -> Result<(OperationChunk, Point, UnmapAction<'alloc>)> {
        let array = self.planner.array(array_id)?;
        let node_id = place.node_id();
        let mut pieces = BumpVec::with_capacity_in(1, &self.alloc);
        let chunks = &*array.chunks;

        array
            .distribution
            .visit_unique(region, Some(place), &mut |r| {
                pieces.push(r);
            });

        let mut is_local = false;
        if pieces.len() == 1 {
            let piece = pieces[0];
            let chunk = &chunks[piece.chunk_index];

            if chunk.owner == place.node_id()
                && (chunk.affinity == place.kind() || chunk.layout.size == piece.extents)
            {
                is_local = true;
            }
        }

        let (id, chunk_index, layout, dep, offset) = if is_local {
            let piece = pieces[0];
            let chunk = &chunks[piece.chunk_index];

            (
                chunk.id,
                Some(piece.chunk_index),
                chunk.layout.clone(),
                chunk.last_readwrite,
                piece.chunk_offset,
            )
        } else {
            let assemble_pieces = match write_only {
                true => &[][..],
                false => &pieces,
            };

            let (id, layout, dep) =
                self.create_assemble_chunk(node_id, array, region, assemble_pieces);
            (id, None, layout, dep, Point::zeros())
        };

        let mut pieces = BumpVec::with_capacity_in(1, &self.alloc);

        array.distribution.visit_replicated(region, &mut |r| {
            if Some(r.chunk_index) == chunk_index {
                return;
            }

            let src_region = self
                .rects
                .put(Rect::new(r.region_offset + offset, r.extents));

            pieces.push(ScatterPiece {
                chunk_index: r.chunk_index,
                dst_offset: r.chunk_offset,
                src_region,
            });
        });

        let unmapping = if let (Some(chunk_idx), []) = (chunk_index, &*pieces) {
            UnmapAction::Write {
                array_id,
                chunk_idx,
            }
        } else {
            UnmapAction::ScatterWrite {
                array_id,
                chunk_idx: chunk_index,
                id,
                layout,
                pieces,
            }
        };

        Ok((
            OperationChunk {
                id,
                exclusive: true,
                dependency: Some(dep),
            },
            offset,
            unmapping,
        ))
    }

    fn unmap_write(&mut self, array_id: ArrayId, exe_op: EventId, chunk_idx: usize) -> Result {
        self.add_access(ArrayChunkId(array_id, chunk_idx), exe_op, true);
        Ok(())
    }

    fn unmap_scatter(
        &mut self,
        array_id: ArrayId,
        exe_op: EventId,
        id: ChunkId,
        chunk_idx: Option<usize>,
        layout: ChunkLayout,
        mut pieces: BumpVec<'_, ScatterPiece>,
    ) -> Result {
        let deps = self.scatter_chunk(
            id,
            &layout,
            exe_op,
            self.planner.array(array_id).unwrap(),
            &mut pieces,
        );

        let dep = if deps.len() > 0 {
            self.plan.join(id.owner(), deps)
        } else {
            exe_op
        };

        if let Some(chunk_idx) = chunk_idx {
            self.add_access(ArrayChunkId(array_id, chunk_idx), dep, true);
        } else {
            self.plan.destroy_chunk(id, &[dep][..]);
        }

        Ok(())
    }

    fn create_identity_chunk(
        plan: &mut Plan,
        place: MemoryId,
        extents: Dim,
        array: &ArrayMeta,
        reduction: Reduction,
    ) -> Result<(ChunkId, EventId)> {
        assert_eq!(array.dtype, reduction.data_type());

        let dtype = array.dtype;
        let value = reduction.identity();

        let (chunk_id, create_op) = plan.create_chunk(
            place.node_id(),
            array.layout.build(dtype, extents, place.kind()),
        );

        let fill_op = plan.add_tasklet(
            place.node_id(),
            place.kind().best_affinity_executor(),
            &FillTasklet {
                transform: Affine::identity(),
                domain: extents,
                value,
            },
            [OperationChunk {
                id: chunk_id,
                exclusive: true,
                dependency: Some(create_op),
            }],
        )?;

        Ok((chunk_id, fill_op))
    }

    fn fold_chunks(
        plan: &mut Plan,
        node_id: WorkerId,
        place: MemoryKind,
        extents: Dim,
        reduction: Reduction,
        src_chunk: ChunkId,
        src_dep: EventId,
        src_offset: Point,
        dst_chunk: ChunkId,
        dst_dep: EventId,
        dst_offset: Point,
    ) -> Result<EventId> {
        plan.add_tasklet(
            node_id,
            place.best_affinity_executor(),
            &FoldTasklet {
                src_transform: Affine::add_offset(src_offset),
                dst_transform: Affine::add_offset(dst_offset),
                extents,
                reduction,
            },
            [
                OperationChunk {
                    id: src_chunk,
                    exclusive: false,
                    dependency: Some(src_dep),
                },
                OperationChunk {
                    id: dst_chunk,
                    exclusive: true,
                    dependency: Some(dst_dep),
                },
            ],
        )
    }

    pub(super) fn map_replicated_reduce(
        &mut self,
        place: MemoryId,
        array_id: ArrayId,
        region: Rect,
        reduction: ReductionFunction,
        num_replicas: u64,
    ) -> Result<(OperationChunk, Point, UnmapAction<'alloc>)> {
        let array = self.planner.array(array_id)?;

        let reduction = Reduction::new(reduction, array.dtype).ok_or_else(|| {
            anyhow!(
                "reduction {:?} not supported for data type {:?}",
                reduction,
                array.dtype
            )
        })?;

        let mut extents = region.extents();
        extents[MAX_DIMS - 1] *= num_replicas;

        let (chunk_id, dep) = self.plan.create_chunk(
            place.node_id(),
            array.layout.build(array.dtype, extents, place.kind()),
        );

        Ok((
            OperationChunk {
                id: chunk_id,
                exclusive: true,
                dependency: Some(dep),
            },
            Point::zeros(),
            UnmapAction::ReplicatedReduction {
                place,
                array_id,
                region: self.rects.put(region),
                reduction,
                num_replicas,
                chunk_id,
            },
        ))
    }

    pub(super) fn unmap_replicated_reduce(
        &mut self,
        exe_op: EventId,
        place: MemoryId,
        array_id: ArrayId,
        region: Rect,
        reduction: Reduction,
        num_replicas: u64,
        chunk_id: ChunkId,
    ) -> Result {
        use crate::types::{AffineNM, TransformNM};
        let (dst_chunk, dst_offset, action) =
            self.map_reduce(place, array_id, region, reduction.function())?;

        let mut extents = region.extents().resize::<{ MAX_DIMS + 1 }>(0);
        extents[MAX_DIMS] = num_replicas;

        let mut transform = TransformNM::<{ MAX_DIMS + 1 }, MAX_DIMS>::identity();
        transform[MAX_DIMS - 1][MAX_DIMS] = extents[MAX_DIMS - 1] as i64;

        let src_transform = AffineNM::<{ MAX_DIMS + 1 }, MAX_DIMS>::from(transform);
        let dst_transform = AffineNM::<MAX_DIMS, MAX_DIMS>::add_offset(dst_offset);

        let reduce_op = self.plan.add_tasklet(
            place.node_id(),
            place.best_affinity_executor().kind(),
            &ReduceTasklet {
                src_transform,
                dst_transform,
                axis: MAX_DIMS,
                reduction,
                extents,
            },
            [
                OperationChunk {
                    id: chunk_id,
                    exclusive: false,
                    dependency: Some(exe_op),
                },
                dst_chunk,
            ],
        )?;

        self.plan.destroy_chunk(chunk_id, &[reduce_op][..]);
        self.unmap_array(reduce_op, action)
    }

    pub(super) fn map_reduce(
        &mut self,
        place: MemoryId,
        array_id: ArrayId,
        region: Rect,
        reduction: ReductionFunction,
    ) -> Result<(OperationChunk, Point, UnmapAction<'alloc>)> {
        use indexmap::map::Entry as IEntry;
        use std::collections::hash_map::Entry as HEntry;

        let array = self.planner.array(array_id)?;

        let reduction = Reduction::new(reduction, array.dtype).ok_or_else(|| {
            anyhow!(
                "reduction {:?} not supported for data type {:?}",
                reduction,
                array.dtype
            )
        })?;

        let key = ReductionKey {
            array_id: array.id,
            region: self.rects.put(region),
            reduction,
        };

        let (reduction_idx, entry) = match self.reductions.entry(key) {
            IEntry::Occupied(e) => (e.index(), e.into_mut()),
            IEntry::Vacant(e) => {
                let mut pieces = BumpVec::new_in(&self.alloc);
                array.distribution.visit_replicated(region, &mut |r| {
                    pieces.push(r);
                });

                (
                    e.index(),
                    e.insert(ReductionFragment {
                        pieces: pieces.into_bump_slice(),
                        replicas: default(),
                    }),
                )
            }
        };

        if entry.pieces.len() == 1 {
            let piece = entry.pieces[0];
            let chunk = &array.chunks[piece.chunk_index];

            if chunk.owner == place.node_id()
                && (chunk.affinity == place.kind() || chunk.layout.size == piece.extents)
            {
                return Ok((
                    OperationChunk {
                        id: chunk.id,
                        exclusive: true,
                        dependency: Some(chunk.last_readwrite),
                    },
                    piece.chunk_offset,
                    UnmapAction::Write {
                        array_id,
                        chunk_idx: piece.chunk_index,
                    },
                ));
            }
        }

        let entry = match entry.replicas.entry(place) {
            HEntry::Occupied(e) => e.into_mut(),
            HEntry::Vacant(e) => {
                let (id, dep) = Self::create_identity_chunk(
                    &mut self.plan,
                    place,
                    region.extents(),
                    array,
                    reduction,
                )?;

                let uses = BumpVec::new_in(&self.alloc);
                e.insert(TemporaryChunk { id, dep, uses })
            }
        };

        Ok((
            OperationChunk {
                id: entry.id,
                exclusive: true,
                dependency: Some(entry.dep),
            },
            Point::zeros(),
            UnmapAction::Reduction {
                reduction_idx,
                place,
            },
        ))
    }

    fn unmap_reduction(
        &mut self,
        exe_op: EventId,
        reduction_idx: usize,
        place: MemoryId,
    ) -> Result {
        self.reductions[reduction_idx]
            .replicas
            .get_mut(&place)
            .unwrap()
            .uses
            .push(exe_op);

        Ok(())
    }

    pub(super) fn commit_reductions(&mut self) -> Result {
        for (key, fragment) in take(&mut self.reductions) {
            let region = self.rects.get(key.region);

            self.commit_reduction(
                key.array_id,
                key.reduction,
                region.extents(),
                fragment.pieces,
                fragment.replicas,
            )?;
        }

        Ok(())
    }

    fn commit_reduction(
        &mut self,
        array_id: ArrayId,
        reduction: Reduction,
        extents: Dim,
        pieces: &[ChunkQueryResult],
        replicas: HashMap<MemoryId, TemporaryChunk<'alloc>>,
    ) -> Result {
        let mut items = replicas.into_iter().collect_vec();
        let array = self.planner.array(array_id)?;

        for (node_id, items) in items.sort_and_group_by_key(|(place, _)| place.node_id()) {
            self.commit_reduction_for_worker(node_id, array, reduction, extents, pieces, items)?;
        }

        Ok(())
    }

    fn commit_reduction_for_worker(
        &mut self,
        node_id: WorkerId,
        array: &ArrayMeta,
        reduction: Reduction,
        extents: Dim,
        pieces: &[ChunkQueryResult],
        items: &[(MemoryId, TemporaryChunk<'_>)],
    ) -> Result {
        if pieces.is_empty() || items.is_empty() {
            return Ok(());
        }

        let (accum_id, accum_dep) =
            self.fold_reductions_for_worker(node_id, array, reduction, extents, items)?;

        let chunks = &*array.chunks;
        let mut uses = SmallVec::with_capacity(pieces.len());

        for piece in pieces {
            let chunk = &chunks[piece.chunk_index];
            if chunk.owner == node_id {
                let fold_op = Self::fold_chunks(
                    &mut self.plan,
                    chunk.owner,
                    chunk.affinity,
                    piece.extents,
                    reduction,
                    accum_id,
                    accum_dep,
                    piece.region_offset,
                    chunk.id,
                    chunk.last_readwrite,
                    piece.chunk_offset,
                )?;

                self.add_access(chunk.key, fold_op, true);
                uses.push(fold_op);
            } else if piece.extents == extents {
                let (recv_buffer, recv_create) = self.plan.create_chunk(
                    chunk.owner,
                    array
                        .layout
                        .build(array.dtype, piece.extents, MemoryKind::Host),
                );

                let (send_op, recv_op) =
                    self.plan
                        .add_sendrecv(accum_id, accum_dep, recv_buffer, recv_create);

                let fold_op = Self::fold_chunks(
                    &mut self.plan,
                    chunk.owner,
                    chunk.affinity,
                    piece.extents,
                    reduction,
                    recv_buffer,
                    recv_op,
                    Point::zeros(),
                    chunk.id,
                    chunk.last_readwrite,
                    piece.chunk_offset,
                )?;

                self.plan.destroy_chunk(recv_buffer, &[fold_op][..]);

                uses.push(send_op);
                self.add_access(chunk.key, fold_op, true);
            } else {
                let (send_buffer, send_create) = self.plan.create_chunk(
                    node_id,
                    array
                        .layout
                        .build(array.dtype, piece.extents, MemoryKind::Host),
                );

                let copyout_op = self.plan.add_copy(
                    accum_id,
                    Affine::add_offset(piece.region_offset),
                    accum_dep,
                    send_buffer,
                    Affine::identity(),
                    send_create,
                    piece.extents,
                );

                let (recv_buffer, recv_create) = self.plan.create_chunk(
                    chunk.owner,
                    array
                        .layout
                        .build(array.dtype, piece.extents, MemoryKind::Host),
                );

                let (send_op, recv_op) =
                    self.plan
                        .add_sendrecv(send_buffer, copyout_op, recv_buffer, recv_create);

                self.plan.destroy_chunk(send_buffer, &[send_op][..]);

                let fold_op = Self::fold_chunks(
                    &mut self.plan,
                    chunk.owner,
                    chunk.affinity,
                    piece.extents,
                    reduction,
                    recv_buffer,
                    recv_op,
                    Point::zeros(),
                    chunk.id,
                    chunk.last_readwrite,
                    piece.chunk_offset,
                )?;

                self.plan.destroy_chunk(recv_buffer, &[fold_op][..]);

                uses.push(copyout_op);
                self.add_access(chunk.key, fold_op, true);
            };
        }

        self.plan.destroy_chunk(accum_id, uses);
        Ok(())
    }

    fn fold_reductions_for_worker(
        &mut self,
        node_id: WorkerId,
        array: &ArrayMeta,
        reduction: Reduction,
        extents: Dim,
        items: &[(MemoryId, TemporaryChunk<'_>)],
    ) -> Result<(ChunkId, EventId)> {
        assert!(items.len() > 0);

        if items.len() == 1 {
            let chunk = &items[0].1;
            let dep = self.plan.join(node_id, &*chunk.uses);
            return Ok((chunk.id, dep));
        }

        if items.len() == 2 {
            let src_chunk = &items[0].1;
            let dst_chunk = &items[1].1;

            let src_dep = self.plan.join(node_id, &*src_chunk.uses);
            let dst_dep = self.plan.join(node_id, &*dst_chunk.uses);

            let fold_op = Self::fold_chunks(
                &mut self.plan,
                node_id,
                MemoryKind::Host,
                extents,
                reduction,
                src_chunk.id,
                src_dep,
                default(),
                dst_chunk.id,
                dst_dep,
                default(),
            )?;

            self.plan.destroy_chunk(src_chunk.id, &[fold_op][..]);
            return Ok((dst_chunk.id, fold_op));
        }

        let mut uses = BumpVec::with_capacity_in(items.len(), &self.alloc);
        let (accum_chunk, accum_dep) = Self::create_identity_chunk(
            &mut self.plan,
            MemoryId::new(node_id, MemoryKind::Host),
            extents,
            array,
            reduction,
        )?;

        for (_, src_chunk) in items {
            let src_dep = self.plan.join(node_id, &*src_chunk.uses);

            let fold_op = Self::fold_chunks(
                &mut self.plan,
                node_id,
                MemoryKind::Host,
                extents,
                reduction,
                src_chunk.id,
                src_dep,
                default(),
                accum_chunk,
                accum_dep,
                default(),
            )?;

            uses.push(fold_op);
            self.plan.destroy_chunk(src_chunk.id, &[fold_op][..]);
        }

        let dep = self.plan.join(node_id, &*uses);
        Ok((accum_chunk, dep))
    }
}
