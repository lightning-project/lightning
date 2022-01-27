pub mod annotations;
pub(crate) mod cuda;
pub mod distribution;
mod stage;
pub(crate) mod task;

use bumpalo::Bump;
use lightning_core::util::GroupByExt;
use lightning_memops::{host_copy, SequentialPolicy};
use serde_bytes::ByteBuf;
use slotmap::SlotMap;
use smallvec::SmallVec;
use std::sync::Arc;

use self::distribution::{ChunkDescriptor, DataDistribution};
pub(crate) use self::stage::PlannerStage;
use crate::driver::{DriverEvent, DriverHandle, Plan};
use crate::planner::task::{ReadDataTasklet, WriteDataTasklet};
use crate::prelude::*;
use crate::types::dag::OperationChunk;
use crate::types::{
    ChunkId, ChunkLayout, ChunkLayoutBuilder, DataType, Dim, EventId, ExecutorKind, HostAccessor,
    HostMutAccessor, MemoryKind, Rect, WorkerId,
};

slotmap::new_key_type! {
    pub struct ArrayId;
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Copy, Clone)]
pub struct ArrayChunkId(ArrayId, usize);

#[derive(Debug)]
struct ArrayMeta {
    id: ArrayId,
    size: Dim,
    dtype: DataType,
    distribution: Arc<dyn DataDistribution>,
    layout: ChunkLayoutBuilder,
    chunks: Box<[ChunkMeta]>,
}

#[derive(Debug)]
struct ChunkMeta {
    key: ArrayChunkId,
    id: ChunkId,
    owner: WorkerId,
    affinity: MemoryKind,
    last_readwrite: EventId,
    last_write: EventId,
    layout: ChunkLayout,
}

#[derive(Debug)]
struct ChunkAccess {
    array_id: ArrayId,
    chunk_idx: usize,
    task_id: EventId,
    is_write: bool,
}

#[derive(Debug)]
pub(crate) struct Planner {
    arrays: SlotMap<ArrayId, ArrayMeta>,
}

impl Planner {
    pub(crate) fn new() -> Self {
        Self {
            arrays: SlotMap::default(),
        }
    }

    pub(crate) fn submit_stage<F>(
        &mut self,
        driver: &DriverHandle,
        callback: F,
    ) -> Result<DriverEvent>
    where
        F: FnOnce(&mut PlannerStage) -> Result,
    {
        let future = driver.submit_stage(|plan| {
            let alloc = Bump::with_capacity(1024);
            let mut stage = PlannerStage::new(plan, self, &alloc);

            callback(&mut stage)?;

            stage.commit_reductions()?;
            stage.commit_cached_chunks();

            let mut chunks_accessed = stage.commit_accesses();
            self.commit_accessed_chunks(plan, &mut chunks_accessed);
            Ok(())
        })?;

        Ok(future)
    }

    fn commit_accessed_chunks(&mut self, plan: &mut Plan, chunks_accessed: &mut [ChunkAccess]) {
        for (array_id, accesses) in chunks_accessed.sort_and_group_by_key(|e| e.array_id) {
            let array = &mut self.arrays[array_id];

            for (chunk_idx, accesses) in accesses.sort_and_group_by_key(|e| e.chunk_idx) {
                let chunk = &mut array.chunks[chunk_idx];

                let mut list = SmallVec::with_capacity(accesses.len() + 1);
                list.extend(accesses.iter().map(|e| e.task_id));
                list.push(chunk.last_readwrite);
                chunk.last_readwrite = plan.join(chunk.owner, list);

                let n = accesses.iter().filter(|e| e.is_write).count();
                if n > 0 {
                    let mut list = SmallVec::with_capacity(n + 1);
                    list.extend(accesses.iter().filter(|e| e.is_write).map(|e| e.task_id));
                    list.push(chunk.last_write);
                    chunk.last_write = plan.join(chunk.owner, list);
                }
            }
        }
    }

    pub(crate) fn create_array(
        &mut self,
        driver: &DriverHandle,
        size: Dim,
        dtype: DataType,
        chunk_descriptions: &[ChunkDescriptor],
        distribution: Arc<dyn DataDistribution>,
        layout: ChunkLayoutBuilder,
    ) -> Result<ArrayId> {
        let array_id = self.arrays.insert_with_key(|id| ArrayMeta {
            id,
            size,
            dtype,
            distribution,
            layout: layout.clone(),
            chunks: Box::new([]),
        });

        let mut chunks = vec![];

        driver.submit_stage(|plan| {
            for (index, descr) in enumerate(chunk_descriptions) {
                let layout = layout.build(dtype, descr.size, descr.affinity);
                let (id, op) = plan.create_chunk(descr.owner, layout.clone());

                chunks.push(ChunkMeta {
                    key: ArrayChunkId(array_id, index),
                    id,
                    owner: descr.owner,
                    affinity: descr.affinity,
                    last_readwrite: op,
                    last_write: op,
                    layout,
                });
            }

            Ok(())
        })?;

        self.arrays[array_id].chunks = chunks.into_boxed_slice();
        Ok(array_id)
    }

    pub(crate) fn delete_array(
        &mut self,
        driver: &DriverHandle,
        id: ArrayId,
    ) -> Result<DriverEvent> {
        let array = self.arrays.remove(id).unwrap();
        let result = driver.submit_stage(|plan| {
            for chunk in &*array.chunks {
                let dep = plan.destroy_chunk(chunk.id, &[chunk.last_readwrite][..]);
                plan.add_terminal(dep);
            }

            Ok(())
        })?;

        Ok(result)
    }

    pub(crate) unsafe fn write_array(
        &mut self,
        driver: &DriverHandle,
        id: ArrayId,
        region: Rect,
        input: HostAccessor,
    ) -> Result<DriverEvent> {
        let array = self.array(id)?;
        let dtype = array.dtype;

        self.submit_stage(driver, |state| {
            let array = &state.planner.arrays[id];
            let chunks = &*array.chunks;
            let mut result = Ok(());

            array.distribution.visit_replicated(region, &mut |r| {
                let src_buffer = input.slice(Rect::new(r.region_offset, r.extents));
                let mut data = vec![0u8; src_buffer.size_in_bytes()];
                let dst_buffer = HostMutAccessor::from_buffer_raw(
                    data.as_mut_ptr(),
                    data.len(),
                    src_buffer.extents(),
                    dtype,
                );

                unsafe {
                    host_copy(SequentialPolicy, src_buffer, dst_buffer);
                }

                let chunk = &chunks[r.chunk_index];

                let ret = state.plan.add_tasklet(
                    chunk.owner,
                    ExecutorKind::Host,
                    &WriteDataTasklet {
                        dtype,
                        region: Rect::new(r.chunk_offset, r.extents),
                        data: ByteBuf::from(data),
                    },
                    [OperationChunk {
                        id: chunk.id,
                        exclusive: true,
                        dependency: Some(chunk.last_readwrite),
                    }],
                );

                match ret {
                    Ok(event_id) => {
                        state.add_access(chunk.key, event_id, true);
                        state.plan.add_terminal(event_id);
                    }
                    Err(err) => {
                        result = Err(err);
                    }
                }
            });

            result
        })
    }

    pub(crate) unsafe fn read_array(
        &mut self,
        driver: &DriverHandle,
        id: ArrayId,
        region: Rect,
        output: HostMutAccessor,
    ) -> Result {
        let mut futures = vec![];
        let array = self.array(id)?;
        let dtype = array.dtype;

        self.submit_stage(driver, |state| {
            let array = &state.planner.arrays[id];
            let chunks = &*array.chunks;
            let dtype = array.dtype;

            let mut result = Ok(());

            array.distribution.visit_unique(region, None, &mut |r| {
                let chunk = &chunks[r.chunk_index];

                let ret = state.plan.add_tasklet_with_reply(
                    chunk.owner,
                    ExecutorKind::Host,
                    &ReadDataTasklet {
                        region: Rect::new(r.chunk_offset, r.extents),
                        dtype,
                    },
                    [OperationChunk {
                        id: chunk.id,
                        exclusive: false,
                        dependency: Some(chunk.last_write),
                    }],
                );

                match ret {
                    Ok((event_id, future)) => {
                        state.add_access(chunk.key, event_id, false);
                        state.plan.add_terminal(event_id);
                        futures.push((future, r.region_offset, r.extents));
                    }
                    Err(err) => result = Err(err),
                }
            });

            result
        })?;

        for (future, region_offset, extents) in futures {
            let buf = future.wait()?;
            let buf = buf.as_ref();

            let src = HostAccessor::from_buffer_raw(buf.as_ptr(), buf.len(), extents, dtype);
            let dst = output.slice(Rect::new(region_offset, extents));

            host_copy(SequentialPolicy, src, dst);
        }

        Ok(())
    }

    pub(crate) fn shutdown(&mut self, driver: &DriverHandle) -> Result<DriverEvent> {
        driver.submit_stage(|plan| {
            for (_, array) in take(&mut self.arrays) {
                for chunk in &*array.chunks {
                    let dep = plan.destroy_chunk(chunk.id, &[chunk.last_readwrite][..]);
                    plan.add_terminal(dep);
                }
            }

            Ok(())
        })
    }

    fn array(&self, id: ArrayId) -> Result<&ArrayMeta> {
        self.arrays.get(id).ok_or_else(|| anyhow!("unknown array"))
    }

    fn array_mut(&mut self, id: ArrayId) -> Result<&mut ArrayMeta> {
        self.arrays
            .get_mut(id)
            .ok_or_else(|| anyhow!("unknown array"))
    }

    fn chunk(&self, key: ArrayChunkId) -> Result<&ChunkMeta> {
        self.array(key.0).map(|a| &a.chunks[key.1])
    }

    fn chunk_mut(&mut self, key: ArrayChunkId) -> Result<&mut ChunkMeta> {
        self.array_mut(key.0).map(|a| &mut a.chunks[key.1])
    }
}
