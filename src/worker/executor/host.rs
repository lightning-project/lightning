use crate::prelude::*;
use crate::types::{
    DataValue, Executor, ExecutorId, ExecutorKind, GenericAccessor, HostAccessor, HostMutAccessor,
    Reduction, TaskletInstance, WorkerId,
};
use crate::worker::task::Completion;
use lightning_memops::RayonPolicy;
use rayon::prelude::ParallelIterator;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::fmt::{self, Display};
use std::marker::PhantomData;

#[derive(Debug)]
pub(crate) struct HostThreadPool {
    node_id: WorkerId,
    pool: ThreadPool,
}

impl HostThreadPool {
    pub(crate) fn new(node_id: WorkerId) -> Self {
        let pool = ThreadPoolBuilder::new()
            .thread_name(|i| format!("host-executor-{}", i))
            .num_threads(4)
            .build()
            .unwrap();

        Self { node_id, pool }
    }

    pub(crate) unsafe fn submit_tasklet(
        &self,
        task: TaskletInstance,
        arrays: Vec<GenericAccessor>,
        _priority: u64, // Unused for now
        completion: Completion,
    ) {
        let node_id = self.node_id;
        self.pool.spawn_fifo(move || {
            let result = task.execute(&arrays, &HostExecutor::new(node_id));
            completion.complete_tasklet(result);
        });
    }

    pub(crate) unsafe fn submit_copy(
        &self,
        src_buffer: HostAccessor,
        dst_buffer: HostMutAccessor,
        completion: Completion,
    ) {
        self.pool.spawn_fifo(move || {
            lightning_memops::host_copy(RayonPolicy, src_buffer, dst_buffer);
            completion.complete_ok();
        });
    }
}

pub struct HostExecutor {
    node_id: WorkerId,
    no_send: PhantomData<*mut u8>,
}

impl Display for HostExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.id())
    }
}

impl HostExecutor {
    fn new(node_id: WorkerId) -> Self {
        HostExecutor {
            node_id,
            no_send: PhantomData,
        }
    }

    pub fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce(&Self) -> RA + Send,
        B: FnOnce(&Self) -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let node_id = self.node_id;
        rayon::join(
            move || (left)(&Self::new(node_id)),
            move || (right)(&Self::new(node_id)),
        )
    }

    pub fn map<I, F>(&self, iter: I, fun: F)
    where
        I: ParallelIterator,
        F: Fn(&Self, I::Item) + Sync + Send,
    {
        let node_id = self.node_id;
        iter.for_each(move |val| (fun)(&Self::new(node_id), val))
    }

    pub unsafe fn fill(&self, buffer: HostMutAccessor, value: DataValue) {
        lightning_memops::host_fill(RayonPolicy, buffer, value);
    }

    pub unsafe fn fold(
        &self,
        src: HostAccessor,
        dst: HostMutAccessor,
        reduction: Reduction,
    ) -> Result {
        lightning_memops::host_fold(RayonPolicy, src, dst, reduction)
    }

    pub unsafe fn copy(&self, src: HostAccessor, dst: HostMutAccessor) -> Result {
        lightning_memops::host_copy(RayonPolicy, src, dst);
        Ok(())
    }
}

impl Executor for HostExecutor {
    fn id(&self) -> ExecutorId {
        ExecutorId::new(self.node_id, ExecutorKind::Host)
    }
}
