use crate::network::Tag;
use crate::prelude::*;
use crate::types::dag::{EventList, NetworkOperation, Operation, OperationChunk, OperationKind};
use crate::types::{
    package_tasklet, package_tasklet_with_callback, Affine, ChunkId, ChunkLayout, Dim, EventId,
    ExecutorKind, SyncId, Tasklet, TaskletCallback, WorkerId,
};
use lightning_core::util::{Counter, Future, GroupByExt, Promise};
use smallvec::{smallvec, SmallVec};
use std::num::NonZeroU64;

pub struct Plan {
    pub(crate) next_id: NonZeroU64,
    pub(super) ops: Box<[Vec<Operation>]>,
    pub(super) pending_replies: Vec<(EventId, TaskletCallback)>,
    pub(super) next_chunk: NonZeroU64,
    pub(super) next_tag: Tag,
    terminals: Vec<EventId>,
    max_tag: Tag,
}

impl Plan {
    pub(super) fn new(
        next_id: NonZeroU64,
        num_nodes: usize,
        next_chunk: NonZeroU64,
        next_tag: Tag,
        max_tag: Tag,
    ) -> Self {
        let mut ops = Vec::with_capacity(num_nodes);
        ops.resize(num_nodes, default());
        let ops = ops.into_boxed_slice();

        Self {
            next_id,
            ops,
            pending_replies: Vec::new(),
            terminals: Vec::new(),
            next_chunk,
            next_tag,
            max_tag,
        }
    }

    fn _add(
        &mut self,
        node: WorkerId,
        kind: Option<Box<OperationKind>>,
        chunks: Box<[OperationChunk]>,
        mut dependencies: EventList,
    ) -> EventId {
        let id = EventId::new(node, self.next_id.get_and_increment());
        let n = dependencies.len();

        if n > 1 {
            dependencies.sort();
            dependencies.dedup();
        }

        self.ops[node.get()].push(Operation {
            event_id: id,
            kind,
            chunks,
            dependencies,
        });

        id
    }

    #[inline(always)]
    pub(super) fn add_with_deps(
        &mut self,
        node: WorkerId,
        kind: Option<OperationKind>,
        dependencies: EventList,
    ) -> EventId {
        self._add(node, kind.map(|k| Box::new(k)), Box::new([]), dependencies)
    }

    pub(super) fn add(
        &mut self,
        node: WorkerId,
        kind: OperationKind,
        chunks: impl Into<Box<[OperationChunk]>>,
    ) -> EventId {
        self._add(node, Some(Box::new(kind)), chunks.into(), smallvec![])
    }

    pub(crate) fn add_terminal(&mut self, dep: EventId) -> EventId {
        self.terminals.push(dep);
        dep
    }

    pub(crate) fn commit_terminals(&mut self, sync_id: SyncId) -> Vec<EventId> {
        let mut event_ids = vec![];

        for (owner, deps) in take(&mut self.terminals).sort_and_group_by_key(|e| e.owner()) {
            let event_id = self.add_with_deps(
                owner,
                Some(OperationKind::Sync { id: sync_id }),
                EventList::from(&*deps),
            );

            event_ids.push(event_id);
        }

        event_ids
    }

    pub(crate) fn create_chunk(
        &mut self,
        node: WorkerId,
        layout: ChunkLayout,
    ) -> (ChunkId, EventId) {
        let id = ChunkId::new(node, self.next_chunk.get_and_increment());
        let op = self.add(node, OperationKind::CreateChunk { id, layout }, []);

        (id, op)
    }

    pub(crate) fn destroy_chunk(&mut self, id: ChunkId, deps: impl Into<EventList>) -> EventId {
        self.add_with_deps(
            id.owner(),
            Some(OperationKind::DestroyChunk { id }),
            deps.into(),
        )
    }

    pub(crate) fn add_copy(
        &mut self,
        src_id: ChunkId,
        src_transform: Affine,
        src_dep: EventId,
        dst_id: ChunkId,
        dst_transform: Affine,
        dst_dep: EventId,
        extents: Dim,
    ) -> EventId {
        let node = src_id.owner();
        assert_eq!(dst_id.owner(), node);

        let kind = OperationKind::CopyData {
            src_transform,
            dst_transform,
            domain: extents,
        };

        if src_id != dst_id {
            self.add(
                node,
                kind,
                [
                    OperationChunk {
                        id: src_id,
                        exclusive: false,
                        dependency: Some(src_dep),
                    },
                    OperationChunk {
                        id: dst_id,
                        exclusive: true,
                        dependency: Some(dst_dep),
                    },
                ],
            )
        } else {
            let dependency = self.join(node, [src_dep, dst_dep]);

            self.add(
                node,
                kind,
                [OperationChunk {
                    id: dst_id,
                    exclusive: true,
                    dependency: Some(dependency),
                }],
            )
        }
    }

    pub(crate) fn add_tasklet_with_reply<T: Tasklet>(
        &mut self,
        node_id: WorkerId,
        executor: ExecutorKind,
        task: &T,
        chunks: impl Into<Box<[OperationChunk]>>,
    ) -> Result<(EventId, Future<Result<T::Output>>)> {
        let (promise, future) = Promise::new();
        let (task, callback) = package_tasklet_with_callback(task, |e| promise.complete(e))?;
        let op = self.add(
            node_id,
            OperationKind::Execute {
                executor,
                tasklet: task,
                needs_reply: true,
            },
            chunks,
        );

        self.pending_replies.push((op, callback));
        Ok((op, future))
    }

    pub(crate) fn add_tasklet<T: Tasklet>(
        &mut self,
        node_id: WorkerId,
        executor: ExecutorKind,
        task: &T,
        chunks: impl Into<Box<[OperationChunk]>>,
    ) -> Result<EventId> {
        let task = package_tasklet(task)?;
        let op = self.add(
            node_id,
            OperationKind::Execute {
                executor,
                tasklet: task,
                needs_reply: false,
            },
            chunks,
        );

        Ok(op)
    }

    pub(crate) fn add_sendrecv(
        &mut self,
        src_id: ChunkId,
        src_dep: EventId,
        dst_id: ChunkId,
        dst_dep: EventId,
    ) -> (EventId, EventId) {
        let src_node = src_id.owner();
        let dst_node = dst_id.owner();

        let tag = self.next_tag;
        if tag.0 + 1 < self.max_tag.0 {
            self.next_tag = Tag(tag.0 + 1);
        } else {
            self.next_tag = Tag(0);
        }

        let send_op = self.add(
            src_node,
            OperationKind::Network(NetworkOperation::SendData {
                receiver: dst_node,
                tag,
            }),
            [OperationChunk {
                id: src_id,
                exclusive: false,
                dependency: Some(src_dep),
            }],
        );

        let probe_op = self.add_with_deps(
            dst_node,
            Some(OperationKind::Network(NetworkOperation::RecvProbe {
                sender: src_node,
                tag,
            })),
            EventList::from(&[dst_dep][..]),
        );

        let recv_op = self.add(
            dst_node,
            OperationKind::Network(NetworkOperation::RecvData {
                sender: src_node,
                tag,
            }),
            [OperationChunk {
                id: dst_id,
                exclusive: true,
                dependency: Some(probe_op),
            }],
        );

        (send_op, recv_op)
    }

    pub(crate) fn join<D: Into<EventList> + AsRef<[EventId]>>(
        &mut self,
        node: WorkerId,
        dependencies: D,
    ) -> EventId {
        let deps = dependencies.as_ref();

        if deps.len() == 1 || (deps.len() == 2 && deps[0] == deps[1]) {
            deps[0]
        } else {
            self.add_with_deps(node, None, dependencies.into())
        }
    }

    pub(crate) fn join_with(
        &mut self,
        node: WorkerId,
        main: EventId,
        dependencies: &[EventId],
    ) -> EventId {
        if dependencies.is_empty() || (dependencies.len() == 1 && dependencies[0] == main) {
            main
        } else if dependencies.len() == 1 {
            self.add_with_deps(node, None, SmallVec::from_buf([main, dependencies[0]]))
        } else {
            let mut result = Vec::with_capacity(dependencies.len() + 1);
            result.extend(dependencies);
            result.push(main);
            self.add_with_deps(node, None, result.into())
        }
    }
}
