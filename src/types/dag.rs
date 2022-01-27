use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::network::Tag;
use crate::types::{
    Affine, ChunkId, ChunkLayout, Dim, EventId, ExecutorKind, SyncId, TaskletInstance, WorkerId,
};

pub(crate) type EventList = SmallVec<[EventId; 2]>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Operation {
    pub(crate) kind: Option<Box<OperationKind>>, // kind of task. None indicates a dummy node
    pub(crate) chunks: Box<[OperationChunk]>,    // chunks required for this task to execute
    pub(crate) dependencies: EventList, // events that must complete before this task can run
    pub(crate) event_id: EventId,       // event that this task will trigger
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct OperationChunk {
    pub(crate) id: ChunkId,
    pub(crate) exclusive: bool,
    pub(crate) dependency: Option<EventId>, // event that must complete before chunk is available.
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) enum OperationKind {
    Empty,
    CreateChunk {
        id: ChunkId,
        layout: ChunkLayout,
    },
    DestroyChunk {
        id: ChunkId,
    },
    Sync {
        id: SyncId,
    },
    CopyData {
        src_transform: Affine,
        dst_transform: Affine,
        domain: Dim,
    },
    Execute {
        executor: ExecutorKind,
        tasklet: TaskletInstance,
        needs_reply: bool,
    },
    Network(NetworkOperation),
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub(crate) enum NetworkOperation {
    SendData { receiver: WorkerId, tag: Tag },
    RecvProbe { sender: WorkerId, tag: Tag },
    RecvData { sender: WorkerId, tag: Tag },
}
