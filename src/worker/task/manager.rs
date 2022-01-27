use crate::network::{WorkerMsg, WorkerRpcSender};
use crate::prelude::*;
use crate::types::dag::{Operation, OperationKind};
use crate::types::{ChunkId, ChunkLayout, EventId, MemoryKind, SyncId, TaskletOutput};
use crate::worker::memory::{
    ChunkHandle, MemoryEvent, MemoryManager, RequestEvent, RequestHandle as MemoryRequest,
};
use crate::worker::task::executor_set::ExecutorSet;
use crate::worker::task::scheduler::Scheduler;
use crossbeam::channel::Sender;
use lightning_core::util::{TCell, TCellOwner};
use ptr_union::{Builder2, Enum2, Union2};
use smallvec::SmallVec;
use std::convert::Infallible;
use std::fmt::{self, Debug};
use std::mem::{self, ManuallyDrop};
use std::sync::Arc;

pub(crate) struct LockMarker;
pub(crate) type Lock = TCellOwner<LockMarker>;

#[derive(Debug)]
enum OperationResult {
    Void,
    Output(TaskletOutput),
    Err(anyhow::Error),
}

enum OperationSuccessor {
    Task(OperationHandle),
    Memory(MemoryRequest),
}

#[derive(Debug)]
pub(crate) struct Event {
    handle: OperationHandle,
    result: OperationResult,
}

#[derive(Debug)]
enum OperationStatus {
    Init,
    Dependencies(usize),
    Queued,
    Staging(usize, Result),
    Scheduled,
    Terminated,
}

pub(crate) struct OperationMeta {
    event_id: EventId,
    kind: Option<Box<OperationKind>>,
    needs_reply: bool,
    requests: SmallVec<[MemoryRequest; 8]>,
    state: TCell<LockMarker, OperationState>,
}

impl Debug for OperationMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskMeta")
            .field("event_id", &self.event_id)
            .field("kind", &self.kind)
            .field("needs_reply", &self.needs_reply)
            .field("requests", &self.requests)
            .field("state", &"...")
            .finish()
    }
}

#[derive(Debug)]
struct OperationState {
    status: OperationStatus,
    successors: SmallVec<[Union2<OperationHandle, MemoryRequest>; 8]>,
}

impl OperationMeta {
    fn new(event_id: EventId, kind: Option<Box<OperationKind>>) -> OperationHandle {
        let needs_reply = match kind.as_deref() {
            Some(OperationKind::Execute { needs_reply, .. }) => *needs_reply,
            _ => false,
        };

        Arc::new(OperationMeta {
            event_id,
            requests: default(),
            needs_reply,
            kind,
            state: TCell::new(OperationState {
                status: OperationStatus::Init,
                successors: default(),
            }),
        })
    }

    pub fn id(&self) -> EventId {
        self.event_id
    }

    pub fn requests(&self) -> &[MemoryRequest] {
        &self.requests
    }

    fn status<'a>(&'a self, token: &'a Lock) -> &'a OperationStatus {
        &self.state.borrow(token).status
    }

    fn set_status(&self, new_status: OperationStatus, token: &mut Lock) -> OperationStatus {
        trace!(
            "task {:?}: {:?} -> {:?}",
            self.event_id,
            self.status(token),
            new_status
        );

        replace(&mut self.state.borrow_mut(token).status, new_status)
    }

    fn add_successor(&self, successor: OperationSuccessor, token: &mut Lock) {
        let builder = unsafe { Builder2::new_unchecked() };
        let successor = match successor {
            OperationSuccessor::Task(handle) => builder.a(handle),
            OperationSuccessor::Memory(handle) => builder.b(handle),
        };

        self.state.borrow_mut(token).successors.push(successor);
    }

    fn take_successors(&self, token: &mut Lock) -> impl Iterator<Item = OperationSuccessor> {
        take(&mut self.state.borrow_mut(token).successors)
            .into_iter()
            .map(|e| match e.unpack() {
                Enum2::A(handle) => OperationSuccessor::Task(handle),
                Enum2::B(handle) => OperationSuccessor::Memory(handle),
            })
    }

    pub(crate) fn inner(&self) -> &OperationKind {
        self.kind.as_deref().unwrap_or(&OperationKind::Empty)
    }
}

pub(crate) type OperationHandle = Arc<OperationMeta>;

pub(crate) struct TaskManager {
    ops: HashMap<EventId, OperationHandle>,
    chunks: HashMap<ChunkId, ChunkHandle>,
    comm: WorkerRpcSender,
    memory: MemoryManager,
    scheduler: Box<dyn Scheduler>,
    executors: ExecutorSet,
    sender: Sender<Event>,
    token: Lock,
    shutdown_requested: bool,
    shutdown_finalized: bool,
}

impl TaskManager {
    pub(crate) fn new(
        comm: WorkerRpcSender,
        memory: MemoryManager,
        executors: ExecutorSet,
        scheduler: Box<dyn Scheduler>,
        sender: Sender<Event>,
    ) -> Result<Self> {
        Ok(Self {
            ops: default(),
            chunks: default(),
            comm,
            memory,
            sender,
            scheduler,
            executors,
            token: Lock::new(),
            shutdown_requested: false,
            shutdown_finalized: false,
        })
    }

    fn create_chunk(&mut self, id: ChunkId, layout: ChunkLayout) {
        let chunk = self.memory.create_chunk(layout);
        let old = self.chunks.insert(id, chunk);
        assert!(old.is_none());
    }

    fn delete_chunk(&mut self, id: ChunkId) {
        let chunk = self.chunks.remove(&id).expect("to find chunk");
        self.memory.delete_chunk(&chunk);
    }

    fn determine_place(&self, kind: &OperationKind) -> Option<MemoryKind> {
        use OperationKind::*;

        match kind {
            Execute { executor, .. } => Some(executor.best_affinity_memory()),
            Network(_) => Some(MemoryKind::Host),
            _ => None,
        }
    }

    pub(crate) fn submit_task(&mut self, op: Operation) {
        assert_eq!(self.shutdown_requested, false);

        let mut kind = op.kind;
        let chunks = &*op.chunks;
        let dependencies = &*op.dependencies;
        let event_id = op.event_id;

        if let Some(inner) = kind.as_mut() {
            match &**inner {
                &OperationKind::CreateChunk { id, ref layout } => {
                    self.create_chunk(id, layout.clone());
                    kind = None;
                }
                &OperationKind::DestroyChunk { id } => {
                    self.delete_chunk(id);
                    kind = None;
                }
                _ => {}
            }
        }

        let mut op = OperationMeta::new(event_id, kind);
        let mut requests = SmallVec::with_capacity(chunks.len());

        for access in chunks {
            let dep = match access.dependency {
                Some(dep) => self.ops.get(&dep),
                None => None,
            };

            let chunk = match self.chunks.get(&access.id) {
                Some(chunk) => chunk,
                None => panic!("unknown chunk: {:?}", access.id),
            };

            let request = self.memory.create_request(
                &chunk,
                OperationHandle::downgrade(&op),
                access.exclusive,
                dep.is_some() as usize,
            );

            if let Some(dep) = dep {
                let succ = OperationSuccessor::Memory(MemoryRequest::clone(&request));
                dep.add_successor(succ, &mut self.token);
            }

            requests.push(request);
        }

        // SAFETY: this is safe since this is the only Arc to the data. Replace this by
        // a call to Arc::new_cyclic once that stabilizes.
        unsafe { &mut *(Arc::as_ptr(&mut op) as *mut OperationMeta) }.requests = requests;
        let mut dependencies_pending = 0;

        for dep in dependencies {
            if let Some(pred) = self.ops.get(&dep) {
                let succ = OperationSuccessor::Task(OperationHandle::clone(&op));
                pred.add_successor(succ, &mut self.token);
                dependencies_pending += 1;
            }
        }

        op.set_status(
            OperationStatus::Dependencies(dependencies_pending),
            &mut self.token,
        );

        self.ops.insert(event_id, OperationHandle::clone(&op));
        self.try_queue(op);
        self.make_progress();
    }

    fn try_queue(&mut self, handle: OperationHandle) -> bool {
        if !matches!(handle.status(&self.token), OperationStatus::Dependencies(0)) {
            return false;
        }

        for request in &handle.requests {
            if !self.memory.may_submit_request(&request) {
                return false;
            }
        }

        handle.set_status(OperationStatus::Queued, &mut self.token);

        if matches!(handle.inner(), OperationKind::Empty) {
            self.stage_task(handle);
        } else {
            self.scheduler.enqueue(handle, &self.token);
        }

        return true;
    }

    fn try_stage_task(&mut self, handle: OperationHandle) -> bool {
        assert!(matches!(
            handle.status(&self.token),
            OperationStatus::Queued
        ));

        for request in &handle.requests {
            if !self.memory.may_submit_request(&request) {
                handle.set_status(OperationStatus::Dependencies(0), &mut self.token);
                return false;
            }
        }

        self.stage_task(handle);
        return true;
    }

    fn stage_task(&mut self, handle: OperationHandle) {
        assert!(matches!(
            handle.status(&self.token),
            OperationStatus::Queued
        ));
        trace!("staging: {:#?}", handle);

        let place = self.determine_place(&handle.inner());

        for child in &handle.requests {
            self.memory.submit_request(child, place);
        }

        let n = handle.requests.len();
        handle.set_status(OperationStatus::Staging(n, Ok(())), &mut self.token);
        self.scheduler.on_start(&handle, &self.token);

        self.try_schedule(handle);
    }

    fn try_schedule(&mut self, handle: OperationHandle) -> bool {
        let (n, result) = match handle.set_status(OperationStatus::Scheduled, &mut self.token) {
            OperationStatus::Staging(n, result) => (n, result),
            other => panic!("invalid status: {:?}", other),
        };

        // If n > 0, this task cannot be scheduled yet.
        if n > 0 {
            handle.set_status(OperationStatus::Staging(n, result), &mut self.token);
            return false;
        }

        if let Err(e) = result {
            self.complete_task(handle, OperationResult::Err(e));
            return true;
        }

        let mut buffers = vec![];
        for request in &handle.requests {
            buffers.push(request.get(&self.memory));
        }

        match handle.inner() {
            &OperationKind::Empty => {
                self.complete_task(handle, OperationResult::Void);
                return true;
            }
            &OperationKind::Sync { id: barrier_id } => {
                // TODO: maybe this should move?
                self.send_barrier(handle.event_id, barrier_id);
                self.complete_task(handle, OperationResult::Void);
                return true;
            }
            other => {
                let completion =
                    Completion::new(OperationHandle::clone(&handle), self.sender.clone());
                debug!("execute: {:?} {:#?} {:#?}", handle.event_id, other, buffers);

                unsafe {
                    self.executors.execute(other, buffers, completion);
                }
            }
        };

        return true;
    }

    fn complete_task(&mut self, handle: OperationHandle, result: OperationResult) {
        assert!(matches!(
            handle.status(&self.token),
            OperationStatus::Scheduled
        ));
        self.scheduler.on_complete(&handle, &self.token);

        handle.set_status(OperationStatus::Terminated, &mut self.token);
        let op = self.ops.remove(&handle.event_id).unwrap();

        for request in &op.requests {
            unsafe {
                self.memory.finish_request(&request);
            }
        }

        if let OperationResult::Err(e) = &result {
            warn!("task {:?} failed: {}", handle, e);
        }

        if op.needs_reply || matches!(result, OperationResult::Err(_)) {
            let event_id = handle.event_id;

            let result = match result {
                OperationResult::Output(v) => Ok(v),
                OperationResult::Void => Err(anyhow!("task completed without output")),
                OperationResult::Err(e) => Err(e),
            };

            self.send_completion(event_id, result);
        }

        for succ in op.take_successors(&mut self.token) {
            match succ {
                OperationSuccessor::Task(handle) => {
                    let n = match handle.status(&self.token) {
                        OperationStatus::Dependencies(n) => n - 1,
                        s => panic!("unexpected status: {:?}", s),
                    };

                    handle.set_status(OperationStatus::Dependencies(n), &mut self.token);

                    if n == 0 {
                        self.try_queue(handle);
                    }
                }
                OperationSuccessor::Memory(request) => {
                    self.memory.satisfy_dependency(&request);
                    let parent = request
                        .parent()
                        .upgrade()
                        .unwrap()
                        .downcast::<OperationMeta>()
                        .unwrap();

                    self.try_queue(parent);
                }
            }
        }
    }

    fn make_progress(&mut self) {
        loop {
            if let Some((request, event)) = self.memory.poll() {
                let op = request
                    .parent()
                    .upgrade()
                    .unwrap()
                    .downcast::<OperationMeta>()
                    .unwrap();

                match event {
                    RequestEvent::Ready => {
                        self.try_queue(op);
                    }
                    RequestEvent::Active => {
                        match op.set_status(OperationStatus::Init, &mut self.token) {
                            OperationStatus::Staging(n, result) => {
                                op.set_status(
                                    OperationStatus::Staging(n - 1, result),
                                    &mut self.token,
                                );
                            }
                            other => panic!("invalid status: {:?}", other),
                        };

                        self.try_schedule(op);
                    }
                    RequestEvent::Abort(err) => {
                        match op.status(&self.token) {
                            OperationStatus::Staging(n, _) => {
                                op.set_status(
                                    OperationStatus::Staging(n - 1, Err(err)),
                                    &mut self.token,
                                );
                            }
                            other => panic!("invalid status: {:?}", other),
                        };

                        self.try_schedule(op);
                    }
                }
            } else if let Some(task) = self.scheduler.dequeue(&self.token) {
                self.try_stage_task(task);
            } else {
                break;
            }
        }

        trace!(
            "{:?} tasks remaining",
            (
                self.shutdown_requested,
                self.shutdown_finalized,
                self.ops.len(),
                self.chunks.len(),
            )
        );

        // TODO: more checks for shutdown
        if self.shutdown_requested && !self.shutdown_finalized && self.ops.is_empty() {
            if self.memory.is_idle() {
                self.shutdown_finalized = true;
            }
        }
    }

    fn send_barrier(&mut self, event_id: EventId, barrier_id: SyncId) {
        let msg = WorkerMsg::Sync(barrier_id, event_id);
        self.comm.message_driver(msg).unwrap();
    }

    fn send_completion(&mut self, event_id: EventId, result: Result<TaskletOutput, anyhow::Error>) {
        let result = result.map_err(|e| e.into());
        let msg = WorkerMsg::Complete(event_id, result);

        self.comm.message_driver(msg).unwrap();
    }

    pub(crate) fn handle_event(&mut self, event: Event) {
        debug!("handle_event: {:?}", event);
        self.complete_task(event.handle, event.result);
        self.make_progress();
    }

    pub(crate) fn handle_memory_event(&mut self, event: MemoryEvent) {
        debug!("handle_memory_event: {:?}", event);
        self.memory.handle_event(event);
        self.make_progress();
    }

    pub(crate) fn request_shutdown(&mut self) {
        self.shutdown_requested = true;

        for (_, chunk) in take(&mut self.chunks) {
            self.memory.delete_chunk(&chunk);
        }

        self.make_progress();
    }

    pub(crate) fn has_shutdown(&mut self) -> bool {
        self.shutdown_finalized
    }
}

#[derive(Debug)]
pub(crate) struct Completion {
    handle: ManuallyDrop<OperationHandle>,
    sender: ManuallyDrop<Sender<Event>>,
}

impl Completion {
    fn new(handle: OperationHandle, sender: Sender<Event>) -> Self {
        Self {
            handle: ManuallyDrop::new(handle),
            sender: ManuallyDrop::new(sender),
        }
    }

    pub(crate) fn complete_ok(self) {
        self.complete::<Infallible>(Ok(()))
    }

    pub(crate) fn complete<E>(self, result: Result<(), E>)
    where
        E: Into<anyhow::Error>,
    {
        self._complete(match result {
            Ok(()) => OperationResult::Void,
            Err(e) => OperationResult::Err(e.into()),
        })
    }

    pub(crate) fn complete_tasklet(self, result: Result<TaskletOutput>) {
        self._complete(match result {
            Ok(v) => OperationResult::Output(v),
            Err(e) => OperationResult::Err(e),
        })
    }

    fn _complete(mut self, result: OperationResult) {
        let handle = unsafe { ManuallyDrop::take(&mut self.handle) };
        let sender = unsafe { ManuallyDrop::take(&mut self.sender) };
        mem::forget(self); // Don't call drop

        let event = Event { handle, result };

        if let Err(_) = sender.send(event) {
            error!("could not call completion since channel was closed");
        }
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        let handle = unsafe { ManuallyDrop::take(&mut self.handle) };
        let sender = unsafe { ManuallyDrop::take(&mut self.sender) };

        let result = OperationResult::Err(anyhow!("task could not be completed"));
        let event = Event { handle, result };

        if let Err(_) = sender.send(event) {
            error!("could complete task since channel was closed");
        }
    }
}
