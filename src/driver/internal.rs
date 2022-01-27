use crossbeam::channel::Select;
use lightning_codegen::ModuleDef as CudaModuleDef;
use lightning_core::util::{Counter, Promise};
use std::fmt::{self, Debug};
use std::num::NonZeroU64;
use std::sync::{Arc, Weak};
use std::thread::{self, JoinHandle};

use super::plan::Plan;
use crate::driver::trace::PlanTrace;
use crate::network::{
    DriverMsg, DriverRpcReceiver, DriverRpcSender, SerializedError, Tag, WorkerMsg,
};
use crate::prelude::*;
use crate::types::{
    CudaKernelId, DriverConfig, EventId, SyncId, SystemInfo, TaskletCallback, TaskletOutput,
    WorkerId, WorkerInfo,
};

#[derive(Debug)]
struct KernelCompilation {
    workers_acknowledged: Vec<WorkerId>,
    waiter: Promise<Result>,
    definition: CudaModuleDef,
}

#[derive(Clone, Debug)]
pub struct Event {
    inner: Arc<Mutex<EventState>>,
}

enum EventState {
    Pending {
        callbacks: Vec<Box<dyn FnOnce(Result) + Send>>,
    },
    Ready {
        result: Result<(), SerializedError>,
    },
}

impl Debug for EventState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventState::Pending { callbacks } => f
                .debug_struct("Pending")
                .field("callbacks", &callbacks.len())
                .finish(),
            EventState::Ready { result } => {
                f.debug_struct("Ready").field("result", &result).finish()
            }
        }
    }
}

impl Event {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(EventState::Pending { callbacks: vec![] })),
        }
    }

    fn ready() -> Self {
        Self {
            inner: Arc::new(Mutex::new(EventState::Ready { result: Ok(()) })),
        }
    }

    fn trigger(&self, result: Result<(), SerializedError>) {
        let mut guard = self.inner.lock();
        match replace(
            &mut *guard,
            EventState::Ready {
                result: result.clone(),
            },
        ) {
            EventState::Pending { callbacks } => {
                for callback in callbacks {
                    match &result {
                        Ok(()) => (callback)(Ok(())),
                        Err(e) => (callback)(Err(e.clone().into())),
                    }
                }
            }
            EventState::Ready { .. } => {
                panic!("event was already triggered");
            }
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(&*self.inner.lock(), EventState::Ready { .. })
    }

    pub fn attach_callback<F>(&self, fun: F)
    where
        F: FnOnce(Result) + Send + 'static,
    {
        let mut guard = self.inner.lock();
        match &mut *guard {
            EventState::Pending { callbacks } => callbacks.push(Box::new(fun)),
            EventState::Ready { result } => match result {
                Ok(()) => (fun)(Ok(())),
                Err(e) => (fun)(Err(e.clone().into())),
            },
        }
    }
}

#[derive(Debug)]
struct SyncMeta {
    events_pending: Vec<EventId>,
    result: Result<(), SerializedError>,
    trigger: Event,
}

impl SyncMeta {
    fn new(dependencies: Vec<EventId>) -> (Self, Event) {
        let event = Event::new();

        let this = Self {
            events_pending: dependencies,
            result: Ok(()),
            trigger: event.clone(),
        };

        (this, event)
    }
}

#[derive(Debug)]
pub(crate) struct DriverState {
    sync_counter: NonZeroU64,
    chunk_counter: NonZeroU64,
    op_counter: NonZeroU64,
    next_tag: Tag,
    next_kernel_id: u64,
    comm: DriverRpcSender,
    pending_compilation: HashMap<CudaKernelId, KernelCompilation>,
    pending_replies: HashMap<EventId, TaskletCallback>,
    barriers: HashMap<SyncId, SyncMeta>,
    workers: Vec<WorkerInfo>,
    shutdown_requested: bool,
    shutdown_acknowledged: Vec<WorkerId>,
    trace_file: Option<PlanTrace>,
}

impl DriverState {
    fn new(config: &DriverConfig, comm: DriverRpcSender, workers: Vec<WorkerInfo>) -> Self {
        const ONE: NonZeroU64 = unsafe { NonZeroU64::new_unchecked(1) };

        let trace_file = match &config.trace_file {
            Some(path) => match PlanTrace::new(path) {
                Ok(f) => Some(f),
                Err(e) => {
                    warn!("failed to open {:?} for writing: {}", path, e);
                    None
                }
            },
            None => None,
        };

        Self {
            sync_counter: ONE,
            chunk_counter: ONE,
            op_counter: ONE,
            next_kernel_id: 1,
            next_tag: Tag(1),
            comm,
            pending_compilation: default(),
            barriers: default(),
            pending_replies: default(),
            workers,
            shutdown_requested: false,
            shutdown_acknowledged: vec![],
            trace_file,
        }
    }

    fn submit_stage(&mut self, fun: Box<dyn FnOnce(&mut Plan) -> Result + '_>) -> Result<Event> {
        let sync_id = SyncId(self.sync_counter.get_and_increment());
        let mut plan = Plan::new(
            self.op_counter,
            self.comm.num_workers(),
            self.chunk_counter,
            self.next_tag,
            self.comm.max_tag(),
        );

        (fun)(&mut plan)?;

        let terminals = plan.commit_terminals(sync_id);
        let event = if !terminals.is_empty() {
            let (meta, future) = SyncMeta::new(terminals);
            self.barriers.insert(sync_id, meta);
            future
        } else {
            Event::ready()
        };

        if let Some(p) = &mut self.trace_file {
            p.add(&plan.ops);
        }

        self.chunk_counter = plan.next_chunk;
        self.next_tag = plan.next_tag;
        self.pending_replies.extend(plan.pending_replies.drain(..));
        self.op_counter = plan.next_id;

        for (node_id, ops) in enumerate(plan.ops.into_vec()) {
            if !ops.is_empty() {
                self.comm
                    .message_worker(WorkerId::new(node_id), DriverMsg::Submit(ops))?;
            }
        }

        Ok(event)
    }

    fn compile_cuda_kernel(
        &mut self,
        definition: CudaModuleDef,
        waiter: Promise<Result>,
    ) -> Result<CudaKernelId> {
        let id = CudaKernelId(self.next_kernel_id.get_and_increment());

        let msg = DriverMsg::Compile(id, definition.clone());
        self.comm.message_all(msg)?;

        self.pending_compilation.insert(
            id,
            KernelCompilation {
                workers_acknowledged: vec![],
                waiter,
                definition,
            },
        );

        Ok(id)
    }

    fn handle_message(&mut self, source: WorkerId, msg: WorkerMsg) {
        use WorkerMsg::*;
        trace!("handling message: {:?}", msg);

        match msg {
            Sync(barrier_id, event_id) => {
                self.handle_barrier_message(barrier_id, event_id);
            }
            Complete(id, result) => {
                self.handle_completion_message(source, id, result);
            }
            CompileResult(kernel_id, result) => {
                self.handle_compilation_result(source, kernel_id, result.map_err(Into::into));
            }
            AcknowledgeShutdown => {
                self.handle_shutdown_ack(source);
            }
            m => {
                warn!("ignoring invalid message: {:?}", m);
            }
        }
    }

    fn handle_completion_message(
        &mut self,
        source: WorkerId,
        event_id: EventId,
        result: Result<TaskletOutput, SerializedError>,
    ) {
        if let Err(msg) = &result {
            error!("error occurred on worker {}: {:?}", source, msg);

            for meta in &mut self.barriers.values_mut() {
                meta.result = Err(msg.clone());
            }
        }

        if let Some(callback) = self.pending_replies.remove(&event_id) {
            callback.process(result.map_err(Into::into));
        }
    }

    fn handle_barrier_message(&mut self, barrier_id: SyncId, event_id: EventId) {
        use std::collections::hash_map::Entry;

        let mut entry = match self.barriers.entry(barrier_id) {
            Entry::Occupied(e) => e,
            Entry::Vacant(_) => {
                warn!("unknown barrier: {:?}", barrier_id);
                return;
            }
        };

        let pending = &mut entry.get_mut().events_pending;
        pending.retain(|&e| e != event_id);

        if pending.is_empty() {
            let entry = entry.remove();
            entry.trigger.trigger(entry.result);
        }
    }

    fn handle_compilation_result(
        &mut self,
        source: WorkerId,
        kernel_id: CudaKernelId,
        result: Result,
    ) {
        let mut comp = match self.pending_compilation.remove(&kernel_id) {
            Some(c) => c,
            _ => return, // Compilation already failed on other node
        };

        comp.workers_acknowledged.push(source);

        if let Err(e) = result {
            comp.waiter.complete(Err(e));
        } else if comp.workers_acknowledged.len() < self.workers.len() {
            self.pending_compilation.insert(kernel_id, comp);
        } else {
            comp.waiter.complete(Ok(()));

            if let Some(trace) = &mut self.trace_file {
                trace.kernel_compiled(kernel_id, comp.definition);
            }
        }
    }

    fn handle_shutdown_ack(&mut self, source: WorkerId) {
        self.shutdown_acknowledged.push(source);
    }

    fn request_shutdown(&mut self) -> Result {
        if self.shutdown_requested {
            return Ok(()); // Already shutting down
        }

        self.comm.message_all(DriverMsg::Shutdown)?;
        self.shutdown_requested = true;
        Ok(())
    }

    fn has_shutdown(&self) -> bool {
        self.shutdown_acknowledged.len() == self.workers.len()
    }
}

#[derive(Debug)]
pub(crate) struct Driver {
    handle: Handle,
    thread_handle: JoinHandle<()>,
}

impl Driver {
    pub(crate) fn handle(&self) -> Handle {
        self.handle.clone()
    }

    pub(crate) fn shutdown_and_wait(self) -> Result {
        if let Some(driver) = self.handle.driver.upgrade() {
            driver.lock().request_shutdown()?;
        }

        self.thread_handle.join().unwrap();
        Ok(())
    }
}

#[derive(Debug)]
struct Inner {
    state: Mutex<DriverState>,
}

#[derive(Debug, Clone)]
pub struct Handle {
    driver: Weak<Mutex<DriverState>>,
    system: Arc<SystemInfo>,
}

impl Handle {
    pub fn compile_kernel(&self, kernel_def: CudaModuleDef) -> Result<CudaKernelId> {
        let (promise, future) = Promise::new();
        let kernel_id = self
            .driver
            .upgrade()
            .ok_or_else(|| anyhow!("driver has shut down"))?
            .lock()
            .compile_cuda_kernel(kernel_def, promise)?;

        future.wait()?;
        Ok(kernel_id)
    }

    pub fn system(&self) -> &SystemInfo {
        &self.system
    }

    pub fn submit_stage<F: FnOnce(&mut Plan) -> Result>(&self, fun: F) -> Result<Event> {
        self.driver
            .upgrade()
            .ok_or_else(|| anyhow!("driver has shut down"))?
            .lock()
            .submit_stage(Box::new(fun))
    }

    pub fn synchronize(&self) -> Result {
        let mut futures = vec![];

        {
            let driver = self
                .driver
                .upgrade()
                .ok_or_else(|| anyhow!("driver has shut down"))?;

            let mut guard = driver.lock();
            for (_, meta) in &mut guard.barriers {
                let (promise, future) = Promise::new();
                meta.trigger.attach_callback(|e| promise.complete(e));
                futures.push(future);
            }
        }

        for future in futures {
            future.wait()?;
        }

        Ok(())
    }
}

fn main_loop(comm: DriverRpcReceiver, state: &Mutex<DriverState>) {
    let mut select = Select::new();
    comm.register(&mut select);

    loop {
        select.ready();

        let mut guard = state.lock();
        while let Some((source, msg)) = comm.poll().unwrap() {
            guard.handle_message(source, msg);

            if guard.has_shutdown() {
                return;
            }
        }
    }
}

pub(crate) fn launch_driver_thread(
    config: DriverConfig,
    sender: DriverRpcSender,
    receiver: DriverRpcReceiver,
) -> Result<Driver> {
    let mut select = Select::new();
    receiver.register(&mut select);

    let mut workers = vec![];
    while workers.len() < sender.num_workers() {
        select.ready();

        if let Some((source, msg)) = receiver.poll()? {
            match msg {
                WorkerMsg::Initialize(Ok(info)) => {
                    workers.push(info);
                }
                WorkerMsg::Initialize(Err(msg)) => {
                    panic!("initialization failed: {:?}: {:?}", source, msg);
                }
                m => {
                    warn!("receive invalid message: {:?}", m);
                }
            }
        }
    }

    workers.sort_by_key(|v| v.node_id);

    let to_gb = 1.0 / 1024.0 / 1024.0 / 1024.0;
    info!("workers initialized:");
    for worker in &workers {
        info!(
            " - node {}: {}, {:.3} GB",
            worker.node_id,
            worker.hostname,
            worker.memory_capacity as f64 * to_gb
        );

        for device in &worker.devices {
            info!(
                " - - {}, {:.3} GB",
                device.capabilities.name,
                device.capabilities.memory_capacity as f64 * to_gb
            );
        }
    }

    let state = DriverState::new(&config, sender, workers.clone());
    let driver = Arc::new(Mutex::new(state));
    let driver_handle = Arc::downgrade(&driver);

    let thread_handle = thread::Builder::new()
        .name("driver".to_string())
        .spawn(move || main_loop(receiver, &driver))
        .unwrap();

    Ok(Driver {
        handle: Handle {
            driver: driver_handle,
            system: Arc::new(SystemInfo::new(workers)),
        },
        thread_handle,
    })
}
