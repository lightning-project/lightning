use self::executor::{CudaExecutorThread, HostThreadPool};
use self::memory::Storage;
use crate::network::{DriverMsg, WorkerEndpoint, WorkerMsg, WorkerRpcReceiver, WorkerRpcSender};
use crate::prelude::*;
use crate::types::{
    CudaKernelId, DeviceId, DeviceInfo, ExecutorId, ExecutorKind, MemoryId, MemoryKind,
    WorkerConfig, WorkerInfo,
};
use crate::worker::memory::{CopyEngine, MemoryEvent, MemoryManager};
use crate::worker::task::{ExecutorSet, GlobalScheduler};
use crossbeam::channel::{unbounded, Receiver, Select, Sender};
use cuda_driver_sys::cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY;
use lightning_codegen::ModuleDef as CudaModuleDef;
use lightning_cuda::prelude::*;
use memory::{DeviceMemoryPool, HostMemoryPool};
use std::thread::sleep;
use std::time::Duration;
use task::{Event, TaskManager};

pub(crate) mod executor;
pub(crate) mod memory;
pub(crate) mod task;

struct Worker {
    comm: WorkerRpcSender,
    executors: Vec<CudaExecutorThread>,
    scheduler: TaskManager,
    shutdown_acknowledged: bool,
}

impl Worker {
    fn new(
        comm: WorkerRpcSender,
        scheduler: TaskManager,
        executors: Vec<CudaExecutorThread>,
    ) -> Result<Self> {
        Ok(Self {
            comm,
            scheduler,
            shutdown_acknowledged: false,
            executors,
        })
    }

    fn handle_message(&mut self, msg: DriverMsg) -> Result {
        use DriverMsg::*;
        match msg {
            Submit(ops) => {
                for op in ops {
                    self.scheduler.submit_task(op);
                }
            }
            Compile(id, def) => {
                self.handle_compilation_request(id, def)?;
            }
            Shutdown => {
                self.handle_shutdown_request()?;
            }
        }

        Ok(())
    }

    fn handle_event(&mut self, event: Event) -> Result {
        self.scheduler.handle_event(event);
        Ok(())
    }

    fn handle_compilation_request(&mut self, id: CudaKernelId, def: CudaModuleDef) -> Result {
        let mut response = Ok(());

        // This is all done synchronize. Is that ok? Maybe offload it to some thread.
        for exec in &self.executors {
            let result = exec.compile_kernel(id, def.clone());

            if let Err(e) = result {
                response = Err(e);
            }
        }

        self.comm
            .message_driver(WorkerMsg::CompileResult(id, response.map_err(Into::into)))?;
        Ok(())
    }

    fn handle_shutdown_request(&mut self) -> Result {
        self.scheduler.request_shutdown();
        Ok(())
    }

    fn has_shutdown(&mut self) -> bool {
        if self.shutdown_acknowledged {
            return true;
        }

        if !self.scheduler.has_shutdown() {
            return false;
        }

        self.shutdown_acknowledged = true;
        self.comm
            .message_driver(WorkerMsg::AcknowledgeShutdown)
            .unwrap();
        true
    }
}

fn initialize_cuda() -> Result<Vec<CudaContextHandle>> {
    cuda_init()?;
    let mut contexts = Vec::new();

    for device in CudaDevice::all()? {
        use CudaDeviceAttribute::*;
        let name = device.name()?;

        if !device.attribute(CONCURRENT_KERNELS)? == 0 {
            warn!("ignoring {}: does not support concurrent kernels", name);
            continue;
        }

        if !device.attribute(UNIFIED_ADDRESSING)? == 0 {
            warn!("ignoring {}: does not support unified addressing", name);
            continue;
        }

        if !device.attribute(CAN_MAP_HOST_MEMORY)? == 0 {
            warn!("ignoring {}: cannot map host memory", name);
            continue;
        }

        let context = cuda_create_context(device, CudaContextFlags::MAP_HOST)?;
        contexts.push(context);
    }

    // Start profilers.
    for &context in &contexts {
        context.try_with(|| cuda_profiler_start())?;
    }

    Ok(contexts)
}

unsafe fn destroy_cuda(contexts: Vec<CudaContextHandle>) -> Result {
    for &context in &contexts {
        if let Err(e) = context.try_with(|| cuda_profiler_stop()) {
            error!("failed to stop profiler: {:?}", e);
        }
    }

    // Stopping the profiler and immediately shutting down seems cause loss of profiler data.
    // Sleep for a second to allow CUDA to write back profiling results.
    sleep(Duration::from_secs(1));

    for &context in &contexts {
        if let Err(e) = cuda_destroy_context(context) {
            error!("failed to destroy CUDA context: {:?}", e);
        }
    }

    Ok(())
}

fn init_worker(
    config: WorkerConfig,
    sender: Sender<Event>,
    mem_sender: Sender<MemoryEvent>,
    rpc: &WorkerRpcSender,
    comm: &WorkerEndpoint,
    contexts: &[CudaContextHandle],
) -> Result<Worker> {
    let node_id = rpc.my_id();

    let host_mem = HostMemoryPool::new(contexts[0], config.host_mem_block, config.host_mem_max);
    let host_executor = HostThreadPool::new(node_id);

    let mut devices = vec![];
    let mut device_executors = vec![];
    let mut device_mems = vec![];

    for (i, &ctx) in enumerate(contexts) {
        let id = DeviceId::new(i);
        let executor = CudaExecutorThread::new(node_id, id, ctx, 2, config.specialization_policy)?;

        devices.push(DeviceInfo {
            id,
            executor_id: ExecutorId::new(node_id, ExecutorKind::Device(id)),
            memory_id: MemoryId::new(node_id, MemoryKind::Device(id)),
            capabilities: executor.capabilities()?,
        });

        device_executors.push(executor);

        let mem = ctx.try_with(|| {
            if let Some(size) = config.device_mem_max {
                CudaDeviceMem::empty(size)
            } else {
                let (free, _total) = ctx.memory_free_and_total()?;
                let mut space = 512 * 1024 * 1024;

                loop {
                    if free < space {
                        break Err(CudaError::new(CUDA_ERROR_OUT_OF_MEMORY).unwrap_err());
                    }

                    match CudaDeviceMem::empty(free - space) {
                        Err(e) if e.raw() == CUDA_ERROR_OUT_OF_MEMORY => {}
                        result => break result,
                    }

                    warn!(
                        "allocation for {} bytes, trying again with {} bytes",
                        free - space,
                        free - 2 * space
                    );
                    space *= 2;
                }
            }
        })?;

        device_mems.push(DeviceMemoryPool::new(mem));
    }

    let memory_capacity = host_mem.max_capacity();
    let hostname = hostname().to_string();

    let storage = if let Some(dir) = config.storage_dir {
        Some(Storage::new(dir, config.storage_capacity)?)
    } else {
        None
    };

    let copy_engine = CopyEngine::new(contexts.to_vec())?;

    let memory = unsafe {
        MemoryManager::new(
            mem_sender,
            host_mem,
            device_mems,
            copy_engine.clone(),
            storage,
        )
    };

    let executors = ExecutorSet::new(
        comm.clone(),
        host_executor,
        device_executors.to_vec(),
        copy_engine.clone(),
    );

    let scheduler = Box::new(GlobalScheduler::new(
        config.scheduling_lookahead_size,
        &devices,
    ));
    let manager = TaskManager::new(rpc.clone(), memory, executors, scheduler, sender.clone())?;

    rpc.message_driver(WorkerMsg::Initialize(Ok(WorkerInfo {
        node_id,
        executor_id: ExecutorId::new(node_id, ExecutorKind::Host),
        memory_id: MemoryId::new(node_id, MemoryKind::Host),
        hostname,
        memory_capacity,
        devices,
    })))?;

    Worker::new(rpc.clone(), manager, device_executors)
}

fn loop_worker(
    mut worker: Worker,
    receiver: Receiver<Event>,
    mem_receiver: Receiver<MemoryEvent>,
    comm: &WorkerRpcReceiver,
) -> Result {
    let mut select = Select::new();
    comm.register(&mut select);
    select.recv(&receiver);
    select.recv(&mem_receiver);

    while !worker.has_shutdown() {
        if let Ok(event) = mem_receiver.try_recv() {
            worker.scheduler.handle_memory_event(event);
        } else if let Ok(event) = receiver.try_recv() {
            worker.scheduler.handle_event(event);
        } else if let Some(msg) = comm.poll()? {
            worker.handle_message(msg)?;
        } else {
            select.ready();
        }
    }

    Ok(())
}

pub(crate) fn execute_worker(
    config: WorkerConfig,
    comm: WorkerEndpoint,
    rpc_sender: WorkerRpcSender,
    rpc_receiver: WorkerRpcReceiver,
) -> Result {
    let (sender, receiver) = unbounded();
    let (mem_sender, mem_receiver) = unbounded();
    let mut init_failed = None;

    match initialize_cuda() {
        Ok(contexts) => {
            match init_worker(config, sender, mem_sender, &rpc_sender, &comm, &contexts) {
                Ok(worker) => loop_worker(worker, receiver, mem_receiver, &rpc_receiver)?,
                Err(e) => init_failed = Some(e),
            }

            // Destroying the CUDA context is unsafe since it cannot be called while CUDA activities are
            // still in progress. We will assume it safe since we have returned from loop_worker, but this
            // may need to be revisited in the future.
            unsafe { destroy_cuda(contexts) }?;
        }
        Err(e) => init_failed = Some(e),
    }

    if let Some(err) = init_failed {
        rpc_sender.message_driver(WorkerMsg::Initialize(Err(err.into())))?;
    }

    trace!("worker is shutting down");
    Ok(())
}
