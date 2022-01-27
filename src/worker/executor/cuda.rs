use crossbeam::channel::{self, Receiver, Sender};
use lightning_codegen::{Kernel, KernelArg, KernelConfig, KernelSpecializationPolicy, ModuleDef};
use lightning_cuda::prelude::*;
use lightning_cuda::Dim3 as CudaDim3;
use lightning_memops::MemOpsKernelsCache;
use parking_lot::{Condvar, RwLock};
use std::fmt::{self, Display};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crate::prelude::*;
use crate::types::{
    Affine, CudaAccessor, CudaAccessor4, CudaArg, CudaKernelId, CudaMutAccessor, CudaMutAccessor3,
    DataValue, DeviceCapabilities, DeviceId, Dim, Executor, ExecutorId, ExecutorKind,
    GenericAccessor, Point, Rect, Reduction, TaskletInstance, Transform, WorkerId, MAX_DIMS,
};
use crate::worker::task::Completion;

#[derive(Debug)]
struct Inner {
    id: DeviceId,
    name: String,
    node_id: WorkerId,
    context: CudaContextHandle,
    policy: KernelSpecializationPolicy,
    kernels: RwLock<HashMap<CudaKernelId, Mutex<Kernel>>>,
    memops_kernels: MemOpsKernelsCache,
    condvar: Condvar,
}

#[derive(Debug)]
enum QueueItem {
    Execute {
        task: TaskletInstance,
        buffers: Vec<GenericAccessor>,
        completion: Completion,
    },
    Copy {
        src: CudaAccessor,
        dst: CudaMutAccessor,
        reduction: Option<Reduction>,
        completion: Completion,
    },
    Shutdown,
}

#[derive(Debug, Clone)]
pub(crate) struct CudaExecutorThread {
    inner: Arc<Inner>,
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
    sender: Sender<QueueItem>,
}

impl CudaExecutorThread {
    pub(crate) fn new(
        node_id: WorkerId,
        id: DeviceId,
        context: CudaContextHandle,
        num_streams: usize,
        policy: KernelSpecializationPolicy,
    ) -> Result<Self> {
        let (sender, receiver) = channel::unbounded();

        let name = context.device()?.name()?;
        let inner = Arc::new(Inner {
            id,
            node_id,
            name,
            context,
            policy,
            kernels: default(),
            memops_kernels: default(),
            condvar: default(),
        });

        let mut executors = vec![];
        for _ in 0..num_streams {
            executors.push(CudaExecutor {
                stream: context.try_with(CudaStream::new)?,
                inner: Arc::clone(&inner),
            });
        }

        let mut handles = vec![];
        for (index, executor) in enumerate(executors) {
            let receiver = receiver.clone();

            handles.push(
                thread::Builder::new()
                    .name(format!("cuda-executor-{}", index))
                    .spawn(move || main_loop(executor, receiver))
                    .unwrap(),
            );
        }

        let handles = Arc::new(Mutex::new(handles));

        Ok(Self {
            sender,
            inner,
            handles,
        })
    }

    pub(crate) fn context(&self) -> CudaContextHandle {
        self.inner.context
    }

    pub(crate) fn name(&self) -> &str {
        &self.inner.name
    }

    pub(crate) fn capabilities(&self) -> Result<DeviceCapabilities> {
        use lightning_cuda::DeviceAttribute::*;
        let device = self.inner.context.device()?;

        Ok(DeviceCapabilities {
            name: device.name()?,
            ordinal: device.ordinal(),
            memory_capacity: device.total_memory()?,
            compute_capability: device.compute_capability()?,
            clock_rate: device.attribute(CLOCK_RATE)? as _,
            memory_clock_rate: device.attribute(MEMORY_CLOCK_RATE)? as _,
            memory_bus_width: device.attribute(GLOBAL_MEMORY_BUS_WIDTH)? as _,
            multiprocessor_count: device.attribute(MULTIPROCESSOR_COUNT)? as _,
            async_engine_count: device.attribute(ASYNC_ENGINE_COUNT)? as _,
        })
    }

    pub(crate) fn compile_kernel(&self, id: CudaKernelId, def: ModuleDef) -> Result {
        let mut kernel = Kernel::new(def, self.inner.policy);
        kernel.compile(self.inner.context, KernelConfig::default())?;

        self.inner.kernels.write().insert(id, Mutex::new(kernel));
        Ok(())
    }

    pub(crate) unsafe fn submit_copy(
        &self,
        src: CudaAccessor,
        dst: CudaMutAccessor,
        completion: Completion,
    ) {
        assert_eq!(src.extents(), dst.extents());
        assert_eq!(src.data_type(), dst.data_type());

        self.schedule_action(QueueItem::Copy {
            src,
            dst,
            completion,
            reduction: None,
        });
    }

    // Unsafe: caller must guarantee that the arguments live until completion is called.
    pub(crate) unsafe fn submit_tasklet(
        &self,
        task: TaskletInstance,
        buffers: Vec<GenericAccessor>,
        completion: Completion,
    ) {
        self.schedule_action(QueueItem::Execute {
            task,
            buffers,
            completion,
        });
    }

    fn schedule_action(&self, action: QueueItem) {
        let _ = self.sender.send(action);
    }

    pub(crate) fn shutdown_and_wait(&self) {
        let handles = take(&mut *self.handles.lock());

        for _ in &handles {
            self.schedule_action(QueueItem::Shutdown);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        self.inner.kernels.write().clear();
    }
}

fn main_loop(executor: CudaExecutor, receiver: Receiver<QueueItem>) {
    use QueueItem::*;
    let inner = &*executor.inner;
    let _guard = inner.context.activate().unwrap();

    while let Ok(item) = receiver.recv() {
        match item {
            Execute {
                task,
                buffers,
                completion,
            } => {
                let task_result = task.execute(&buffers, &executor);
                let sync_result = executor.stream.synchronize().map_err(|e| e.into());
                completion.complete_tasklet(Result::and(sync_result, task_result));
            }
            Copy {
                src,
                dst,
                reduction,
                completion,
            } => unsafe {
                let result = match reduction {
                    Some(red) => executor.fold_async(src, dst, red),
                    None => executor.copy_async(src, dst),
                };

                let sync_result = executor.stream.synchronize();

                let result = match (result, sync_result) {
                    (Ok(()), Ok(())) => Ok(()),
                    (Err(e), _) => Err(e),
                    (_, Err(e)) => Err(e.into()),
                };

                completion.complete(result)
            },
            Shutdown => break,
        }
    }
}

#[derive(Debug)]
pub struct CudaExecutor {
    stream: CudaStream,
    inner: Arc<Inner>,
}

impl Executor for CudaExecutor {
    fn id(&self) -> ExecutorId {
        ExecutorId::new(self.inner.node_id, ExecutorKind::Device(self.inner.id))
    }
}

impl Display for CudaExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.id())
    }
}

impl CudaExecutor {
    pub fn device_id(&self) -> DeviceId {
        self.inner.id
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    pub fn context(&self) -> CudaContextHandle {
        self.inner.context
    }

    pub unsafe fn fill_async(&self, dst: CudaMutAccessor, value: DataValue) -> Result {
        let inner = &self.inner;

        lightning_memops::cuda_fill(
            inner.context,
            &self.stream,
            &inner.memops_kernels,
            dst,
            value,
        )
    }

    pub unsafe fn copy_async(&self, src: CudaAccessor, dst: CudaMutAccessor) -> Result {
        let inner = &self.inner;

        lightning_memops::cuda_copy(inner.context, &self.stream, &inner.memops_kernels, src, dst)
    }

    pub unsafe fn reduce_async(
        &self,
        src: CudaAccessor4,
        dst: CudaMutAccessor3,
        axis: usize,
        reduction: Reduction,
    ) -> Result {
        let inner = &self.inner;

        lightning_memops::cuda_reduce(
            inner.context,
            &self.stream,
            &inner.memops_kernels,
            src,
            dst,
            axis,
            reduction,
        )
    }

    pub unsafe fn fold_async(
        &self,
        src: CudaAccessor,
        dst: CudaMutAccessor,
        reduction: Reduction,
    ) -> Result {
        let inner = &self.inner;

        lightning_memops::cuda_fold(
            inner.context,
            &self.stream,
            &inner.memops_kernels,
            src,
            dst,
            reduction,
        )
    }

    pub fn with_kernel<F>(&self, id: CudaKernelId, callback: F) -> Result
    where
        F: FnOnce(&CudaStream, &mut Kernel) -> Result,
    {
        let guard = self.inner.kernels.read();
        if let Some(e) = guard.get(&id) {
            (callback)(&self.stream, &mut e.lock())
        } else {
            bail!("no kernel found for id: {}", id);
        }
    }

    pub unsafe fn launch_kernel_async(
        &self,
        id: CudaKernelId,
        block_count: Dim,
        block_size: Dim,
        block_offset: Point,
        smem_size: u32,
        args: &[CudaArg],
        arrays: &[GenericAccessor],
    ) -> Result {
        fn to_dim3(v: [u64; 3]) -> Result<CudaDim3> {
            match (v[0].try_into(), v[1].try_into(), v[2].try_into()) {
                (Ok(x), Ok(y), Ok(z)) => Ok(CudaDim3::new(x, y, z)),
                _ => bail!("failed to cast {:?} to Dim3", v),
            }
        }

        unsafe fn build_array_arg_const<const N: usize>(
            array: CudaAccessor,
            ndims: usize,
            lbnd: [i64; N],
            ubnd: [i64; N],
            transform: [[i64; N]; MAX_DIMS],
            translate: [i64; MAX_DIMS],
        ) -> Result<KernelArg> {
            let mut is_empty = false;

            for j in 0..ndims {
                if lbnd[j] > ubnd[j] {
                    bail!("invalid access bounds");
                }

                is_empty |= lbnd[j] == ubnd[j];
            }

            if !is_empty {
                let array_size = array.extents();

                for i in 0..MAX_DIMS {
                    let mut min = translate[i];
                    let mut max = translate[i];

                    for j in 0..ndims {
                        let f = transform[i][j];
                        min = min.saturating_add(i64::min(f * lbnd[j], f * (ubnd[j] - 1)));
                        max = max.saturating_add(i64::max(f * lbnd[j], f * (ubnd[j] - 1)));
                    }

                    if min < 0 || max < min || (max as u64) >= array_size[i] {
                        bail!(
                            "failed to launch kernel: access {}..={} out of bounds 0..{}",
                            min,
                            max,
                            array_size[i]
                        );
                    }
                }
            }

            let old_strides = array.strides();
            let mut new_strides = [0; N];

            for i in 0..MAX_DIMS {
                for j in 0..ndims {
                    new_strides[j] += transform[i][j] * old_strides[i];
                }
            }

            let mut ptr = array.as_ptr().raw();
            let dtype = array.data_type();
            let elem_size = dtype.size_in_bytes() as u64;
            for i in 0..MAX_DIMS {
                ptr = ptr.wrapping_add((translate[i] * old_strides[i]) as u64 * elem_size);
            }

            Ok(KernelArg::array_dyn(
                ndims,
                dtype,
                CudaDevicePtr::from_raw(ptr),
                &lbnd,
                &ubnd,
                &new_strides,
            ))
        }

        unsafe fn build_array_arg(
            array: CudaAccessor,
            ndims: usize,
            domain: Rect,
            affine: &Affine,
        ) -> Result<KernelArg> {
            let mut lbnd = [0; MAX_DIMS];
            let mut ubnd = [0; MAX_DIMS];

            for i in 0..MAX_DIMS {
                lbnd[i] = domain.low()[i] as i64;
                ubnd[i] = domain.high()[i] as i64;
            }

            let matrix = if let Some(transform) = affine.transform() {
                **transform
            } else {
                *Transform::identity()
            };

            build_array_arg_const(array, ndims, lbnd, ubnd, matrix, *affine.translate())
        }

        unsafe fn build_array_per_block_arg(
            array: CudaAccessor,
            block_count: Dim,
            ndims: usize,
            domain: Rect,
            affine: &Affine,
            per_block: &Transform,
        ) -> Result<KernelArg> {
            let mut lbnd = [0; 2 * MAX_DIMS];
            let mut ubnd = [0; 2 * MAX_DIMS];

            for i in 0..MAX_DIMS {
                lbnd[i] = 0;
                ubnd[i] = block_count[i] as i64;

                lbnd[MAX_DIMS + i] = domain.low()[i] as i64;
                ubnd[MAX_DIMS + i] = domain.high()[i] as i64;
            }

            let mut matrix = [[0; 2 * MAX_DIMS]; MAX_DIMS];
            for i in 0..MAX_DIMS {
                for j in 0..MAX_DIMS {
                    matrix[i][j] = per_block[i][j];
                }
            }

            if let Some(transform) = affine.transform() {
                for i in 0..MAX_DIMS {
                    for j in 0..MAX_DIMS {
                        matrix[i][j + MAX_DIMS] = transform[i][j];
                    }
                }
            } else {
                for i in 0..MAX_DIMS {
                    for j in 0..MAX_DIMS {
                        matrix[i][j + MAX_DIMS] = (i == j) as i64;
                    }
                }
            }

            build_array_arg_const(array, ndims + 3, lbnd, ubnd, matrix, *affine.translate())
        }

        self.with_kernel(id, |stream, kernel| {
            let device_id = self.inner.id;
            let mut largs = Vec::with_capacity(args.len());

            for arg in args {
                largs.push(match &arg {
                    CudaArg::Value(ref v) => KernelArg::value(v.clone()),
                    CudaArg::Array(arg) => {
                        let array = arrays
                            .get(arg.array_index)
                            .ok_or_else(|| anyhow!("array index out of bounds"))?
                            .as_device(device_id)
                            .ok_or_else(|| anyhow!("invalid array type"))?;

                        if let Some(per_block) = &arg.per_block {
                            build_array_per_block_arg(
                                array,
                                block_count,
                                arg.ndims,
                                arg.domain,
                                &arg.transform,
                                per_block,
                            )?
                        } else {
                            build_array_arg(array, arg.ndims, arg.domain, &arg.transform)?
                        }
                    }
                })
            }

            kernel.launch_async(
                self.context(),
                stream,
                block_count,
                block_size,
                block_offset.to_dim(),
                smem_size,
                &largs,
            )?;

            Ok(())
        })
    }
}
