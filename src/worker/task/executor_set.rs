use crate::network::{Tag, WorkerEndpoint};
use crate::prelude::*;
use crate::types::dag::{NetworkOperation, OperationKind};
use crate::types::{
    ExecutorKind, GenericAccessor, HostAccessor, HostMutAccessor, TaskletInstance, UnifiedPtr,
    WorkerId,
};
use crate::worker::executor::{CudaExecutorThread, HostThreadPool};
use crate::worker::memory::CopyEngine;
use crate::worker::task::Completion;

pub(crate) struct ExecutorSet {
    comm: WorkerEndpoint,
    host: HostThreadPool,
    devices: Vec<CudaExecutorThread>,
    copy_engine: CopyEngine,
}

impl ExecutorSet {
    pub(crate) fn new(
        comm: WorkerEndpoint,
        host: HostThreadPool,
        devices: Vec<CudaExecutorThread>,
        copy_engine: CopyEngine,
    ) -> Self {
        Self {
            comm,
            host,
            devices,
            copy_engine,
        }
    }

    pub(crate) unsafe fn execute(
        &mut self,
        kind: &OperationKind,
        buffers: Vec<GenericAccessor>,
        completion: Completion,
    ) {
        use NetworkOperation::*;
        use OperationKind::*;

        match kind {
            &Execute {
                executor,
                ref tasklet,
                ..
            } => {
                self.execute_tasklet(executor, tasklet.clone(), buffers, completion);
            }
            &CopyData {
                ref src_transform,
                ref dst_transform,
                domain,
            } => {
                let (src_buffer, dst_buffer) = if buffers.len() == 1 {
                    (&buffers[0], &buffers[0])
                } else if buffers.len() == 2 {
                    (&buffers[0], &buffers[1])
                } else {
                    panic!("invalid number of buffers: {:?}", buffers.len())
                };

                let src_buffer = src_buffer.transform(&src_transform, domain);
                let dst_buffer = dst_buffer.transform(&dst_transform, domain);

                self.execute_copy(src_buffer, dst_buffer, completion);
            }
            &Network(task) => match task {
                RecvProbe { sender, tag } => {
                    assert_eq!(buffers.len(), 0);
                    self.execute_probe(sender, tag, completion);
                }
                SendData { receiver, tag } => {
                    assert_eq!(buffers.len(), 1);
                    let buffer = buffers[0].as_host().unwrap();

                    self.execute_send(receiver, tag, buffer, completion);
                }
                RecvData { sender, tag } => {
                    assert_eq!(buffers.len(), 1);
                    let buffer = buffers[0].as_host_mut().unwrap();

                    self.execute_recv(sender, tag, buffer, completion);
                }
            },
            other => {
                completion.complete(Err(anyhow!("not implemented: {:?}", other)));
            }
        }
    }

    unsafe fn execute_tasklet(
        &mut self,
        executor: ExecutorKind,
        task: TaskletInstance,
        buffers: Vec<GenericAccessor>,
        completion: Completion,
    ) {
        match executor {
            ExecutorKind::Host => {
                self.host.submit_tasklet(task, buffers, 0, completion);
            }
            ExecutorKind::Device(id) => {
                self.devices[id.get()].submit_tasklet(task, buffers, completion);
            }
        }
    }

    unsafe fn execute_copy(
        &mut self,
        src_buffer: GenericAccessor,
        dst_buffer: GenericAccessor,
        completion: Completion,
    ) {
        assert_eq!(src_buffer.data_type(), dst_buffer.data_type());

        use UnifiedPtr::*;

        match (src_buffer.as_ptr(), dst_buffer.as_ptr_mut()) {
            (Host(_), HostMut(_)) => {
                let src_buffer = src_buffer.as_host().unwrap();
                let dst_buffer = dst_buffer.as_host_mut().unwrap();

                self.host.submit_copy(src_buffer, dst_buffer, completion);
            }
            (Device(_, src_id), DeviceMut(_, dst_id)) if src_id == dst_id => {
                self.devices[src_id.get()].submit_copy(
                    src_buffer.as_device(src_id).unwrap(),
                    dst_buffer.as_device_mut(dst_id).unwrap(),
                    completion,
                );
            }
            (Device(src_ptr, src_id), DeviceMut(dst_ptr, dst_id)) => {
                self.copy_engine.copy_d2d_strided(
                    src_id,
                    src_ptr,
                    src_buffer.strides_in_bytes(),
                    dst_id,
                    dst_ptr,
                    dst_buffer.strides_in_bytes(),
                    src_buffer.extents(),
                    src_buffer.data_type().size_in_bytes(),
                    move |result| {
                        completion.complete(result);
                    },
                )
            }
            (Host(src_ptr), DeviceMut(dst_ptr, dst_id)) => {
                self.copy_engine.copy_h2d_strided(
                    src_ptr as *const (),
                    src_buffer.strides_in_bytes(),
                    dst_id,
                    dst_ptr,
                    dst_buffer.strides_in_bytes(),
                    src_buffer.extents(),
                    src_buffer.data_type().size_in_bytes(),
                    move |result| {
                        completion.complete(result);
                    },
                );
            }
            (Device(src_ptr, src_id), HostMut(dst_ptr)) => {
                self.copy_engine.copy_d2h_strided(
                    src_id,
                    src_ptr,
                    src_buffer.strides_in_bytes(),
                    dst_ptr as *mut (),
                    dst_buffer.strides_in_bytes(),
                    src_buffer.extents(),
                    src_buffer.data_type().size_in_bytes(),
                    move |result| {
                        completion.complete(result);
                    },
                );
            }
            other => panic!("unreachable ptrs: {:?}", other),
        }
    }

    unsafe fn execute_send(
        &mut self,
        receiver: WorkerId,
        tag: Tag,
        buffer: HostAccessor,
        completion: Completion,
    ) {
        self.comm.send_async(
            receiver,
            buffer.as_ptr(),
            buffer.size_in_bytes(),
            tag,
            move |result| {
                completion.complete(result);
            },
        );
    }

    unsafe fn execute_recv(
        &mut self,
        sender: WorkerId,
        tag: Tag,
        buffer: HostMutAccessor,
        completion: Completion,
    ) {
        self.comm.recv_async(
            sender,
            buffer.as_ptr_mut(),
            buffer.size_in_bytes(),
            tag,
            move |result| {
                completion.complete(result);
            },
        );
    }

    fn execute_probe(&mut self, sender: WorkerId, tag: Tag, completion: Completion) {
        self.comm.probe_async(sender, tag, move |result| {
            completion.complete(result);
        });
    }
}
