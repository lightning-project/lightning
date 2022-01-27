use crate::prelude::*;
use crate::types::dag::{NetworkOperation, OperationKind};
use crate::types::{DeviceInfo, EventId, ExecutorKind};
use crate::worker::task::manager::{Lock as OpLock, OperationHandle};
use lightning_core::util::OrderedQueue;

pub(crate) trait Scheduler {
    fn enqueue(&mut self, op: OperationHandle, lock: &OpLock);
    fn dequeue(&mut self, lock: &OpLock) -> Option<OperationHandle>;
    fn on_start(&mut self, op: &OperationHandle, lock: &OpLock);
    fn on_complete(&mut self, op: &OperationHandle, lock: &OpLock);
}

#[derive(Default)]
struct SimpleScheduler {
    ready_queue: OrderedQueue<EventId, OperationHandle>,
}

impl Scheduler for SimpleScheduler {
    fn enqueue(&mut self, handle: OperationHandle, _lock: &OpLock) {
        self.ready_queue.push(handle.id(), handle);
    }

    fn dequeue(&mut self, _lock: &OpLock) -> Option<OperationHandle> {
        match self.ready_queue.pop_min() {
            Some((_, task)) => Some(task),
            None => None,
        }
    }

    fn on_start(&mut self, _op: &OperationHandle, _lock: &OpLock) {
        //
    }

    fn on_complete(&mut self, _op: &OperationHandle, _lock: &OpLock) {
        //
    }
}

struct CapacityScheduler {
    max_concurrent_tasks: usize,
    num_tasks_active: usize,
    max_bytes_tasks: usize,
    sum_tasks_bytes: usize,
    inner: SimpleScheduler,
}

impl CapacityScheduler {
    fn new(max_scheduled: usize, max_bytes: usize) -> Self {
        Self {
            max_concurrent_tasks: max_scheduled,
            max_bytes_tasks: max_bytes,
            num_tasks_active: 0,
            sum_tasks_bytes: 0,
            inner: SimpleScheduler::default(),
        }
    }
}

impl Scheduler for CapacityScheduler {
    fn enqueue(&mut self, handle: OperationHandle, lock: &OpLock) {
        self.inner.enqueue(handle, lock);
    }

    fn dequeue(&mut self, lock: &OpLock) -> Option<OperationHandle> {
        if self.num_tasks_active >= self.max_concurrent_tasks {
            return None;
        }

        if self.sum_tasks_bytes >= self.max_bytes_tasks {
            return None;
        }

        self.inner.dequeue(lock)
    }

    fn on_start(&mut self, op: &OperationHandle, lock: &OpLock) {
        self.inner.on_start(op, lock);

        self.num_tasks_active += 1;
        for req in op.requests() {
            self.sum_tasks_bytes += req.chunk().size_in_bytes()
        }

        trace!(
            "start: {:?} (limits: {} / {} tasks, {} / {} bytes)",
            op.inner(),
            self.num_tasks_active,
            self.max_concurrent_tasks,
            self.sum_tasks_bytes,
            self.max_bytes_tasks
        );
    }

    fn on_complete(&mut self, op: &OperationHandle, lock: &OpLock) {
        self.inner.on_complete(op, lock);

        self.num_tasks_active -= 1;
        for req in op.requests() {
            self.sum_tasks_bytes -= req.chunk().size_in_bytes()
        }

        trace!(
            "finish: {:?} (limits: {} / {} tasks, {} / {} bytes)",
            op.id(),
            self.num_tasks_active,
            self.max_concurrent_tasks,
            self.sum_tasks_bytes,
            self.max_bytes_tasks
        );
    }
}

pub(crate) struct GlobalScheduler {
    host: CapacityScheduler,
    devices: Vec<CapacityScheduler>,
    network_incoming: SimpleScheduler,
    network_outgoing: CapacityScheduler,
    aux: SimpleScheduler,
}

impl GlobalScheduler {
    pub(crate) fn new(lookahead_size: usize, devices_info: &[DeviceInfo]) -> GlobalScheduler {
        let mut devices = vec![];
        devices.resize_with(devices_info.len(), || {
            CapacityScheduler::new(10, lookahead_size)
        });

        Self {
            host: CapacityScheduler::new(10, lookahead_size),
            devices,
            network_incoming: SimpleScheduler::default(),
            network_outgoing: CapacityScheduler::new(10, lookahead_size),
            aux: SimpleScheduler::default(),
        }
    }
}

impl GlobalScheduler {
    fn determine_subscheduler(&mut self, op: &OperationKind) -> &mut dyn Scheduler {
        use NetworkOperation::*;
        use OperationKind::*;

        match op {
            Network(SendData { .. }) => &mut self.network_outgoing,
            Network(RecvData { .. }) => &mut self.network_incoming,
            Execute { executor, .. } => match *executor {
                ExecutorKind::Host => &mut self.host,
                ExecutorKind::Device(id) => &mut self.devices[id.get()],
            },
            _ => &mut self.aux,
        }
    }
}

impl Scheduler for GlobalScheduler {
    fn enqueue(&mut self, op: OperationHandle, lock: &OpLock) {
        self.determine_subscheduler(op.inner()).enqueue(op, lock);
    }

    fn dequeue(&mut self, lock: &OpLock) -> Option<OperationHandle> {
        if let Some(result) = self.aux.dequeue(lock) {
            Some(result)
        } else if let Some(result) = self.host.dequeue(lock) {
            Some(result)
        } else if let Some(result) = self.network_incoming.dequeue(lock) {
            Some(result)
        } else if let Some(result) = self.network_outgoing.dequeue(lock) {
            Some(result)
        } else {
            let mut current: Option<(&mut CapacityScheduler, OperationHandle)> = None;
            let mut min_workload = usize::MAX;

            for device in &mut self.devices {
                if device.max_bytes_tasks < min_workload {
                    if let Some(new_task) = device.dequeue(lock) {
                        min_workload = device.max_bytes_tasks;

                        if let Some((old_device, old_task)) = current.replace((device, new_task)) {
                            old_device.enqueue(old_task, lock);
                        }
                    }
                }
            }

            current.map(|(_, task)| task)
        }
    }

    fn on_start(&mut self, op: &OperationHandle, lock: &OpLock) {
        self.determine_subscheduler(op.inner()).on_start(op, lock);
    }

    fn on_complete(&mut self, op: &OperationHandle, lock: &OpLock) {
        self.determine_subscheduler(op.inner())
            .on_complete(op, lock);
    }
}
