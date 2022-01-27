use crate::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug, Display};
use std::hint::unreachable_unchecked;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WorkerId(pub u8);

impl WorkerId {
    pub fn new(i: usize) -> Self {
        WorkerId(i as u8)
    }

    pub fn get(&self) -> usize {
        self.0 as usize
    }
}

impl Display for WorkerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Maximum number of possible devices.
pub const MAX_DEVICES: usize = 4;

/// A devices number between 0 and [`MAX_DEVICES`].
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DeviceId(DeviceIdPriv);

// Device id is represented using an enum. This has two advantages:
//  * Value is guaranteed to be in range 0..MAX_DEVICES
//  * Compiler can apply niche optimizations
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
enum DeviceIdPriv {
    _0 = 0,
    _1 = 1,
    _2 = 2,
    _3 = 3,
}

impl DeviceId {
    pub fn new(device_index: usize) -> Self {
        Self(match device_index {
            0 => DeviceIdPriv::_0,
            1 => DeviceIdPriv::_1,
            2 => DeviceIdPriv::_2,
            3 => DeviceIdPriv::_3,
            _ => panic!("device id out of range"),
        })
    }

    #[inline]
    pub fn get(&self) -> usize {
        let x = self.0 as u8 as usize;

        // Helps compile understand that get is in range 0..MAX_DEVICES
        if x >= MAX_DEVICES {
            unsafe { unreachable_unchecked() }
        }

        x
    }
}

impl Debug for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DeviceId").field(&self.get()).finish()
    }
}

impl Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.get())
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Debug, Hash)]
pub enum ExecutorKind {
    Host,
    Device(DeviceId),
}

impl ExecutorKind {
    pub fn best_affinity_memory(&self) -> MemoryKind {
        match *self {
            ExecutorKind::Host => MemoryKind::Host,
            ExecutorKind::Device(id) => MemoryKind::Device(id),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Debug, Hash)]
pub struct ExecutorId {
    pub node: WorkerId,
    pub executor: ExecutorKind,
}

impl ExecutorId {
    pub fn new(node: WorkerId, executor: ExecutorKind) -> Self {
        Self { node, executor }
    }

    pub fn node_id(&self) -> WorkerId {
        self.node
    }

    pub fn kind(&self) -> ExecutorKind {
        self.executor
    }

    pub fn best_affinity_memory(&self) -> MemoryId {
        MemoryId::new(self.node, self.executor.best_affinity_memory())
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Debug, Hash)]
pub enum MemoryKind {
    FileSystem,
    Host,
    Device(DeviceId),
}

impl MemoryKind {
    pub fn device_id(&self) -> Option<DeviceId> {
        match self {
            &MemoryKind::Device(i) => Some(i),
            _ => None,
        }
    }

    pub fn best_affinity_executor(&self) -> ExecutorKind {
        match *self {
            MemoryKind::FileSystem => ExecutorKind::Host,
            MemoryKind::Host => ExecutorKind::Host,
            MemoryKind::Device(id) => ExecutorKind::Device(id),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Debug, Hash)]
pub struct MemoryId {
    node: WorkerId,
    kind: MemoryKind,
}

impl MemoryId {
    pub fn new(node: WorkerId, kind: MemoryKind) -> Self {
        Self { node, kind }
    }

    pub fn node_id(&self) -> WorkerId {
        self.node
    }

    pub fn kind(&self) -> MemoryKind {
        self.kind
    }

    pub fn best_affinity_executor(&self) -> ExecutorId {
        ExecutorId {
            node: self.node,
            executor: match self.kind {
                MemoryKind::Device(id) => ExecutorKind::Device(id),
                _ => ExecutorKind::Host,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    workers: Vec<WorkerInfo>,
    devices: Vec<DeviceInfo>,
}

impl SystemInfo {
    pub fn new(workers: Vec<WorkerInfo>) -> Self {
        let devices = workers
            .iter()
            .flat_map(|e| &e.devices)
            .cloned()
            .collect_vec();

        Self { workers, devices }
    }

    pub fn workers(&self) -> &[WorkerInfo] {
        &self.workers
    }

    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    pub fn worker(&self, id: WorkerId) -> Option<&WorkerInfo> {
        self.workers.iter().find(|r| r.node_id == id)
    }

    pub fn device(&self, id: DeviceId) -> Option<&DeviceInfo> {
        self.devices.iter().find(|r| r.id == id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub node_id: WorkerId,
    pub executor_id: ExecutorId,
    pub memory_id: MemoryId,
    pub hostname: String,
    pub memory_capacity: usize,
    pub devices: Vec<DeviceInfo>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DeviceInfo {
    pub id: DeviceId,
    pub executor_id: ExecutorId,
    pub memory_id: MemoryId,
    pub capabilities: DeviceCapabilities,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DeviceCapabilities {
    pub name: String,
    pub ordinal: usize,
    pub memory_capacity: usize,
    pub compute_capability: (i32, i32),
    pub clock_rate: usize,
    pub memory_clock_rate: usize,
    pub memory_bus_width: usize,
    pub multiprocessor_count: usize,
    pub async_engine_count: usize,
}
