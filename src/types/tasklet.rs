use lazy_static::lazy_static;
use lightning_core::util::{AsAny, InlineByteBuf as ByteBuf};
use parking_lot::RwLock;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::any::{type_name, Any, TypeId};
use std::fmt::{self, Debug, Display};
use std::mem::{self, ManuallyDrop};

use crate::prelude::*;
use crate::types::{ExecutorId, GenericAccessor};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TaskletTypeId(u64);

impl TaskletTypeId {
    pub fn new(v: u64) -> Self {
        Self(v)
    }

    pub fn of<T: Tasklet>() -> Self {
        Self::from_type_id(TypeId::of::<T>())
    }

    pub fn from_type_id(id: TypeId) -> Self {
        TaskletTypeId(unsafe { std::mem::transmute_copy(&id) })
    }
}

impl Display for TaskletTypeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tasklet:{:#016X}", self.0)
    }
}

pub trait Tasklet: Debug + Clone + Send + Sync + Serialize + DeserializeOwned + Any {
    type Output: Debug + Clone + Send + Sync + Serialize + DeserializeOwned + Any;

    fn id() -> TaskletTypeId {
        TaskletTypeId::of::<Self>()
    }

    fn execute(self, arrays: &[GenericAccessor], executor: &dyn Executor) -> Result<Self::Output>;
}

pub trait Executor: Display + AsAny {
    fn id(&self) -> ExecutorId;
}

impl dyn Executor {
    pub fn downcast_ref<T: Executor>(&self) -> Result<&T> {
        if let Some(result) = self.as_any().downcast_ref::<T>() {
            Ok(result)
        } else {
            bail!("executor {} is not of type {:?}", self, type_name::<T>());
        }
    }
}

lazy_static! {
    static ref GLOBAL_TASKLET_REGISTRY: Registry = Registry::new();
}

struct Registry {
    inner: RwLock<HashMap<TaskletTypeId, Record>>,
}

struct Record {
    id: TaskletTypeId,
    type_id: TypeId,
    type_name: &'static str,
    debug: fn(&ByteBuf, &mut fmt::Formatter<'_>) -> fmt::Result,
    execute:
        fn(&ByteBuf, arrays: &[GenericAccessor], executor: &dyn Executor) -> Result<TaskletOutput>,
}

impl Registry {
    fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::default()),
        }
    }

    fn register<T: Tasklet>(&self) {
        use std::collections::hash_map::Entry;

        let id = T::id();
        let record = Record {
            id,
            type_id: TypeId::of::<T>(),
            type_name: type_name::<T>(),
            debug: |buffer, formatter| {
                let result: bincode::Result<T> = bincode::deserialize(&**buffer);
                match result {
                    Ok(task) => Debug::fmt(&task, formatter),
                    Err(err) => Debug::fmt(&err, formatter),
                }
            },
            execute: |buffer, arrays, executor| {
                let task: T = bincode::deserialize(&**buffer)?;
                let output = task.execute(arrays, executor)?;
                Ok(TaskletOutput {
                    id: T::id(),
                    data: ByteBuf::from(bincode::serialize(&output)?),
                })
            },
        };

        match self.inner.write().entry(id) {
            Entry::Vacant(e) => {
                e.insert(record);
            }
            Entry::Occupied(entry) => {
                let other = entry.get();
                assert_eq!(other.id, id);
                assert_eq!(other.type_id, record.type_id);
                assert_eq!(other.type_name, record.type_name);
            }
        }
    }

    fn package_task<T>(&self, task: &T) -> Result<TaskletInstance>
    where
        T: Tasklet,
    {
        let id = T::id();
        if !self.inner.read().contains_key(&id) {
            bail!(
                "tasklet {:?} (id: {}) is not registered",
                type_name::<T>(),
                id
            );
        }

        let data = bincode::serialize(&task)?;

        Ok(TaskletInstance {
            id,
            data: ByteBuf::from(data),
        })
    }

    fn package_task_with_callback<T, F>(
        &self,
        task: &T,
        callback: F,
    ) -> Result<(TaskletInstance, TaskletCallback)>
    where
        T: Tasklet,
        F: FnOnce(Result<T::Output>),
        F: Send + 'static,
    {
        let task = self.package_task(task)?;
        let callback = move |result: Result<ByteBuf>| {
            let result = match result {
                Ok(buffer) => bincode::deserialize(buffer.as_ref()).map_err(Into::into),
                Err(e) => Err(e),
            };

            (callback)(result);
        };

        let callback = TaskletCallback {
            id: task.id,
            inner: ManuallyDrop::new(Box::new(callback)),
        };

        Ok((task, callback))
    }

    fn execute_task(
        &self,
        task: &TaskletInstance,
        arrays: &[GenericAccessor],
        executor: &dyn Executor,
    ) -> Result<TaskletOutput> {
        let guard = self.inner.read();
        let processor = match guard.get(&task.id) {
            Some(e) => e,
            None => bail!("tasklet ??? (id: {}) is not registered", task.id),
        };

        (processor.execute)(&task.data, arrays, executor)
    }

    fn name_task(&self, id: TaskletTypeId) -> Result<&'static str> {
        let guard = self.inner.read();
        match guard.get(&id) {
            Some(e) => Ok(e.type_name),
            None => bail!("tasklet ??? (id: {}) is not registered", id),
        }
    }
}

// Serialized tasklet
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct TaskletInstance {
    id: TaskletTypeId,
    data: ByteBuf,
}

impl TaskletInstance {
    pub fn id(&self) -> TaskletTypeId {
        self.id
    }

    pub fn is<T: Tasklet>(&self) -> bool {
        self.id == T::id()
    }

    pub fn name(&self) -> Result<&str> {
        GLOBAL_TASKLET_REGISTRY.name_task(self.id)
    }

    pub fn execute(
        &self,
        arrays: &[GenericAccessor],
        executor: &dyn Executor,
    ) -> Result<TaskletOutput> {
        GLOBAL_TASKLET_REGISTRY.execute_task(self, arrays, executor)
    }
}

impl Debug for TaskletInstance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let registry = GLOBAL_TASKLET_REGISTRY.inner.read();

        if let Some(record) = registry.get(&self.id) {
            (record.debug)(&self.data, f)
        } else {
            f.debug_struct("TaskletInstance")
                .field("id", &self.id)
                .field("data", &format_args!("<{} bytes>", self.data.len()))
                .finish()
        }
    }
}

// Serialized output of tasklet
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct TaskletOutput {
    id: TaskletTypeId,
    data: ByteBuf,
}

impl Debug for TaskletOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PackagedTaskOutput")
            .field("id", &self.id)
            .field("data", &format_args!("<{} bytes>", self.data.len()))
            .finish()
    }
}

// Handles result of a serialized tasklet
pub(crate) struct TaskletCallback {
    id: TaskletTypeId,
    inner: ManuallyDrop<Box<dyn FnOnce(Result<ByteBuf>) + Send>>,
}

impl Debug for TaskletCallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskCallback")
            .field("id", &self.id)
            .field("inner", &"...")
            .finish()
    }
}

impl TaskletCallback {
    pub(crate) fn process(mut self, result: Result<TaskletOutput>) {
        if let Ok(m) = &result {
            assert_eq!(self.id, m.id);
        }

        let inner = unsafe { ManuallyDrop::take(&mut self.inner) };
        mem::forget(self);

        (inner)(result.map(|e| e.data));
    }
}

impl Drop for TaskletCallback {
    fn drop(&mut self) {
        let inner = unsafe { ManuallyDrop::take(&mut self.inner) };

        (inner)(Err(anyhow!("system is shutting down")))
    }
}

pub(crate) fn register_tasklet<T>()
where
    T: Tasklet,
{
    GLOBAL_TASKLET_REGISTRY.register::<T>()
}

pub(crate) fn package_tasklet<T>(task: &T) -> Result<TaskletInstance>
where
    T: Tasklet,
{
    GLOBAL_TASKLET_REGISTRY.package_task(task)
}

pub(crate) fn package_tasklet_with_callback<T, F>(
    task: &T,
    callback: F,
) -> Result<(TaskletInstance, TaskletCallback)>
where
    T: Tasklet,
    F: FnOnce(Result<T::Output>),
    F: Send + 'static,
{
    GLOBAL_TASKLET_REGISTRY.package_task_with_callback(task, callback)
}
