use lightning_codegen::ModuleDef as CudaModuleDef;
use lightning_cuda::prelude::CudaError;
use serde::{Deserialize, Serialize};

use crate::network::internal::NetworkError;
use crate::types::dag::Operation;
use crate::types::{CudaKernelId, EventId, SyncId, TaskletOutput, WorkerInfo};

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum DriverMsg {
    Submit(Vec<Operation>),
    Compile(CudaKernelId, CudaModuleDef),
    Shutdown,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum WorkerMsg {
    Initialize(Result<WorkerInfo, SerializedError>),
    Sync(SyncId, EventId),
    Complete(EventId, Result<TaskletOutput, SerializedError>),
    CompileResult(CudaKernelId, Result<(), SerializedError>),
    AcknowledgeShutdown,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) enum SerializedError {
    Cuda(CudaError),
    Network(NetworkError),
    Other(String),
}

impl From<&anyhow::Error> for SerializedError {
    fn from(r: &anyhow::Error) -> Self {
        use SerializedError::*;

        if let Some(e) = r.downcast_ref::<CudaError>() {
            Cuda(e.clone())
        } else if let Some(e) = r.downcast_ref::<NetworkError>() {
            Network(e.clone())
        } else {
            Other(r.to_string())
        }
    }
}

impl From<SerializedError> for anyhow::Error {
    fn from(r: SerializedError) -> Self {
        use SerializedError::*;

        match r {
            Cuda(e) => anyhow::Error::from(e),
            Network(e) => anyhow::Error::from(e),
            Other(msg) => anyhow::Error::msg(msg),
        }
    }
}

impl From<anyhow::Error> for SerializedError {
    fn from(r: anyhow::Error) -> Self {
        (&r).into()
    }
}

impl From<&SerializedError> for anyhow::Error {
    fn from(r: &SerializedError) -> Self {
        r.clone().into()
    }
}
