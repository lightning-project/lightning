use crate::prelude::*;
use crate::types::dag::{NetworkOperation, Operation, OperationChunk, OperationKind};
use crate::types::{CudaKernelId, WorkerId};
use lightning_codegen::ModuleDef;
use serde_json::{json, Value as Json};
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Debug)]
pub(super) struct PlanTrace {
    file: File,
    kernel_names: HashMap<CudaKernelId, String>,
}

impl PlanTrace {
    pub(super) fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().canonicalize()?;
        info!("writing trace to: {:?}", path);

        Ok(Self {
            file: File::create(path)?,
            kernel_names: default(),
        })
    }

    pub(super) fn kernel_compiled(&mut self, id: CudaKernelId, def: ModuleDef) {
        self.kernel_names.insert(id, def.kernel.function_name);
    }

    pub(super) fn add(&mut self, plan: &[Vec<Operation>]) {
        let result = self.convert_plan(&plan);

        if let Err(e) = serde_json::to_writer(&self.file, &result) {
            warn!("writing trace failed: {}", e);
        }

        let _ = self.file.write(&[b'\n']);
    }

    fn convert_plan(&self, plan: &[Vec<Operation>]) -> Json {
        let mut tasks = vec![];

        for (worker_id, ops) in enumerate(plan) {
            for op in ops {
                tasks.push(self.process_op(WorkerId::new(worker_id), op));
            }
        }

        Json::Array(tasks)
    }

    fn process_op(&self, node_id: WorkerId, op: &Operation) -> Json {
        json!({
            "node_id": node_id.get(),
            "id": op.event_id.get(),
            "dependencies": op.dependencies.iter().map(|e| e.get()).collect_vec(),
            "chunks": op.chunks.iter().map(|e| self.process_chunk(e)).collect_vec(),
            "task": self.process_kind(op.kind.as_deref().unwrap_or(&OperationKind::Empty)),
        })
    }

    fn process_chunk(&self, chunk: &OperationChunk) -> Json {
        json!({
            "id": chunk.id.get(),
            "exclusive": chunk.exclusive,
            "dependencies": chunk.dependency.iter().map(|e| e.get()).collect_vec(),
        })
    }

    fn process_kind(&self, op: &OperationKind) -> Json {
        use NetworkOperation::*;
        use OperationKind::*;

        match op {
            CreateChunk { id, .. } => {
                json!({
                    "kind": "create_chunk",
                    "chunk_id": id.get(),
                })
            }
            DestroyChunk { id } => {
                json!({
                    "kind": "destroy_chunk",
                    "chunk_id": id.get(),
                })
            }
            CopyData { .. } => {
                json!({
                    "kind": "copy",
                })
            }
            Network(SendData { receiver, tag }) => {
                json!({
                    "kind": "send",
                    "receiver": receiver.get(),
                    "tag": tag,
                })
            }
            Network(RecvData { sender, tag }) => {
                json!({
                    "kind": "recv",
                    "sender": sender.get(),
                    "tag": tag,
                })
            }
            Network(RecvProbe { sender, tag }) => {
                json!({
                    "kind": "probe",
                    "sender": sender.get(),
                    "tag": tag,
                })
            }
            Sync { id } => {
                json!({
                    "kind": "sync",
                    "sync_id": id.0.get(),
                })
            }
            Execute { tasklet, .. } => {
                json!({
                    "kind": "execute",
                    "name": tasklet.name().unwrap(),
                })
            }
            Empty => {
                json!({
                    "kind": "empty",
                })
            }
        }
    }
}
