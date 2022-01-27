use bincode::Options;
use crossbeam::channel::{self, Receiver, Sender};
use crossbeam::channel::{Select, TryRecvError};
use serde::{de::DeserializeOwned, Serialize};
use std::panic::{catch_unwind, UnwindSafe};
use std::thread;

use super::internal::{MPIRank, Network, NetworkHandle, NetworkHandler, Tag};
use crate::network::internal::NetworkError;
use crate::network::message::{DriverMsg, WorkerMsg};
use crate::prelude::*;
use crate::types::WorkerId;

#[derive(Error, Debug)]
pub(crate) enum Error {
    #[error("bincode error: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("network error: {0}")]
    Network(#[from] NetworkError),
    #[error("connection was disconnected")]
    Disconnected,
}

fn rank(id: WorkerId) -> MPIRank {
    MPIRank(id.get())
}

fn serialize<S>(msg: &S) -> bincode::Result<Vec<u8>>
where
    S: Serialize,
{
    bincode::DefaultOptions::new()
        .with_varint_encoding()
        .serialize(msg)
}

fn deserialize<D>(buffer: &[u8]) -> bincode::Result<D>
where
    D: DeserializeOwned,
{
    bincode::DefaultOptions::new()
        .with_varint_encoding()
        .deserialize(buffer)
}

#[derive(Debug)]
enum Message {
    Remote(WorkerId, Vec<u8>),
    LocalWorker(WorkerMsg),
    LocalDriver(DriverMsg),
}

impl Message {
    fn into_worker_msg(self, my_id: WorkerId) -> Result<(WorkerId, WorkerMsg), Error> {
        use Message::*;
        match self {
            Remote(node_id, buffer) => Ok((node_id, deserialize(&buffer)?)),
            LocalWorker(m) => Ok((my_id, m)),
            LocalDriver(m) => panic!("expecting WorkerMessage, received DriverMessage: {:?}", m),
        }
    }

    fn into_driver_msg(self) -> Result<DriverMsg, Error> {
        use Message::*;
        match self {
            Remote(_node_id, buffer) => Ok(deserialize(&buffer)?),
            LocalDriver(m) => Ok(m),
            LocalWorker(m) => panic!("expecting DriverMessage, received WorkerMessage: {:?}", m),
        }
    }
}

type Callback = Box<dyn FnOnce(Result<(), NetworkError>) + Send>;

struct Handler {
    outbox: Sender<Message>,
}

impl NetworkHandler for Handler {
    type Token = Callback;

    fn handle_message(&mut self, source: MPIRank, message: &[u8]) {
        let source = WorkerId::new(source.0);
        let buffer = message.to_vec();

        if let Err(e) = self.outbox.send(Message::Remote(source, buffer)) {
            warn!("error: {}", e);
        }
    }

    fn transfer_finished(&mut self, token: Callback, result: Result<(), NetworkError>) {
        (token)(result);
    }

    fn probe_finished(&mut self, token: Self::Token, result: Result<(MPIRank, Tag), NetworkError>) {
        (token)(result.map(|_| ()))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct WorkerEndpoint {
    network: NetworkHandle<Callback>,
}

impl WorkerEndpoint {
    pub(crate) fn my_id(&self) -> WorkerId {
        WorkerId::new(self.network.my_rank().0)
    }

    pub(crate) unsafe fn send_async<F: FnOnce(Result<(), NetworkError>) + Send + 'static>(
        &self,
        dst: WorkerId,
        send_buffer: *const u8,
        size: usize,
        tag: Tag,
        callback: F,
    ) {
        if let Err(e) = self.network.send_async(
            rank(dst),
            send_buffer,
            size,
            tag,
            Box::new(callback) as Callback,
        ) {
            (e.token)(Err(e.error));
        }
    }

    pub(crate) unsafe fn recv_async<F: FnOnce(Result<(), NetworkError>) + Send + 'static>(
        &self,
        src: WorkerId,
        recv_buffer: *mut u8,
        size: usize,
        tag: Tag,
        callback: F,
    ) {
        if let Err(e) = self.network.recv_async(
            rank(src),
            recv_buffer,
            size,
            tag,
            Box::new(callback) as Callback,
        ) {
            (e.token)(Err(e.error));
        }
    }

    pub(crate) fn probe_async<F: FnOnce(Result<(), NetworkError>) + Send + 'static>(
        &self,
        src: WorkerId,
        tag: Tag,
        callback: F,
    ) {
        if let Err(e) =
            self.network
                .probe_async(Some(rank(src)), Some(tag), Box::new(callback) as Callback)
        {
            (e.token)(Err(e.error));
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct WorkerRpcSender {
    network: NetworkHandle<Callback>,
    local_driver_outbox: Option<Sender<Message>>,
    driver_id: WorkerId,
}

impl WorkerRpcSender {
    pub(crate) fn message_driver(&self, message: WorkerMsg) -> Result<(), Error> {
        debug!("sending to driver: {:#?}", message);
        if let Some(outbox) = &self.local_driver_outbox {
            outbox
                .send(Message::LocalWorker(message))
                .map_err(|_| NetworkError::Disconnected)?;
        } else {
            self.network
                .message(rank(self.driver_id), serialize(&message)?)?;
        }

        Ok(())
    }

    pub(crate) fn my_id(&self) -> WorkerId {
        WorkerId::new(self.network.my_rank().0)
    }
}

#[derive(Debug)]
pub(crate) struct WorkerRpcReceiver {
    inbox: Receiver<Message>,
}

impl WorkerRpcReceiver {
    pub(crate) fn register<'a>(&'a self, select: &mut Select<'a>) {
        let _ = select.recv(&self.inbox);
    }

    /// Returns
    /// * Ok(Some(message)) if a message was received
    /// * Ok(None) if no message is available
    /// * Err(e) if an error occurred while attempting to receive or deserialize a message.
    pub(crate) fn poll(&self) -> Result<Option<DriverMsg>, Error> {
        match self.inbox.try_recv() {
            Ok(m) => Ok(Some(m.into_driver_msg()?)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(Error::Disconnected),
        }
    }
}

#[derive(Debug)]
pub(crate) struct DriverRpcSender {
    network: NetworkHandle<Callback>,
    local_worker_outbox: Sender<Message>,
}

impl DriverRpcSender {
    pub(crate) fn message_worker(
        &self,
        worker_id: WorkerId,
        message: DriverMsg,
    ) -> Result<(), Error> {
        trace!("sending to worker {:?}: {:#?}", worker_id, message);
        let rank = rank(worker_id);

        if rank != self.network.my_rank() {
            self.network.message(rank, serialize(&message)?)?;
        } else {
            self.local_worker_outbox
                .send(Message::LocalDriver(message))
                .map_err(|_| Error::Disconnected)?;
        }

        Ok(())
    }

    pub(crate) fn message_all(&self, message: DriverMsg) -> Result<(), Error> {
        trace!("sending to all workers: {:#?}", message);
        if self.num_workers() > 1 {
            let buffer = serialize(&message)?;

            for rank in 0..self.num_workers() {
                if MPIRank(rank) != self.network.my_rank() {
                    self.network.message(MPIRank(rank), buffer.clone())?;
                }
            }
        }

        self.local_worker_outbox
            .send(Message::LocalDriver(message))
            .map_err(|_| Error::Disconnected)?;

        Ok(())
    }

    pub(crate) fn num_workers(&self) -> usize {
        self.network.num_ranks().0 as usize
    }

    pub(crate) fn max_tag(&self) -> Tag {
        self.network.max_tag()
    }
}

#[derive(Debug)]
pub(crate) struct DriverRpcReceiver {
    my_id: WorkerId,
    local_worker_inbox: Receiver<Message>,
    network_inbox: Receiver<Message>,
}

impl DriverRpcReceiver {
    pub(crate) fn register<'a>(&'a self, select: &mut Select<'a>) {
        let _ = select.recv(&self.local_worker_inbox);
        let _ = select.recv(&self.network_inbox);
    }

    pub(crate) fn poll(&self) -> Result<Option<(WorkerId, WorkerMsg)>, Error> {
        match self.local_worker_inbox.try_recv() {
            Ok(m) => return Ok(Some(m.into_worker_msg(self.my_id)?)),
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => return Err(Error::Disconnected),
        };

        match self.network_inbox.try_recv() {
            Ok(m) => return Ok(Some(m.into_worker_msg(self.my_id)?)),
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => return Err(Error::Disconnected),
        };

        Ok(None)
    }
}

pub(crate) fn execute_endpoints<F, G>(driver_main: F, worker_main: G) -> Result<(), Error>
where
    F: FnOnce(DriverRpcSender, DriverRpcReceiver) + UnwindSafe,
    G: Send + 'static + FnOnce(WorkerRpcSender, WorkerRpcReceiver, WorkerEndpoint) + UnwindSafe,
{
    let (network_outbox, network_inbox) = channel::unbounded();
    let network = Network::new(Handler {
        outbox: network_outbox,
    })?;

    let driver_id = WorkerId(0);
    let is_driver = network.handle().my_rank() == rank(driver_id);

    let network_ref = network.handle();
    let my_id = WorkerId(network_ref.my_rank().0 as _);

    if is_driver {
        let (worker_outbox, worker_inbox) = channel::unbounded();
        let (driver_outbox, driver_inbox) = channel::unbounded();

        let worker = thread::Builder::new()
            .name(format!("worker-{}", hostname()))
            .spawn(move || {
                worker_main(
                    WorkerRpcSender {
                        network: network_ref.clone(),
                        driver_id,
                        local_driver_outbox: Some(driver_outbox.clone()),
                    },
                    WorkerRpcReceiver {
                        inbox: worker_inbox.clone(),
                    },
                    WorkerEndpoint {
                        network: network_ref,
                    },
                );

                if let Ok(msg) = worker_inbox.recv() {
                    panic!("received message after worker terminated: {:?}", msg);
                }
            })
            .unwrap();

        let network_ref = network.handle();
        let result = catch_unwind(|| {
            driver_main(
                DriverRpcSender {
                    network: network_ref,
                    local_worker_outbox: worker_outbox,
                },
                DriverRpcReceiver {
                    my_id,
                    local_worker_inbox: driver_inbox,
                    network_inbox,
                },
            );
        });

        // Forcefully close connection.
        drop(network);

        worker.join().unwrap();
        result.unwrap();
    } else {
        thread::Builder::new()
            .name(format!("worker-{}", hostname()))
            .spawn(move || {
                worker_main(
                    WorkerRpcSender {
                        network: network_ref.clone(),
                        driver_id,
                        local_driver_outbox: None,
                    },
                    WorkerRpcReceiver {
                        inbox: network_inbox,
                    },
                    WorkerEndpoint {
                        network: network_ref,
                    },
                );
            })
            .unwrap()
            .join()
            .unwrap();
    };

    Ok(())
}
