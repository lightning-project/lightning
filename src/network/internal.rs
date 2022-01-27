use crossbeam::channel::{self, Receiver, Sender, TryRecvError};
use mpi_sys::*;
use serde::{Deserialize, Serialize};
use std::ffi::c_void;
use std::fmt::{self, Debug, Display};
use std::mem::MaybeUninit;
use std::os::raw::c_int;
use std::ptr;
use std::thread;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::prelude::*;

#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Tag(pub c_int);

impl Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MPIRank(pub(crate) usize);

impl Display for MPIRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MPIError(c_int);

impl MPIError {
    fn new(code: c_int) -> Result<(), MPIError> {
        if code == MPI_SUCCESS as i32 {
            Ok(())
        } else {
            Err(Self(code))
        }
    }

    fn code(&self) -> c_int {
        self.0
    }

    fn message(&self) -> String {
        let mut buffer = Vec::<u8>::with_capacity(MPI_MAX_ERROR_STRING as usize);
        let mut n: c_int = 0;

        unsafe {
            if MPI_Error_string(self.0, buffer.as_mut_ptr() as *mut i8, &mut n)
                == MPI_SUCCESS as i32
            {
                buffer.set_len(n as usize);
                String::from_utf8(buffer).unwrap()
            } else {
                format!("error code {}", self.code())
            }
        }
    }
}

impl StdError for MPIError {}

impl Display for MPIError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MPI error: {}", self.message())
    }
}

#[inline(always)]
fn mpi_check(code: c_int) -> Result<(), MPIError> {
    MPIError::new(code)
}

pub trait NetworkHandler: Send + 'static {
    type Token: Send;
    fn handle_message(&mut self, source: MPIRank, message: &[u8]);
    fn transfer_finished(&mut self, token: Self::Token, result: Result<(), NetworkError>);
    fn probe_finished(&mut self, token: Self::Token, result: Result<(MPIRank, Tag), NetworkError>);
}

#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum NetworkError {
    #[error("{0}")]
    MPI(#[from] MPIError),

    #[error("support for MPI_THREAD_FUNNELED or higher is required")]
    InvalidThreading,

    #[error("node {0} does not exist")]
    InvalidRank(MPIRank),

    #[error("tag {0} is not a valid MPI tag")]
    InvalidTag(Tag),

    #[error("message of size {0} exceeds maximum message size")]
    MessageToLarge(usize),

    #[error("connection was closed")]
    Disconnected,
}

pub struct NetworkErrorWithToken<T> {
    pub token: T,
    pub error: NetworkError,
}

pub struct Network<Token> {
    handle: NetworkHandle<Token>,
    thread_handle: Option<JoinHandle<()>>,
}

pub struct NetworkHandle<Token> {
    my_id: MPIRank,
    num_nodes: MPIRank,
    max_tag: Tag,
    queue: Sender<Command<Token>>,
}

// Manual impl since Token: ?Clone
impl<Token> Clone for NetworkHandle<Token> {
    fn clone(&self) -> Self {
        Self {
            my_id: self.my_id,
            num_nodes: self.num_nodes,
            max_tag: self.max_tag,
            queue: self.queue.clone(),
        }
    }
}

// Manual impl since Token: ?Debug
impl<Token> Debug for NetworkHandle<Token> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Network")
            .field("my_id", &self.my_id)
            .field("num_nodes", &self.num_nodes)
            .field("max_tag", &self.max_tag)
            .field("queue", &self.queue)
            .finish()
    }
}

impl<Token> Network<Token> {
    pub fn new<H: NetworkHandler<Token = Token>>(handler: H) -> Result<Self, NetworkError> {
        launch_network_thread(handler)
    }

    pub fn handle(&self) -> NetworkHandle<Token> {
        self.handle.clone()
    }
}

impl<Token> Drop for Network<Token> {
    fn drop(&mut self) {
        let _ = self.handle.queue.send(Command::Shutdown);
        let _ = self.thread_handle.take().unwrap().join();
    }
}

impl<Token> NetworkHandle<Token> {
    fn check_valid_node(&self, peer: MPIRank) -> Result<(), NetworkError> {
        if (0..self.num_nodes.0).contains(&peer.0) {
            Ok(())
        } else {
            Err(NetworkError::InvalidRank(peer))
        }
    }

    fn check_tag_and_size(&self, tag: Tag, size: usize) -> Result<(), NetworkError> {
        if tag.0 >= self.max_tag.0 {
            Err(NetworkError::InvalidTag(tag))
        } else if TryInto::<c_int>::try_into(size).is_err() {
            Err(NetworkError::MessageToLarge(size))
        } else {
            Ok(())
        }
    }

    pub fn my_rank(&self) -> MPIRank {
        self.my_id
    }

    pub fn num_ranks(&self) -> MPIRank {
        self.num_nodes
    }

    pub fn max_tag(&self) -> Tag {
        self.max_tag
    }

    pub unsafe fn send_async(
        &self,
        dst: MPIRank,
        send_buffer: *const u8,
        size: usize,
        tag: Tag,
        token: Token,
    ) -> Result<(), NetworkErrorWithToken<Token>> {
        if let Err(error) = Result::and(
            self.check_valid_node(dst),
            self.check_tag_and_size(tag, size),
        ) {
            return Err(NetworkErrorWithToken { error, token });
        }

        self.queue
            .send(Command::TransferTo {
                dst,
                buffer: send_buffer,
                size,
                tag,
                token,
            })
            .map_err(|e| {
                let token = match e.into_inner() {
                    Command::TransferTo { token, .. } => token,
                    _ => unreachable!(),
                };

                NetworkErrorWithToken {
                    token,
                    error: NetworkError::Disconnected,
                }
            })
    }

    pub unsafe fn recv_async(
        &self,
        src: MPIRank,
        recv_buffer: *mut u8,
        size: usize,
        tag: Tag,
        token: Token,
    ) -> Result<(), NetworkErrorWithToken<Token>> {
        if let Err(error) = Result::and(
            self.check_valid_node(src),
            self.check_tag_and_size(tag, size),
        ) {
            return Err(NetworkErrorWithToken { error, token });
        }

        self.queue
            .send(Command::TransferFrom {
                src,
                buffer: recv_buffer,
                size,
                tag,
                token,
            })
            .map_err(|e| {
                let token = match e.into_inner() {
                    Command::TransferFrom { token, .. } => token,
                    _ => unreachable!(),
                };

                NetworkErrorWithToken {
                    token,
                    error: NetworkError::Disconnected,
                }
            })
    }

    pub fn probe_async(
        &self,
        src: Option<MPIRank>,
        tag: Option<Tag>,
        token: Token,
    ) -> Result<(), NetworkErrorWithToken<Token>> {
        if let Some(src) = src {
            if let Err(error) = self.check_valid_node(src) {
                return Err(NetworkErrorWithToken { error, token });
            }
        }

        if let Some(tag) = tag {
            if let Err(error) = self.check_tag_and_size(tag, 0) {
                return Err(NetworkErrorWithToken { error, token });
            }
        }

        self.queue
            .send(Command::Probe { src, tag, token })
            .map_err(|e| {
                let token = match e.into_inner() {
                    Command::Probe { token, .. } => token,
                    _ => unreachable!(),
                };

                NetworkErrorWithToken {
                    token,
                    error: NetworkError::Disconnected,
                }
            })
    }

    pub fn message(&self, dst: MPIRank, payload: Vec<u8>) -> Result<(), NetworkError> {
        self.check_valid_node(dst)?;

        self.queue
            .send(Command::Rpc { dst, payload })
            .map_err(|_| NetworkError::Disconnected)
    }
}

// Command send between main thread and network thread
pub enum Command<Token> {
    TransferTo {
        dst: MPIRank,
        buffer: *const u8,
        size: usize,
        tag: Tag,
        token: Token,
    },
    TransferFrom {
        src: MPIRank,
        buffer: *mut u8,
        size: usize,
        tag: Tag,
        token: Token,
    },
    Probe {
        src: Option<MPIRank>,
        tag: Option<Tag>,
        token: Token,
    },
    Rpc {
        dst: MPIRank,
        payload: Vec<u8>,
    },
    Shutdown,
}

unsafe impl<T: Send> Send for Command<T> {}

enum Completion<T> {
    TransferFinished {
        peer: MPIRank,
        size: usize,
        tag: Tag,
        token: T,
    },
    SendRpc {
        dst: MPIRank,
        buffer: Buffer,
    },
    RecvRpc {
        seq: usize,
        src: MPIRank,
        buffer: Buffer,
    },
}

struct Probe<T> {
    src: MPIRank,
    tag: Tag,
    token: T,
}

const MEMORY_POOL_BUFFER_SIZE: usize = 256 * 1024;
const MEMORY_POOL_MAX_BUFFERS: usize = 64;
type Buffer = Vec<u8>;

struct State<H: NetworkHandler> {
    my_id: MPIRank,
    num_nodes: MPIRank,
    handler: H,
    has_shutdown: bool,
    queue: Receiver<Command<H::Token>>,
    communicator: MPI_Comm,

    mpi_requests: Vec<MPI_Request>,
    complete_actions: Vec<Completion<H::Token>>,
    temp_indices: Vec<c_int>,
    temp_statuses: Vec<MPI_Status>,

    pending_probes: Vec<Probe<H::Token>>,

    rpc_tag: Tag,
    memory_pool: Vec<Buffer>,
    received_rpc_seq: Box<[usize]>,
    processed_rpc_seq: Box<[usize]>,
    pending_incoming_rpcs: HashMap<(MPIRank, usize), Buffer>,
}

fn launch_network_thread<H>(handler: H) -> Result<Network<H::Token>, NetworkError>
where
    H: NetworkHandler,
{
    let (promise, future) = channel::bounded(0);
    let (sender, receiver) = channel::unbounded();

    let handle = thread::Builder::new()
        .name("network".to_string())
        .spawn(move || {
            let state = match State::new(handler, receiver) {
                Ok(state) => state,
                Err(e) => return promise.send(Err(e)).unwrap(),
            };

            let result = (state.my_id, state.num_nodes, state.rpc_tag);
            promise.send(Ok(result)).unwrap();

            state.run_forever();
        })
        .expect("failed to launch new thread");

    let (my_id, num_nodes, max_tag) = future.recv().unwrap()?;
    Ok(Network {
        handle: NetworkHandle {
            my_id,
            num_nodes,
            max_tag,
            queue: sender,
        },
        thread_handle: Some(handle),
    })
}

impl<H> State<H>
where
    H: NetworkHandler,
{
    fn new(handler: H, queue: Receiver<Command<H::Token>>) -> Result<Self, NetworkError> {
        // Check threading level provided by MPI_Init_thread
        let required = MPI_THREAD_FUNNELED as c_int;
        let mut provided: u32 = !0;

        unsafe {
            mpi_check(MPI_Init_thread(
                &mut 0,
                &mut ptr::null_mut(),
                required,
                &mut provided as *mut u32 as *mut i32,
            ))?;
        }

        // Check if provided level supports threads in any way.
        trace!("MPI_Init_thread returned level {}", provided);
        if ![
            MPI_THREAD_FUNNELED,
            MPI_THREAD_SERIALIZED,
            MPI_THREAD_MULTIPLE,
        ]
        .contains(&provided)
        {
            return Err(NetworkError::InvalidThreading);
        }

        // Perform barrier to make sure we are all on the same page.
        let communicator = unsafe { RSMPI_COMM_WORLD };
        unsafe {
            mpi_check(MPI_Barrier(communicator))?;
        }

        // Get ranks and number of  ranks
        let mut rank: c_int = 0;
        let mut num_ranks: c_int = 0;
        unsafe {
            mpi_check(MPI_Comm_rank(communicator, &mut rank))?;
            mpi_check(MPI_Comm_size(communicator, &mut num_ranks))?;
        }

        // Get MPI_TAG_UB attribute
        let mut flag: c_int = 0;
        let mut max_tag_ptr: *const c_int = ptr::null();
        let max_tag: c_int;
        unsafe {
            mpi_check(MPI_Comm_get_attr(
                communicator,
                MPI_TAG_UB as _,
                &mut max_tag_ptr as *mut *const c_int as *mut c_void,
                &mut flag,
            ))?;
            assert_ne!(flag, 0);

            max_tag = *max_tag_ptr;
        }

        let rank = rank as usize;
        let num_ranks = num_ranks as usize;

        trace!("network initialized");
        Ok(Self {
            my_id: MPIRank(rank),
            num_nodes: MPIRank(num_ranks),
            handler,
            queue,
            communicator,
            has_shutdown: false,
            temp_indices: default(),
            temp_statuses: default(),
            mpi_requests: default(),
            complete_actions: default(),
            rpc_tag: Tag(max_tag),
            memory_pool: vec![],
            received_rpc_seq: new_boxed_slice(num_ranks, || 0),
            processed_rpc_seq: new_boxed_slice(num_ranks, || 0),
            pending_incoming_rpcs: default(),
            pending_probes: default(),
        })
    }

    fn run_forever(mut self) {
        const TIMEOUT: Duration = Duration::from_micros(10);

        while !self.has_shutdown {
            let before = Instant::now();

            self.poll_rpc();
            self.poll_probes();
            self.poll_requests();
            self.poll_queue();

            if let Some(sleep_time) = TIMEOUT.checked_sub(before.elapsed()) {
                thread::sleep(sleep_time);
            }
        }

        self.shutdown()
    }

    fn shutdown(mut self) {
        trace!("preparing to shutdown network");

        // We need handle requests which are still progress. We cancel any data transfers and
        // wait until completion on RPC calls since there might still be "shutdown" msgs in progress.
        for (mut request, completion) in zip(
            take(&mut self.mpi_requests),
            take(&mut self.complete_actions),
        ) {
            let result = match &completion {
                Completion::TransferFinished { .. } => {
                    unsafe {
                        MPI_Cancel(&mut request);
                    }

                    Err(NetworkError::Disconnected)
                }
                Completion::SendRpc { .. } | Completion::RecvRpc { .. } => {
                    let status = unsafe {
                        let mut status = MaybeUninit::<MPI_Status>::uninit();
                        MPI_Wait(&mut request, status.as_mut_ptr());
                        status.assume_init()
                    };

                    match MPIError::new(status.MPI_ERROR) {
                        Ok(()) => Ok(status),
                        Err(e) => Err(NetworkError::MPI(e)),
                    }
                }
            };

            self.handle_completion(completion, result);
        }

        unsafe {
            MPI_Barrier(self.communicator);
        }
        trace!("call MPI_Finalize");
        unsafe { MPI_Finalize() };
    }

    fn poll_rpc(&mut self) -> bool {
        loop {
            let mut flag: c_int = 0;
            let mut status = MaybeUninit::<MPI_Status>::zeroed();

            unsafe {
                MPI_Iprobe(
                    RSMPI_ANY_SOURCE,
                    self.rpc_tag.0,
                    self.communicator,
                    &mut flag,
                    status.as_mut_ptr(),
                );
            }

            if flag == 0 {
                break false;
            }

            let status = unsafe { status.assume_init() };
            let src = MPIRank(status.MPI_SOURCE as usize);

            let count = unsafe {
                let mut n: c_int = 0;
                MPI_Get_count(&status, RSMPI_UINT8_T, &mut n);
                n as usize
            };

            let seq = self.received_rpc_seq[src.0];
            self.received_rpc_seq[src.0] += 1;

            trace!(
                "expecting message of {} bytes from node {} (seq: {})",
                count,
                src,
                seq
            );
            let mut buffer = self.allocate_buffer(count);
            let communicator = self.communicator;

            unsafe {
                self.register_request(move |request| {
                    MPI_Irecv(
                        buffer.as_mut_ptr() as *mut c_void,
                        buffer.len() as c_int,
                        RSMPI_UINT8_T,
                        status.MPI_SOURCE,
                        status.MPI_TAG,
                        communicator,
                        request,
                    );

                    Completion::RecvRpc { seq, src, buffer }
                });
            }
        }
    }

    fn poll_probes(&mut self) -> bool {
        let mut index = 0;
        let mut len = self.pending_probes.len();

        while index < len {
            let probe = &self.pending_probes[index];
            let tag = probe.tag;
            let src = probe.src;

            let mut flag: c_int = 0;
            let mut status = MaybeUninit::<MPI_Status>::zeroed();

            unsafe {
                mpi_check(MPI_Iprobe(
                    src.0 as i32,
                    tag.0,
                    self.communicator,
                    &mut flag,
                    status.as_mut_ptr(),
                ))
                .unwrap();
            }

            if flag == 0 {
                index += 1;
                continue;
            }

            let probe = self.pending_probes.swap_remove(index);
            len -= 1;

            let status = unsafe { status.assume_init() };
            let result = if let Err(e) = MPIError::new(status.MPI_ERROR) {
                Err(NetworkError::MPI(e))
            } else {
                let src = MPIRank(status.MPI_SOURCE as usize);
                let tag = Tag(status.MPI_TAG);
                Ok((src, tag))
            };

            trace!("complete probe tag {} from {}", tag, src);
            self.handler.probe_finished(probe.token, result);
        }

        true
    }

    fn poll_requests(&mut self) -> bool {
        let n = self.mpi_requests.len();
        if n == 0 {
            return false;
        }

        let mut indices = take(&mut self.temp_indices);
        let mut statuses = take(&mut self.temp_statuses);

        // reserve additional memory.
        if let Some(add) = usize::checked_sub(n, indices.len()) {
            indices.reserve(add);
            statuses.reserve(add);
        }

        unsafe {
            let mut num_completed: c_int = 0;
            mpi_check(MPI_Testsome(
                self.mpi_requests.len() as c_int,
                self.mpi_requests.as_mut_ptr(),
                &mut num_completed,
                indices.as_mut_ptr(),
                statuses.as_mut_ptr(),
            ))
            .unwrap();

            // Only first few entries are actually initialized by MPI_Testsome.
            indices.set_len(num_completed as usize);
            statuses.set_len(num_completed as usize);
        }

        // In practice the indices will be be sorted, but the MPI standard does not require this
        // of Testsome. Here is a simple bubble sort to sort indices if this is not the case.
        let m = indices.len();
        let mut changed = true;
        while changed {
            changed = false;

            for i in 1..m {
                if indices[i - 1] > indices[i] {
                    indices.swap(i, i - 1);
                    statuses.swap(i, i - 1);
                    changed = true;
                }
            }
        }

        // Iterate over indices in reverse order (from high to low), this allows us to use
        // swap_remove to remove the request without shifting all other elements.
        for (&i, &status) in reversed(zip(&indices, &statuses)) {
            self.mpi_requests.swap_remove(i as usize);
            let c = self.complete_actions.swap_remove(i as usize);

            // TODO: One might expect that status.MPI_ERROR contains the error status of the request.
            // However, one might be wrong. The field MPI_ERROR seems to be untouched by MPI and
            // contain junk data. Maybe it has a different purpose? Who knows?
            //let result = match MPIError::new(status.MPI_ERROR) {
            //    Ok(()) => Ok(status),
            //    Err(e) => Err(NetworkError::MPI(e)),
            //};
            let result = Ok(status);

            self.handle_completion(c, result);
        }

        assert!(self.temp_indices.is_empty());
        assert!(self.temp_statuses.is_empty());
        self.temp_indices = indices;
        self.temp_statuses = statuses;

        true
    }

    fn handle_completion(
        &mut self,
        c: Completion<H::Token>,
        result: Result<MPI_Status, NetworkError>,
    ) {
        match c {
            Completion::TransferFinished {
                token,
                size,
                peer,
                tag,
                ..
            } => {
                trace!(
                    "finished transfer of {} bytes with node {} (tag: {})",
                    size,
                    peer,
                    tag
                );
                self.handler.transfer_finished(token, result.map(|_| ()));
            }

            Completion::SendRpc { dst, buffer } => {
                trace!("finished sending {} bytes to node {}", buffer.len(), dst);
                self.release_buffer(buffer);
            }

            Completion::RecvRpc { src, buffer, seq } => {
                trace!(
                    "finished receiving message of {} bytes from node {} (seq: {})",
                    buffer.len(),
                    src,
                    seq
                );
                self.handle_incoming_rpc(src, seq, buffer);
            }
        }
    }

    fn handle_incoming_rpc(&mut self, src: MPIRank, mut seq: usize, mut buffer: Buffer) {
        if self.processed_rpc_seq[src.0] == seq {
            loop {
                trace!("processing message from {} (seq: {})", src, seq);

                seq += 1;
                self.processed_rpc_seq[src.0] = seq;
                self.handler.handle_message(src, &buffer);

                self.release_buffer(buffer);

                if let Some(buf) = self.pending_incoming_rpcs.remove(&(src, seq)) {
                    buffer = buf;
                } else {
                    break;
                }
            }
        } else {
            trace!("received message out-of-order from {} (seq: {})", src, seq);
            self.pending_incoming_rpcs.insert((src, seq), buffer);
        }
    }

    unsafe fn register_request<F>(&mut self, fun: F)
    where
        F: FnOnce(*mut MPI_Request) -> Completion<H::Token>,
    {
        let mut request = MaybeUninit::<MPI_Request>::uninit();
        let completion = fun(request.as_mut_ptr());

        // Some requests complete immediately (for instance, if the buffer was small enough
        // that it was copied into the OS). Test now for completion and immediately trigger the
        // callback if the request has completed.
        let mut flag: c_int = 0;
        let mut status = MaybeUninit::<MPI_Status>::uninit();
        MPI_Test(request.as_mut_ptr(), &mut flag, status.as_mut_ptr());

        if flag != 0 {
            let status = status.assume_init();
            // // Ignore status for now, seems to be faulty?
            //let result = match MPIError::new(status.MPI_ERROR) {
            //    Ok(()) => Ok(status),
            //    Err(e) => Err(NetworkError::MPI(e)),
            //};
            let result = Ok(status);

            self.handle_completion(completion, result);
        } else {
            self.mpi_requests.push(request.assume_init());
            self.complete_actions.push(completion);
        }
    }

    fn poll_queue(&mut self) -> bool {
        loop {
            match self.queue.try_recv() {
                Ok(cmd) => {
                    self.handle_command(cmd);
                }
                Err(TryRecvError::Disconnected) => {
                    self.has_shutdown = true;
                    break false;
                }
                Err(TryRecvError::Empty) => {
                    break false;
                }
            }
        }
    }

    fn handle_command(&mut self, command: Command<H::Token>) {
        match command {
            Command::TransferTo {
                token,
                dst,
                buffer,
                size,
                tag,
            } => {
                let comm = self.communicator;
                trace!(
                    "start transfer of {} bytes to node {} (tag: {})",
                    size,
                    dst,
                    tag
                );

                unsafe {
                    self.register_request(|request| {
                        mpi_check(MPI_Isend(
                            buffer as *mut c_void,
                            size as _,
                            RSMPI_UINT8_T,
                            dst.0 as _,
                            tag.0,
                            comm,
                            request,
                        ))
                        .unwrap();

                        Completion::TransferFinished {
                            peer: dst,
                            size,
                            tag,
                            token,
                        }
                    });
                }
            }

            Command::TransferFrom {
                token,
                src,
                buffer,
                size,
                tag,
            } => {
                let comm = self.communicator;
                trace!(
                    "start transfer of {} bytes from node {} (tag: {})",
                    size,
                    src,
                    tag
                );

                unsafe {
                    self.register_request(|request| {
                        mpi_check(MPI_Irecv(
                            buffer as *mut c_void,
                            size as _,
                            RSMPI_UINT8_T,
                            src.0 as _,
                            tag.0,
                            comm,
                            request,
                        ))
                        .unwrap();

                        Completion::TransferFinished {
                            peer: src,
                            size,
                            tag,
                            token,
                        }
                    });
                }
            }

            Command::Probe { src, tag, token } => {
                let src = src.unwrap_or(MPIRank(MPI_ANY_SOURCE as _));
                let tag = tag.unwrap_or(Tag(MPI_ANY_TAG as _));

                trace!("start probe from node {} (tag: {})", tag, src);
                self.pending_probes.push(Probe { src, tag, token });
            }

            Command::Rpc { dst, payload } => {
                // Avoid MPI entirely for loopback messages.
                if dst == self.my_id {
                    self.handler.handle_message(dst, &payload);
                    return;
                }

                let comm = self.communicator;
                let tag = self.rpc_tag.0;
                trace!(
                    "start sending message of {} bytes to node {}",
                    payload.len(),
                    dst
                );

                unsafe {
                    self.register_request(move |request| {
                        mpi_check(MPI_Isend(
                            payload.as_ptr() as *const c_void,
                            payload.len() as c_int,
                            RSMPI_UINT8_T,
                            dst.0 as _,
                            tag,
                            comm,
                            request,
                        ))
                        .unwrap();

                        Completion::SendRpc {
                            dst,
                            buffer: payload,
                        }
                    });
                }
            }

            Command::Shutdown => {
                self.has_shutdown = true;
            }
        }
    }

    fn allocate_buffer(&mut self, len: usize) -> Buffer {
        // Buffer too large, allocate new buffer.
        if len > MEMORY_POOL_BUFFER_SIZE {
            return vec![0; len];
        }

        // Take buffer from the memory pool
        let mut buffer = match self.memory_pool.pop() {
            Some(buf) => buf,
            None => Vec::with_capacity(MEMORY_POOL_BUFFER_SIZE),
        };

        unsafe {
            buffer.set_len(len);
        }

        buffer
    }

    fn release_buffer(&mut self, buffer: Buffer) {
        // Only insert buffer into pool if its size is MEMORY_POOL_BUFFER_SIZE and there
        // are at most MEMORY_POOL_MAX_BUFFERS entries already in the pool.
        if buffer.capacity() == MEMORY_POOL_BUFFER_SIZE
            && self.memory_pool.len() < MEMORY_POOL_MAX_BUFFERS
        {
            self.memory_pool.push(buffer);
        }
    }
}
