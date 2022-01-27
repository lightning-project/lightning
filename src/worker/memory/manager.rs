use super::{DeviceMemoryPool, HostMemoryPool, Storage, StorageError, StorageId};
use crate::prelude::*;
use crate::types::{ChunkLayout, DeviceId, GenericAccessor, MemoryKind, UnifiedPtr, MAX_DEVICES};
use crate::worker::memory::copy_engine::CopyEngine;
use by_address::ByAddress;
use crossbeam::channel::Sender;
use lightning_core::util::{TCell, TCellOwner};
use lightning_cuda::prelude::*;
use lru::LruCache;
use rc_borrow::ArcBorrow;
use std::any::Any;
use std::fmt::{self, Debug};
use std::num::NonZeroU32;
use std::ptr;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::task::Poll::{self, Pending, Ready};

struct LockMarker;
type Lock = TCellOwner<LockMarker>;

#[derive(Debug)]
pub(crate) enum Event {
    MarkValidFile {
        chunk: ChunkHandle,
        result: Result<StorageId, StorageError>,
    },
    MarkValidHost {
        chunk: ChunkHandle,
        result: Result,
    },
    MarkValidDevice {
        chunk: ChunkHandle,
        device_id: DeviceId,
        result: CudaResult,
    },
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
enum EntryStatus {
    Invalid,
    Valid,
}

impl Default for EntryStatus {
    fn default() -> Self {
        Self::Invalid
    }
}

#[derive(Debug)]
struct Entry<T> {
    status: EntryStatus,
    data: Option<T>,
    locks: usize,
}

unsafe impl Send for Entry<HostPtr> {}
unsafe impl Sync for Entry<HostPtr> {}

impl<T> Default for Entry<T> {
    fn default() -> Self {
        Self {
            status: default(),
            data: None,
            locks: 0,
        }
    }
}

impl<T> Entry<T> {
    fn is_valid(&self) -> bool {
        matches!(&self.status, EntryStatus::Valid)
    }
}

#[derive(Debug)]
enum FileEntry {
    Valid(StorageId),
    Error(StorageError),
    Invalid,
}

impl FileEntry {
    fn is_valid(&self) -> bool {
        matches!(self, FileEntry::Valid(_))
    }
}

impl Default for FileEntry {
    fn default() -> Self {
        FileEntry::Invalid
    }
}

type HostPtr = NonNull<u8>;

#[derive(Debug)]
struct DataTransfer {
    src: MemoryKind,
    requests: RequestList,
}

impl DataTransfer {
    fn add_request(&mut self, request: RequestRef) {
        self.requests.push_back(request);
    }
}

#[derive(Default, Debug)]
struct DataTransferList {
    to_file: Option<DataTransfer>,
    to_host: Option<DataTransfer>,
    to_devices: [Option<DataTransfer>; MAX_DEVICES],
}

impl DataTransferList {
    fn find_by_src_or_dst(&mut self, mem: MemoryKind) -> Option<&mut DataTransfer> {
        if self.to_file.as_ref().map_or(false, |t| t.src == mem) {
            return self.to_file.as_mut();
        }

        if self.to_host.as_ref().map_or(false, |t| t.src == mem) {
            return self.to_host.as_mut();
        }

        if let Some(i) = self.to_devices.iter().flatten().position(|t| t.src == mem) {
            return self.to_devices[i].as_mut();
        }

        self.find_by_dst(mem)
    }

    fn find_by_dst(&mut self, dst: MemoryKind) -> Option<&mut DataTransfer> {
        match dst {
            MemoryKind::FileSystem => self.to_file.as_mut(),
            MemoryKind::Host => self.to_host.as_mut(),
            MemoryKind::Device(i) => self.to_devices[i.get()].as_mut(),
        }
    }

    fn find_any(&mut self) -> Option<&mut DataTransfer> {
        if let Some(transfer) = &mut self.to_file {
            return Some(transfer);
        }

        if let Some(transfer) = &mut self.to_host {
            return Some(transfer);
        }

        for device in &mut self.to_devices {
            if let Some(transfer) = device {
                return Some(transfer);
            }
        }

        None
    }

    fn insert(&mut self, src: MemoryKind, dst: MemoryKind) -> &mut DataTransfer {
        let t = match dst {
            MemoryKind::FileSystem => &mut self.to_file,
            MemoryKind::Host => &mut self.to_host,
            MemoryKind::Device(i) => &mut self.to_devices[i.get()],
        };

        assert!(t.is_none());

        *t = Some(DataTransfer {
            src,
            requests: RequestList::new(),
        });

        t.as_mut().unwrap()
    }

    #[inline]
    fn remove_by_dst(&mut self, dst: MemoryKind) -> RequestList {
        match dst {
            MemoryKind::FileSystem => &mut self.to_file,
            MemoryKind::Host => &mut self.to_host,
            MemoryKind::Device(i) => &mut self.to_devices[i.get()],
        }
        .take()
        .unwrap()
        .requests
    }
}

pub(crate) struct ChunkMeta {
    layout: ChunkLayout,
    size_in_bytes: usize,
    state: TCell<LockMarker, ChunkState>,
}

impl ChunkMeta {
    pub(crate) fn size_in_bytes(&self) -> usize {
        self.size_in_bytes
    }
}

impl Debug for ChunkMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChunkMeta")
            .field("layout", &self.layout)
            .field("size_in_bytes", &self.size_in_bytes)
            .field("state", &"...")
            .finish()
    }
}

#[derive(Debug)]
struct ChunkState {
    status: ChunkStatus,
    active_transfers: DataTransferList,
    file_entry: FileEntry,
    host_entry: Entry<HostPtr>,
    device_entries: [Entry<CudaDevicePtr>; MAX_DEVICES],
    waiters: RequestList,
    refcount: usize,
    deletion_planned: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ChunkStatus {
    Available,
    Readers(NonZeroU32),
    Writer,
    Deleted,
}

impl ChunkState {
    fn is_valid_anywhere(&self) -> bool {
        self.host_entry.is_valid()
            || self.file_entry.is_valid()
            || any(&self.device_entries, |e| e.is_valid())
    }

    fn is_valid_at(&self, place: MemoryKind) -> bool {
        match place {
            MemoryKind::FileSystem => self.file_entry.is_valid(),
            MemoryKind::Host => self.host_entry.is_valid(),
            MemoryKind::Device(i) => self.device_entries[i.get()].is_valid(),
        }
    }

    fn is_valid_only_at(&self, place: MemoryKind) -> bool {
        if (place == MemoryKind::Host) ^ self.host_entry.is_valid() {
            return false;
        }

        if (place == MemoryKind::FileSystem) ^ self.file_entry.is_valid() {
            return false;
        }

        for (index, entry) in enumerate(&self.device_entries) {
            if (place == MemoryKind::Device(DeviceId::new(index))) ^ entry.is_valid() {
                return false;
            }
        }

        true
    }

    fn is_unlocked(&self) -> bool {
        matches!(self.status, ChunkStatus::Available)
    }

    fn poll_lock(&mut self, request: RequestRef) -> Poll<()> {
        match (self.status, request.exclusive) {
            (ChunkStatus::Available, _) => Ready(()),
            (ChunkStatus::Readers(_), false) => Ready(()),
            _ => {
                self.waiters.try_push_back(request);
                Pending
            }
        }
    }

    fn lock(&mut self, request: RequestRef) {
        self.status = match (self.status, request.exclusive) {
            (ChunkStatus::Available, true) => ChunkStatus::Writer,
            (ChunkStatus::Available, false) => ChunkStatus::Readers(NonZeroU32::new(1).unwrap()),
            (ChunkStatus::Readers(n), false) => {
                ChunkStatus::Readers(NonZeroU32::new(n.get() + 1).unwrap())
            }
            l => panic!("invalid status, cannot lock {:?}", l),
        }
    }

    fn unlock(&mut self, request: RequestRef, queue: &mut RequestList) {
        self.status = match (self.status, request.exclusive) {
            (ChunkStatus::Writer, true) => ChunkStatus::Available,
            (ChunkStatus::Readers(n), false) => {
                if let Some(new_n) = NonZeroU32::new(n.get() - 1) {
                    ChunkStatus::Readers(new_n)
                } else {
                    ChunkStatus::Available
                }
            }
            l => panic!("invalid status, cannot unlock {:?}", l),
        };

        queue.extend(&mut self.waiters);
    }

    fn flag_for_deletion(&mut self) {
        self.deletion_planned = true;
    }
}

#[derive(Error, Debug)]
pub(crate) enum AllocationError {
    #[error("transaction failed, insufficient host memory to satisfy request")]
    HostMemoryExhausted,

    #[error("transaction failed, insufficient device memory to satisfy request")]
    DeviceMemoryExhausted,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
struct TransactionId(usize);

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
enum RequestStatus {
    DependenciesPending(usize),
    WaitingForLock,
    WaitingForSubmission,
    Submitted,
    AllocateHost,
    AllocateDevice(DeviceId),
    WaitingForData,
    Active,
    Terminated,
}

pub(crate) struct Request {
    transaction_id: TransactionId,
    chunk: ChunkHandle,
    exclusive: bool,
    parent: RequestParent,
    next_in_queue: RequestLink,
    state: TCell<LockMarker, RequestState>,
}

impl Debug for Request {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Request")
            .field("transaction_id", &self.transaction_id)
            .field("chunk", &self.chunk)
            .field("exclusive", &self.exclusive)
            .field("state", &"...")
            .finish()
    }
}

struct RequestState {
    status: RequestStatus,
    place: Option<MemoryKind>,
}

impl Request {
    #[inline]
    pub(crate) fn parent(&self) -> &RequestParent {
        &self.parent
    }

    #[inline]
    pub(crate) fn chunk(&self) -> &ChunkHandle {
        &self.chunk
    }

    #[inline]
    fn status(&self, token: &Lock) -> RequestStatus {
        self.state.borrow(token).status
    }

    #[inline]
    fn set_status(&self, new_status: RequestStatus, token: &mut Lock) {
        /*trace!("request {:?}: {:?} -> {:?}", &self, self.status(token), new_status);*/
        self.state.borrow_mut(token).status = new_status;
    }

    pub(crate) fn get(&self, manager: &Manager) -> GenericAccessor {
        let state = self.state.borrow(&manager.token);
        assert_eq!(state.status, RequestStatus::Active);
        let place = state.place;

        let chunk = &self.chunk;
        let state = chunk.state.borrow(&manager.token);

        let ptr = match place {
            Some(MemoryKind::Host) => {
                let ptr = state.host_entry.data.unwrap().as_ptr();
                match self.exclusive {
                    true => UnifiedPtr::HostMut(ptr),
                    false => UnifiedPtr::Host(ptr),
                }
            }
            Some(MemoryKind::Device(i)) => {
                let ptr = state.device_entries[i.get()].data.unwrap();
                match self.exclusive {
                    true => UnifiedPtr::DeviceMut(ptr, i),
                    false => UnifiedPtr::Device(ptr, i),
                }
            }
            _ => unreachable!(),
        };

        let layout = &chunk.layout;
        unsafe { GenericAccessor::new(ptr, layout.strides, layout.size, layout.data_type) }
    }
}

pub(crate) type RequestHandle = Arc<Request>;
type RequestRef<'a> = ArcBorrow<'a, Request>;

pub(crate) type RequestParent = Weak<dyn Any + Send + Sync>;

pub(crate) enum RequestEvent {
    Ready,
    Active,
    Abort(anyhow::Error),
}

#[derive(Debug)]
struct RequestList {
    head: Option<RequestHandle>,
    tail: *const Request,
}

unsafe impl Send for RequestList {}
unsafe impl Sync for RequestList {}

impl RequestList {
    fn new() -> Self {
        Self {
            head: None,
            tail: ptr::null(),
        }
    }

    fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    fn push_back(&mut self, handle: RequestRef) {
        assert!(self.try_push_back(handle));
    }

    fn try_push_back(&mut self, handle: RequestRef) -> bool {
        if !handle.next_in_queue.try_acquire() {
            return false;
        }

        let old_tail = replace(&mut self.tail, &handle as &Request as *const Request);
        let handle = RequestRef::upgrade(handle);

        if old_tail.is_null() {
            self.head = Some(handle);
        } else {
            unsafe { (&*old_tail).next_in_queue.set(handle) };
        }

        true
    }

    fn pop_front(&mut self) -> Option<RequestHandle> {
        if let Some(head) = &self.head {
            let next = head.next_in_queue.release();

            if next.is_none() {
                self.tail = ptr::null();
            };

            replace(&mut self.head, next)
        } else {
            None
        }
    }

    fn extend(&mut self, other: &mut RequestList) {
        let middle = if let Some(head) = take(&mut other.head) {
            head
        } else {
            return;
        };

        if self.tail.is_null() {
            self.head = Some(middle);
        } else {
            unsafe { (&*self.tail).next_in_queue.set(middle) };
        };

        self.tail = other.tail;
        other.tail = ptr::null();
    }
}

impl Drop for RequestList {
    fn drop(&mut self) {
        while let Some(_) = self.pop_front() {
            //
        }
    }
}

struct RequestLink {
    ptr: AtomicUsize,
}

const FREE_LINK: usize = 0;
const ACQUIRED_LINK: usize = 1;

impl RequestLink {
    fn new() -> Self {
        Self {
            ptr: AtomicUsize::new(FREE_LINK),
        }
    }

    fn try_acquire(&self) -> bool {
        self.ptr
            .compare_exchange(FREE_LINK, ACQUIRED_LINK, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    unsafe fn set(&self, request: RequestHandle) {
        let ptr = RequestHandle::into_raw(request) as usize;
        self.ptr.store(ptr, Ordering::SeqCst);
    }

    fn release(&self) -> Option<RequestHandle> {
        match self.ptr.swap(FREE_LINK, Ordering::SeqCst) {
            x if x == FREE_LINK => panic!("link not locked"),
            x if x == ACQUIRED_LINK => None,
            ptr => unsafe { Some(Arc::from_raw(ptr as *const Request)) },
        }
    }
}

pub(crate) type ChunkHandle = Arc<ChunkMeta>;
type ChunkRef<'a> = ArcBorrow<'a, ChunkMeta>;

struct MemoryContext<P> {
    memory: P,
    lru: LruCache<ByAddress<ChunkHandle>, ()>,
    pending_allocations: RequestList,
    active_allocation: Option<RequestHandle>,
    allocs_per_transaction: HashMap<TransactionId, usize>,
}

trait MemoryOps {
    type Ptr;
    fn allocate_entry(&mut self, chunk: ChunkRef, token: &mut Lock) -> bool;
    fn deallocate_entry(&mut self, chunk: ChunkRef, token: &mut Lock);
    fn entry<'a>(&mut self, chunk: ChunkRef<'a>, token: &'a mut Lock) -> &'a mut Entry<Self::Ptr>;
}

impl MemoryOps for HostMemoryPool {
    type Ptr = HostPtr;

    fn allocate_entry(&mut self, chunk: ChunkRef, token: &mut Lock) -> bool {
        let state = chunk.state.borrow_mut(token);
        let entry = &mut state.host_entry;

        if entry.data.is_none() {
            let size_in_bytes = chunk.size_in_bytes;
            let alignment = chunk.layout.alignment;

            if let Ok(ptr) = self.allocate(size_in_bytes, alignment) {
                entry.data = HostPtr::new(ptr);
            } else {
                return false;
            }
        }

        true
    }

    fn deallocate_entry(&mut self, chunk: ChunkRef, token: &mut Lock) {
        let entry = self.entry(chunk, token);

        if let Some(hptr) = entry.data.take() {
            self.deallocate(hptr.as_ptr(), chunk.size_in_bytes);
        }
    }

    fn entry<'a>(&mut self, chunk: ChunkRef<'a>, token: &'a mut Lock) -> &'a mut Entry<HostPtr> {
        let state = ChunkRef::downgrade(chunk).state.borrow_mut(token);
        &mut state.host_entry
    }
}

impl MemoryOps for (DeviceId, DeviceMemoryPool) {
    type Ptr = CudaDevicePtr;

    fn allocate_entry(&mut self, chunk: ChunkRef, token: &mut Lock) -> bool {
        let (device_id, memory) = self;
        let state = chunk.state.borrow_mut(token);
        let entry = &mut state.device_entries[device_id.get()];

        if entry.data.is_none() {
            let size_in_bytes = chunk.size_in_bytes;
            let alignment = chunk.layout.alignment;

            if let Ok(ptr) = memory.allocate(size_in_bytes, alignment) {
                entry.data = Some(ptr.cast());
            } else {
                return false;
            }
        }

        true
    }

    fn deallocate_entry(&mut self, chunk: ChunkRef, token: &mut Lock) {
        let entry = self.entry(chunk, token);

        if let Some(dptr) = entry.data.take() {
            let (_, memory) = self;
            memory.deallocate(dptr.cast(), chunk.size_in_bytes);
        };
    }

    fn entry<'a>(
        &mut self,
        chunk: ChunkRef<'a>,
        token: &'a mut Lock,
    ) -> &'a mut Entry<CudaDevicePtr> {
        let (device_id, _) = self;
        let state = ChunkRef::downgrade(chunk).state.borrow_mut(token);
        &mut state.device_entries[device_id.get()]
    }
}

impl<T: MemoryOps> MemoryContext<T> {
    fn new(inner: T) -> Self {
        Self {
            memory: inner,
            lru: LruCache::unbounded(),
            pending_allocations: RequestList::new(),
            active_allocation: None,
            allocs_per_transaction: default(),
        }
    }

    fn is_out_of_memory(&mut self, request: RequestRef) -> bool {
        let id = request.transaction_id;

        self.allocs_per_transaction.len() == 0
            || (self.allocs_per_transaction.len() == 1
                && self.allocs_per_transaction.contains_key(&id))
    }

    fn try_reserve_allocation(&mut self, request: RequestRef, token: &mut Lock) -> bool {
        let chunk = ChunkRef::from(&request.chunk);

        if !self.memory.allocate_entry(chunk, token) {
            return false;
        }

        self.reserve_allocation(request, token);
        true
    }

    fn reserve_allocation(&mut self, request: RequestRef, token: &mut Lock) {
        let chunk = ChunkRef::from(&request.chunk);
        let transaction = request.transaction_id;

        let entry = self.memory.entry(chunk, token);
        assert!(entry.data.is_some());

        if entry.locks == 0 {
            self.lru.pop(&ByAddress(ChunkRef::upgrade(chunk)));
        }
        entry.locks += 1;

        *self.allocs_per_transaction.entry(transaction).or_default() += 1;
    }

    fn unreserve_allocation(&mut self, request: RequestRef, token: &mut Lock) {
        let chunk = ChunkRef::from(&request.chunk);
        let transaction = request.transaction_id;

        use std::collections::hash_map::Entry::Occupied;
        if let Occupied(mut e) = self.allocs_per_transaction.entry(transaction) {
            if *e.get() > 1 {
                *e.get_mut() -= 1;
            } else {
                e.remove();
            }
        } else {
            panic!();
        }

        let entry = self.memory.entry(chunk, token);
        assert!(entry.data.is_some());
        entry.locks -= 1;

        if entry.locks == 0 {
            self.lru.put(ByAddress(ChunkRef::upgrade(chunk)), ());
        }
    }

    fn deallocate_entry(&mut self, chunk: ChunkRef, token: &mut Lock) {
        let entry = self.memory.entry(chunk, token);
        assert_eq!(entry.locks, 0);
        entry.status = EntryStatus::Invalid;

        self.memory.deallocate_entry(chunk, token);
        self.lru.pop(&ByAddress(ChunkRef::upgrade(chunk)));
    }
}

pub(crate) struct Manager {
    sender: Sender<Event>,
    chunks: HashSet<ByAddress<ChunkHandle>>,
    queue: RequestList,
    host: MemoryContext<HostMemoryPool>,
    devices: Vec<MemoryContext<(DeviceId, DeviceMemoryPool)>>,
    copy_engine: CopyEngine,
    storage: Option<Storage>,
    token: Lock,
}

impl Manager {
    pub(crate) unsafe fn new(
        sender: Sender<Event>,
        host: HostMemoryPool,
        devices: Vec<DeviceMemoryPool>,
        copy_engine: CopyEngine,
        storage: Option<Storage>,
    ) -> Self {
        let host = MemoryContext::new(host);
        let devices = devices
            .into_iter()
            .enumerate()
            .map(|(index, device)| MemoryContext::new((DeviceId::new(index), device)))
            .collect_vec();

        Self {
            sender,
            chunks: default(),
            queue: RequestList::new(),
            host,
            devices,
            copy_engine,
            storage,
            token: Lock::new(),
        }
    }

    pub(crate) fn create_chunk(&mut self, layout: ChunkLayout) -> ChunkHandle {
        let size_in_bytes = match layout.size_in_bytes() {
            Some(s) => s,
            None => {
                panic!("layout is not contiguous: {:?}", layout);
            }
        };

        let state = ChunkState {
            status: ChunkStatus::Available,
            active_transfers: default(),
            file_entry: default(),
            host_entry: default(),
            device_entries: default(),
            waiters: RequestList::new(),
            refcount: 0,
            deletion_planned: false,
        };

        let chunk = Arc::new(ChunkMeta {
            layout,
            size_in_bytes,
            state: TCell::new(state),
        });

        let by_address = ByAddress(ChunkHandle::clone(&chunk));
        let inserted = self.chunks.insert(by_address);
        assert_eq!(inserted, true);

        chunk
    }

    pub(crate) fn delete_chunk(&mut self, chunk: &ChunkHandle) {
        let state = chunk.state.borrow_mut(&mut self.token);
        state.flag_for_deletion();

        self.delete_chunk_when_idle(chunk);
    }

    pub(crate) fn is_idle(&mut self) -> bool {
        if !self.queue.is_empty() {
            return false;
        }

        for chunk in &self.chunks {
            let state = chunk.state.borrow_mut(&mut self.token);

            if let Some(_) = state.active_transfers.find_any() {
                return false;
            }
        }

        true
    }

    fn delete_chunk_when_idle(&mut self, chunk: &ChunkHandle) {
        use ChunkStatus::*;
        let chunk = ChunkRef::from(chunk);
        let state = chunk.state.borrow_mut(&mut self.token);

        if !state.deletion_planned || state.refcount != 0 || state.status == Deleted {
            return;
        }

        if let Some(_) = state.active_transfers.find_any() {
            eprintln!(
                "cannot release buffer {:#?} (state: {:#?}): transfers still in progress",
                chunk,
                chunk.state.borrow(&self.token),
            );
            return;
        }

        assert!(state.is_unlocked());

        let by_address = ByAddress(ChunkRef::upgrade(chunk));
        let found = self.chunks.remove(&by_address);
        assert!(found);

        let state = chunk.state.borrow_mut(&mut self.token);

        // Make deletion permanent
        state.status = Deleted;

        // Delete file
        if let FileEntry::Valid(id) = take(&mut state.file_entry) {
            self.storage.as_ref().unwrap().delete_async(id);
        }

        self.host.deallocate_entry(chunk, &mut self.token);
        self.make_progress_host_allocations();

        for index in 0..self.devices.len() {
            let device = &mut self.devices[index];
            device.deallocate_entry(chunk, &mut self.token);

            self.make_progress_device_allocations(DeviceId::new(index));
        }
    }

    pub(crate) fn create_request(
        &mut self,
        chunk: &ChunkHandle,
        parent: RequestParent,
        exclusive: bool,
        num_dependencies: usize,
    ) -> RequestHandle {
        let transaction_id = TransactionId(parent.as_ptr() as *const () as usize);

        let state = chunk.state.borrow_mut(&mut self.token);
        assert_eq!(state.deletion_planned, false);
        state.refcount += 1;

        let request = Arc::new(Request {
            state: TCell::new(RequestState {
                status: RequestStatus::DependenciesPending(num_dependencies),
                place: None,
            }),
            parent,
            chunk: ChunkHandle::clone(chunk),
            exclusive,
            transaction_id,
            next_in_queue: RequestLink::new(),
        });

        self.queue.push_back(RequestRef::from(&request));
        request
    }

    pub(crate) fn satisfy_dependency(&mut self, request: &RequestHandle) {
        let n = match request.status(&self.token) {
            RequestStatus::DependenciesPending(n) => n - 1,
            other => panic!("invalid status: {:?}", other),
        };

        request.set_status(RequestStatus::DependenciesPending(n), &mut self.token);

        if n == 0 {
            self.queue.try_push_back(request.into());
        }
    }

    pub(crate) fn may_submit_request(&mut self, request: &RequestHandle) -> bool {
        let request: RequestRef = request.into();
        let status = request.status(&self.token);
        if status != RequestStatus::WaitingForSubmission {
            return false;
        }

        let state = request.chunk.state.borrow_mut(&mut self.token);
        if state.poll_lock(request).is_pending() {
            request.set_status(RequestStatus::WaitingForLock, &mut self.token);
            return false;
        }

        true
    }

    pub(crate) fn submit_request(&mut self, request: &RequestHandle, place: Option<MemoryKind>) {
        let request: RequestRef = request.into();
        assert_eq!(
            request.status(&self.token),
            RequestStatus::WaitingForSubmission
        );

        let state = request.chunk.state.borrow_mut(&mut self.token);
        state.lock(request);

        let state = request.state.borrow_mut(&mut self.token);
        state.place = place;
        state.status = RequestStatus::Submitted;

        self.queue.push_back(request);
    }

    pub(crate) unsafe fn finish_request(&mut self, request: &RequestHandle) {
        use RequestStatus::*;

        let request: RequestRef = request.into();
        assert!(matches!(request.status(&self.token), Active | Terminated));

        self.release_request_resources(request);
    }

    pub(crate) fn handle_event(&mut self, event: Event) {
        //warn!("event: {:#?}", event);

        let (dst, chunk) = match event {
            Event::MarkValidFile { chunk, result } => {
                let mut state = chunk.state.borrow_mut(&mut self.token);

                state.file_entry = match result {
                    Ok(id) => FileEntry::Valid(id),
                    Err(e) => FileEntry::Error(e),
                };

                (MemoryKind::FileSystem, chunk)
            }
            Event::MarkValidHost { chunk, result } => {
                result.expect("???");

                let state = chunk.state.borrow_mut(&mut self.token);
                let entry = &mut state.host_entry;
                entry.status = EntryStatus::Valid;

                (MemoryKind::Host, chunk)
            }
            Event::MarkValidDevice {
                chunk,
                device_id,
                result,
            } => {
                result.expect("???");

                let state = chunk.state.borrow_mut(&mut self.token);
                let entry = &mut state.device_entries[device_id.get()];
                entry.status = EntryStatus::Valid;

                (MemoryKind::Device(device_id), chunk)
            }
        };

        let state = chunk.state.borrow_mut(&mut self.token);
        let mut requests = state.active_transfers.remove_by_dst(dst);
        self.queue.extend(&mut requests);

        self.delete_chunk_when_idle(&chunk);
    }

    pub(crate) fn poll(&mut self) -> Option<(RequestHandle, RequestEvent)> {
        while let Some(request) = self.queue.pop_front() {
            if let Some(event) = self.poll_request(RequestRef::from(&request)) {
                return Some((request, event));
            }
        }

        None
    }

    fn release_request_resources(&mut self, request: RequestRef) {
        use RequestStatus::*;

        let state = request.state.borrow_mut(&mut self.token);
        let old_status = replace(&mut state.status, RequestStatus::Terminated);
        let place = state.place;

        match old_status {
            AllocateDevice(_) | WaitingForData | Active => {
                self.host.unreserve_allocation(request, &mut self.token);
                self.make_progress_host_allocations();
            }
            AllocateHost
            | Terminated
            | DependenciesPending(_)
            | WaitingForLock
            | WaitingForSubmission
            | Submitted => {}
        }

        if let Some(MemoryKind::Device(i)) = place {
            match old_status {
                WaitingForData | Active => {
                    let device = &mut self.devices[i.get()];
                    device.unreserve_allocation(request, &mut self.token);

                    self.make_progress_device_allocations(i);
                }
                AllocateHost | AllocateDevice(_) | Terminated => {}
                WaitingForLock | WaitingForSubmission | Submitted | DependenciesPending(_) => {}
            }
        }

        match old_status {
            AllocateDevice(_) | WaitingForData | Active | AllocateHost => {
                let state = request.chunk.state.borrow_mut(&mut self.token);
                state.unlock(request, &mut self.queue);
            }
            Terminated => {}
            WaitingForLock | WaitingForSubmission | Submitted | DependenciesPending(_) => {}
        }

        if old_status != Terminated {
            let state = request.chunk.state.borrow_mut(&mut self.token);
            state.refcount -= 1;
        }

        self.delete_chunk_when_idle(&request.chunk);
    }

    fn abort_request(&mut self, request: RequestRef, error: anyhow::Error) -> Option<RequestEvent> {
        self.release_request_resources(request); // Will also update status
        Some(RequestEvent::Abort(error))
    }

    fn poll_request(&mut self, request: RequestRef) -> Option<RequestEvent> {
        use RequestStatus::*;

        loop {
            match request.status(&self.token) {
                DependenciesPending(n) => {
                    if n > 0 {
                        return None;
                    }

                    request.set_status(WaitingForLock, &mut self.token);
                }
                WaitingForLock => {
                    let state = request.chunk.state.borrow_mut(&mut self.token);
                    if state.poll_lock(request).is_pending() {
                        return None;
                    }

                    request.set_status(WaitingForSubmission, &mut self.token);
                    return Some(RequestEvent::Ready);
                }
                WaitingForSubmission => {
                    //
                }
                Submitted => {
                    request.set_status(AllocateHost, &mut self.token);
                    if !self.push_host_allocation(request) {
                        return None;
                    }
                }
                AllocateHost => {
                    match self.poll_host_allocation(request) {
                        Ready(Ok(())) => {}
                        Ready(Err(e)) => {
                            return self.abort_request(request, e.into());
                        }
                        Pending => return None,
                    }

                    let place = request.state.borrow(&self.token).place;

                    match place {
                        Some(MemoryKind::Device(device_id)) => {
                            request.set_status(AllocateDevice(device_id), &mut self.token);
                            if !self.push_device_allocation(device_id, request) {
                                return None;
                            }
                        }
                        Some(MemoryKind::Host) | None => {
                            request.set_status(WaitingForData, &mut self.token);
                        }
                        _ => {
                            unimplemented!();
                        }
                    }
                }
                AllocateDevice(device_id) => {
                    match self.poll_device_allocation(device_id, request) {
                        Ready(Ok(())) => {}
                        Ready(Err(e)) => {
                            return self.abort_request(request, e.into());
                        }
                        Pending => return None,
                    }

                    request.set_status(WaitingForData, &mut self.token);
                }
                WaitingForData => {
                    let place = self.determine_request_place(request);

                    match self.poll_data(place, request) {
                        Ready(Ok(())) => {}
                        Ready(Err(e)) => {
                            return self.abort_request(request, e.into());
                        }
                        Pending => return None,
                    }

                    if request.exclusive {
                        if self.poll_exclusive(place, request).is_pending() {
                            return None;
                        }
                    }

                    request.set_status(Active, &mut self.token);
                    return Some(RequestEvent::Active);
                }
                Active => {}
                Terminated => {}
            }
        }
    }

    fn determine_request_place(&mut self, request: RequestRef) -> MemoryKind {
        let state = request.state.borrow_mut(&mut self.token);
        if let Some(place) = state.place {
            return place;
        }

        let chunk = &request.chunk;
        let affinity = chunk.layout.affinity;
        let state = chunk.state.borrow_mut(&mut self.token);

        // Try to find a valid place
        // - if buffer is valid at affinity, use affinity
        // - otherwise, if buffer is valid at host, use host
        // - otherwise, if buffer valid at some device, use that device
        // - otherwise, buffer is not valid anywhere, use affinity
        // - otherwise, if affinity is None, use host
        let mut valid_place = 'a: loop {
            if let Some(p) = affinity {
                if state.is_valid_at(p) {
                    break p;
                }
            }

            if state.host_entry.is_valid() {
                break MemoryKind::Host;
            }

            for (i, entry) in enumerate(&mut state.device_entries) {
                if entry.is_valid() {
                    break 'a MemoryKind::Device(DeviceId::new(i));
                }
            }

            if let Some(p) = affinity {
                break p;
            }

            break MemoryKind::Host;
        };

        if let MemoryKind::Device(i) = valid_place {
            let device = &mut self.devices[i.get()];

            if !device.try_reserve_allocation(request, &mut self.token) {
                valid_place = MemoryKind::Host;
            }
        }

        let state = request.state.borrow_mut(&mut self.token);
        state.place = Some(valid_place);
        valid_place
    }

    fn push_host_allocation(&mut self, request: RequestRef) -> bool {
        if self.host.active_allocation.is_none() {
            self.host.active_allocation = Some(RequestRef::upgrade(request));
            true
        } else {
            self.host.pending_allocations.push_back(request);
            false
        }
    }

    fn poll_host_allocation(&mut self, request: RequestRef) -> Poll<Result> {
        if !ptr::eq(
            RequestRef::into_raw(request),
            self.host
                .active_allocation
                .as_ref()
                .map_or(ptr::null(), Arc::as_ptr),
        ) {
            return Pending;
        }

        let result = loop {
            if self.host.try_reserve_allocation(request, &mut self.token) {
                break Ok(());
            }

            match self.evict_host_allocation(request) {
                Pending => return Pending,
                Ready(Err(e)) => break Err(e),
                Ready(Ok(true)) => continue, // Retry
                Ready(Ok(false)) => {}       //
            }

            if self.host.is_out_of_memory(request) {
                break Err(AllocationError::HostMemoryExhausted.into());
            } else {
                return Pending;
            }
        };

        self.host.active_allocation = self.host.pending_allocations.pop_front();

        if let Some(request) = &self.host.active_allocation {
            self.queue.try_push_back(request.into());
        }

        Ready(result)
    }

    fn make_progress_host_allocations(&mut self) {
        if let Some(request) = &self.host.active_allocation {
            self.queue.try_push_back(request.into());
        }
    }

    fn evict_host_allocation(&mut self, request: RequestRef) -> Poll<Result<bool>> {
        if self.storage.is_none() {
            return Ready(Ok(false));
        }

        let chunk = match self.host.lru.peek_lru() {
            Some((chunk, _)) => ChunkHandle::clone(&chunk),
            None => return Ready(Ok(false)),
        };

        let chunk = ChunkRef::from(&chunk);

        match self.poll_file_data(chunk, request) {
            Pending => return Pending,
            Ready(Ok(())) => {}
            Ready(Err(e)) => {
                return Ready(match e {
                    StorageError::CapacityExceeded(_) => Ok(false),
                    e => {
                        // TODO: IO error occurred while copying to file. Not sure how to handle this.
                        error!("IO error while evicting to file: {:?}", e);
                        Err(e.into())
                    }
                });
            }
        }

        let state = chunk.state.borrow_mut(&mut self.token);

        if let Some(transfer) = state.active_transfers.find_any() {
            transfer.add_request(request);
            return Pending;
        }

        // Deallocate all device entries.
        for device in &mut self.devices {
            device.deallocate_entry(chunk, &mut self.token);
        }

        // Deallocate host entry
        self.host.deallocate_entry(chunk, &mut self.token);

        Ready(Ok(true))
    }

    fn push_device_allocation(&mut self, device_id: DeviceId, request: RequestRef) -> bool {
        let device = &mut self.devices[device_id.get()];

        if device.active_allocation.is_none() {
            device.active_allocation = Some(RequestRef::upgrade(request));
            true
        } else {
            device.pending_allocations.push_back(request);
            false
        }
    }

    fn make_progress_device_allocations(&mut self, device_id: DeviceId) {
        let device = &mut self.devices[device_id.get()];

        if let Some(request) = &device.active_allocation {
            self.queue.try_push_back(request.into());
        }
    }

    fn poll_device_allocation(
        &mut self,
        device_id: DeviceId,
        request: RequestRef,
    ) -> Poll<Result<(), AllocationError>> {
        let device = &mut self.devices[device_id.get()];

        if !ptr::eq(
            RequestRef::into_raw(request),
            device
                .active_allocation
                .as_ref()
                .map_or(ptr::null(), Arc::as_ptr),
        ) {
            return Pending;
        }

        let result = loop {
            let device = &mut self.devices[device_id.get()];

            if device.try_reserve_allocation(request, &mut self.token) {
                break Ok(());
            }

            match self.evict_device_allocation(device_id, request) {
                Pending => return Pending,
                Ready(true) => continue,
                Ready(false) => {}
            };

            let device = &mut self.devices[device_id.get()];
            if device.is_out_of_memory(request) {
                break Err(AllocationError::DeviceMemoryExhausted);
            } else {
                return Pending;
            }
        };

        let device = &mut self.devices[device_id.get()];
        device.active_allocation = device.pending_allocations.pop_front();

        if let Some(request) = &device.active_allocation {
            self.queue.try_push_back(request.into());
        }

        Ready(result)
    }

    fn evict_device_allocation(&mut self, device_id: DeviceId, request: RequestRef) -> Poll<bool> {
        let device = &mut self.devices[device_id.get()];

        let chunk = match device.lru.peek_lru() {
            Some((chunk, _)) => ChunkHandle::clone(&chunk),
            None => return Ready(false),
        };

        let state = chunk.state.borrow_mut(&mut self.token);
        if let Some(transfer) = state
            .active_transfers
            .find_by_src_or_dst(MemoryKind::Device(device_id))
        {
            transfer.add_request(request);
            return Pending;
        }

        let chunk = ChunkRef::from(&chunk);

        if state.is_valid_only_at(MemoryKind::Device(device_id)) {
            let poll = self.poll_host_data(chunk, request);
            assert!(poll.is_pending());
            return Pending;
        }

        device.deallocate_entry(chunk, &mut self.token);

        Ready(true)
    }

    fn poll_data(&mut self, place: MemoryKind, request: RequestRef) -> Poll<Result> {
        let chunk = ChunkRef::from(&request.chunk);

        match place {
            MemoryKind::Host => self.poll_host_data(chunk, request),
            MemoryKind::Device(device_id) => self.poll_device_data(device_id, chunk, request),
            _ => unreachable!(), // Place must be known by now.
        }
    }

    fn poll_file_data(
        &mut self,
        chunk: ChunkRef,
        request: RequestRef,
    ) -> Poll<Result<(), StorageError>> {
        let state = chunk.state.borrow_mut(&mut self.token);
        let entry = &mut state.file_entry;

        match take(entry) {
            FileEntry::Valid(id) => {
                *entry = FileEntry::Valid(id);
                return Ready(Ok(()));
            }
            FileEntry::Error(e) => {
                return Ready(Err(e));
            }
            FileEntry::Invalid => {}
        }

        if let Some(transfer) = state.active_transfers.find_by_dst(MemoryKind::FileSystem) {
            transfer.add_request(request);
            return Pending;
        }

        if self.poll_host_data(chunk, request).is_pending() {
            return Pending;
        }

        let transfer = unsafe { self.copy_host_to_file(chunk) };
        transfer.add_request(request);

        Pending
    }

    fn poll_host_data(&mut self, chunk: ChunkRef, request: RequestRef) -> Poll<Result> {
        let state = chunk.state.borrow_mut(&mut self.token);
        let entry = &state.host_entry;

        if entry.is_valid() {
            return Ready(Ok(()));
        }

        if let Some(transfer) = state.active_transfers.find_by_dst(MemoryKind::Host) {
            transfer.add_request(request);
            return Pending;
        }

        if let Some(index) = state.device_entries.iter().position(|e| e.is_valid()) {
            let device_id = DeviceId::new(index);
            let transfer = unsafe { self.copy_device_to_host(device_id, chunk) };
            transfer.add_request(request);

            return Pending;
        }

        if state.file_entry.is_valid() {
            return match unsafe { self.copy_file_to_host(chunk) } {
                Ok(transfer) => {
                    transfer.add_request(request);
                    Pending
                }
                Err(e) => Ready(Err(e)),
            };
        }

        // Data seems to be valid nowhere, set entry to valid.
        state.host_entry.status = EntryStatus::Valid;
        Ready(Ok(()))
    }

    fn poll_device_data(
        &mut self,
        device_id: DeviceId,
        chunk: ChunkRef,
        request: RequestRef,
    ) -> Poll<Result> {
        let state = chunk.state.borrow_mut(&mut self.token);
        let entry = &mut state.device_entries[device_id.get()];

        if entry.is_valid() {
            return Ready(Ok(()));
        }

        if let Some(transfer) = state
            .active_transfers
            .find_by_dst(MemoryKind::Device(device_id))
        {
            transfer.add_request(request);
            return Pending;
        }

        for i in 0..self.devices.len() {
            let src_id = DeviceId::new(i);

            if !self.copy_engine.supported_d2d(src_id, device_id) {
                continue;
            }

            if state.device_entries[src_id.get()].is_valid() {
                let transfer = unsafe { self.copy_device_to_device(src_id, device_id, chunk) };
                transfer.add_request(request);
                return Pending;
            }
        }

        if !state.is_valid_anywhere() {
            let entry = &mut state.device_entries[device_id.get()];
            entry.status = EntryStatus::Valid;

            return Ready(Ok(()));
        }

        if self.poll_host_data(chunk, request).is_pending() {
            return Pending;
        }

        let transfer = unsafe { self.copy_host_to_device(device_id, chunk) };
        transfer.add_request(request);
        Pending
    }

    fn poll_exclusive(&mut self, place: MemoryKind, request: RequestRef) -> Poll<()> {
        let chunk = &request.chunk;
        let state = chunk.state.borrow_mut(&mut self.token);

        if let Some(transfer) = state.active_transfers.find_any() {
            transfer.add_request(request);
            return Pending;
        }

        // Invalidate host
        if place == MemoryKind::Host {
            assert_eq!(state.host_entry.status, EntryStatus::Valid);
        } else {
            state.host_entry.status = EntryStatus::Invalid;
        }

        // Invalidate devices
        for (index, dentry) in enumerate(&mut state.device_entries) {
            if place == MemoryKind::Device(DeviceId::new(index)) {
                assert_eq!(dentry.status, EntryStatus::Valid);
            } else {
                dentry.status = EntryStatus::Invalid;
            }
        }

        // Invalidate file
        if let FileEntry::Valid(id) = take(&mut state.file_entry) {
            self.storage.as_ref().unwrap().delete_async(id);
        }

        Ready(())
    }

    unsafe fn copy_device_to_host<'a>(
        &'a mut self,
        device_id: DeviceId,
        chunk_ref: ChunkRef<'a>,
    ) -> &'a mut DataTransfer {
        debug!("COPY D2H");

        let sender = self.sender.clone();
        let chunk = ChunkRef::upgrade(chunk_ref);

        let state = ChunkRef::downgrade(chunk_ref)
            .state
            .borrow_mut(&mut self.token);
        let nbytes = chunk.size_in_bytes;
        let src = state.device_entries[device_id.get()].data.unwrap();
        let dst = state.host_entry.data.unwrap();

        self.copy_engine.copy_d2h(
            device_id,
            src,
            dst.as_ptr() as *mut (),
            nbytes,
            move |result| {
                sender
                    .send(Event::MarkValidHost {
                        chunk,
                        result: result.map_err(|e| e.into()),
                    })
                    .unwrap();
            },
        );

        state
            .active_transfers
            .insert(MemoryKind::Device(device_id), MemoryKind::Host)
    }

    unsafe fn copy_host_to_device<'a>(
        &'a mut self,
        device_id: DeviceId,
        chunk_ref: ChunkRef<'a>,
    ) -> &'a mut DataTransfer {
        debug!("COPY H2D");

        let sender = self.sender.clone();
        let chunk = ChunkRef::upgrade(chunk_ref);

        let state = ChunkRef::downgrade(chunk_ref)
            .state
            .borrow_mut(&mut self.token);
        let nbytes = chunk.size_in_bytes;
        let src = state.host_entry.data.unwrap();
        let dst = state.device_entries[device_id.get()].data.unwrap();

        self.copy_engine.copy_h2d(
            src.as_ptr() as *const (),
            device_id,
            dst,
            nbytes,
            move |result| {
                sender
                    .send(Event::MarkValidDevice {
                        chunk,
                        device_id,
                        result,
                    })
                    .unwrap();
            },
        );

        state
            .active_transfers
            .insert(MemoryKind::Host, MemoryKind::Device(device_id))
    }

    unsafe fn copy_device_to_device<'a>(
        &'a mut self,
        src_id: DeviceId,
        dst_id: DeviceId,
        chunk_ref: ChunkRef<'a>,
    ) -> &'a mut DataTransfer {
        let sender = self.sender.clone();
        let chunk = ChunkRef::upgrade(chunk_ref);

        let state = ChunkRef::downgrade(chunk_ref)
            .state
            .borrow_mut(&mut self.token);
        let nbytes = chunk.size_in_bytes;
        let src_ptr = state.device_entries[src_id.get()].data.unwrap();
        let dst_ptr = state.device_entries[dst_id.get()].data.unwrap();

        self.copy_engine
            .copy_d2d(src_id, src_ptr, dst_id, dst_ptr, nbytes, move |result| {
                sender
                    .send(Event::MarkValidDevice {
                        chunk,
                        device_id: dst_id,
                        result,
                    })
                    .unwrap();
            });

        state
            .active_transfers
            .insert(MemoryKind::Device(src_id), MemoryKind::Device(dst_id))
    }

    unsafe fn copy_host_to_file<'a>(&'a mut self, chunk_ref: ChunkRef<'a>) -> &'a mut DataTransfer {
        let sender = self.sender.clone();
        let chunk = ChunkRef::upgrade(chunk_ref);

        let state = ChunkRef::downgrade(chunk_ref)
            .state
            .borrow_mut(&mut self.token);
        let hptr = state.host_entry.data.unwrap();
        let size_in_bytes = chunk_ref.size_in_bytes;

        self.storage
            .as_ref()
            .unwrap()
            .create_async(hptr.as_ptr(), size_in_bytes, move |result| {
                sender.send(Event::MarkValidFile { chunk, result }).unwrap();
            });

        state
            .active_transfers
            .insert(MemoryKind::Host, MemoryKind::FileSystem)
    }

    unsafe fn copy_file_to_host<'a>(
        &'a mut self,
        chunk_ref: ChunkRef<'a>,
    ) -> Result<&'a mut DataTransfer> {
        let sender = self.sender.clone();
        let chunk = ChunkRef::upgrade(chunk_ref);

        let state = ChunkRef::downgrade(chunk_ref)
            .state
            .borrow_mut(&mut self.token);
        let path = match &state.file_entry {
            FileEntry::Valid(id) => *id,
            _ => unreachable!(),
        };

        let hptr = state.host_entry.data.unwrap();
        let size_in_bytes = chunk.size_in_bytes;

        self.storage.as_ref().unwrap().read_async(
            path,
            hptr.as_ptr(),
            size_in_bytes,
            move |result| {
                sender
                    .send(Event::MarkValidHost {
                        chunk,
                        result: result.map_err(|e| e.into()),
                    })
                    .unwrap();
            },
        );

        let transfer = state
            .active_transfers
            .insert(MemoryKind::FileSystem, MemoryKind::Host);
        Ok(transfer)
    }
}
