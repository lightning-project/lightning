use crate::prelude::*;
use lightning_core::util::DropGuard;
use rand::distributions::Alphanumeric;
use rand::prelude::*;
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use threadpool::ThreadPool;

const FILE_NAME_LENGTH: usize = 40;

#[derive(Error, Debug)]
pub(crate) enum Error {
    #[error("io error: {0}")]
    IO(#[from] io::Error),

    #[error("path {0:?} does not exist or is not a directory")]
    InvalidDirectory(PathBuf),

    #[error("storage id not found: {0:?}")]
    NotFound(StorageId),

    #[error("file size mismatch: {0:?} bytes")]
    InvalidSize(usize),

    #[error("operation failed since storage is shutting down")]
    Shutdown,

    #[error("storage has exceeded maximum capacity of {0:?} bytes")]
    CapacityExceeded(u64),
}

struct Entry {
    path: PathBuf,
    size: u64,
}

struct Inner {
    tmp_directory: PathBuf,                    // Where to store files
    next_id: AtomicU64,                        // Next id to generate unique ID.
    entries: Mutex<HashMap<StorageId, Entry>>, // Mapping of storage id => file name.
    used_size: AtomicU64,
    max_size: u64,
}

impl Drop for Inner {
    fn drop(&mut self) {
        for entry in take(&mut self.entries).into_inner().values() {
            if let Err(e) = fs::remove_file(&entry.path) {
                warn!("error while deleting {:?}: {}", entry.path, e);
            }
        }
    }
}

pub(crate) struct Storage {
    io_thread: ThreadPool,
    state: Arc<Inner>,
}

#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug)]
pub(crate) struct StorageId(u64);

impl Storage {
    pub(crate) fn new(tmp_directory: PathBuf, max_size: u64) -> Result<Self, Error> {
        if !tmp_directory.exists() {
            let _ = fs::create_dir_all(&tmp_directory);
        }

        if !tmp_directory.is_dir() {
            return Err(Error::InvalidDirectory(tmp_directory));
        }

        Ok(Self {
            io_thread: ThreadPool::new(1),
            state: Arc::new(Inner {
                tmp_directory: tmp_directory.canonicalize()?,
                next_id: default(),
                entries: default(),
                used_size: AtomicU64::new(0),
                max_size,
            }),
        })
    }

    pub(super) unsafe fn create_async<F>(&self, ptr: *const u8, nbytes: usize, completion: F)
    where
        F: FnOnce(Result<StorageId, Error>),
        F: Send + 'static,
    {
        let state = Arc::clone(&self.state);
        let completion = DropGuard::new(completion, |f: F| f(Err(Error::Shutdown)));
        let ptr = ptr as usize;

        self.io_thread.execute(move || {
            loop {
                let size = state.used_size.load(Ordering::SeqCst);
                let new_size = size + nbytes as u64;

                if new_size > state.max_size {
                    let result = Err(Error::CapacityExceeded(state.max_size));
                    return (completion.into_inner())(result);
                }

                if state
                    .used_size
                    .compare_exchange(size, new_size, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    break;
                }
            }

            let id = state.next_id.fetch_add(1, Ordering::SeqCst);

            let mut tries = 0;
            let path = loop {
                tries += 1;
                let filename: String = Alphanumeric
                    .sample_iter(thread_rng())
                    .take(FILE_NAME_LENGTH)
                    .map(char::from)
                    .collect();

                let mut path = state.tmp_directory.clone();
                path.push(filename);

                if !path.exists() || tries > 100 {
                    break path;
                }
            };

            let result = try_block! {
                let mut f = OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(&path)?;

                //let before = std::time::Instant::now();

                let data = unsafe { slice::from_raw_parts(ptr as *const u8, nbytes) };
                f.write_all(data)?;
                f.sync_data()?;

                //warn!("write IO b/s: {}", nbytes as f64 / before.elapsed().as_secs_f64());

                Ok(StorageId(id))
            };

            // If there was an error, delete the file again.
            if let Ok(id) = result {
                state.entries.lock().insert(
                    id,
                    Entry {
                        path,
                        size: nbytes as u64,
                    },
                );
            } else {
                let _ = fs::remove_file(&path);
            }

            (completion.into_inner())(result);
        });
    }

    pub(super) unsafe fn read_async<F>(
        &self,
        id: StorageId,
        ptr: *mut u8,
        nbytes: usize,
        completion: F,
    ) where
        F: FnOnce(Result<(), Error>),
        F: Send + 'static,
    {
        let state = Arc::clone(&self.state);
        let ptr = ptr as usize;
        let completion = DropGuard::new(completion, |f: F| f(Err(Error::Shutdown)));

        self.io_thread.execute(move || {
            let result = try_block! {
                let guard = state.entries.lock();
                let entry = guard.get(&id).ok_or_else(move || Error::NotFound(id))?;

                if entry.size != nbytes as u64 {
                    return Err(Error::InvalidSize(nbytes));
                }

                let mut f = OpenOptions::new().read(true).open(&entry.path)?;

                // Drop lock, we must not hold it while reading the file.
                drop(guard);

                //let before = std::time::Instant::now();
                let data = unsafe { slice::from_raw_parts_mut(ptr as *mut u8, nbytes) };
                f.read_exact(data)?;
                //warn!("read IO b/s: {}", nbytes as f64 / before.elapsed().as_secs_f64());

                Ok(())
            };

            (completion.into_inner())(result);
        });
    }

    pub(super) fn delete_async(&self, id: StorageId) {
        let state = Arc::clone(&self.state);

        self.io_thread.execute(move || {
            let entry = match state.entries.lock().remove(&id) {
                Some(f) => f,
                None => return,
            };

            state.used_size.fetch_sub(entry.size, Ordering::SeqCst);

            if let Err(e) = fs::remove_file(&entry.path) {
                warn!("error while deleting {:?}: {}", entry.path, e);
            }
        });
    }
}
