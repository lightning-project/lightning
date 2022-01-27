use crate::prelude::*;
use parking_lot::Condvar;
use std::fmt;
use std::mem::ManuallyDrop;

use std::sync::Arc;

/// Value of `T` which will arrive at some later point in time.
#[derive(Debug)]
pub struct Future<T = ()> {
    inner: ManuallyDrop<Arc<Inner<T>>>,
}

/// Object used to resolve a `Future<T>`.
#[derive(Debug)]
pub struct Promise<T = ()> {
    inner: ManuallyDrop<Arc<Inner<T>>>,
}

#[derive(Debug)]
struct Inner<T> {
    cond: Condvar,
    state: Mutex<State<T>>,
}

use State::*;

enum State<T> {
    Empty,
    Completed(T),
    Poisoned,
    Callback(Box<dyn FnOnce(Result<T, FutureError>) + Send>),
}

impl<T> fmt::Debug for State<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Empty => write!(f, "Emtpy"),
            Completed(_) => write!(f, "Completed(_)"),
            Poisoned => write!(f, "Poinsoned"),
            Callback(_) => write!(f, "Callback(_)"),
        }
    }
}

impl<T> Promise<T> {
    /// Create a new future and promise. The promise should be given to the producer to write
    /// the value while the future must be given to the consumer to read the value.
    pub fn new() -> (Promise<T>, Future<T>) {
        let inner = Arc::new(Inner {
            cond: Condvar::new(),
            state: Mutex::new(Empty),
        });
        let inner2 = Arc::clone(&inner);

        (
            Promise {
                inner: ManuallyDrop::new(inner),
            },
            Future {
                inner: ManuallyDrop::new(inner2),
            },
        )
    }

    /// Complete this promise with the given value. This will resolve the associated [`Future`].
    pub fn complete(mut self, value: T) {
        let mut state = self.inner.state.lock();
        match &*state {
            Empty => {
                *state = Completed(value);
                self.inner.cond.notify_all();
            }
            Callback(_) => match replace(&mut *state, Poisoned) {
                Callback(fun) => {
                    fun(Ok(value));
                }
                _ => unreachable!(),
            },
            Poisoned => {
                // The consumer dropped the future, there is no point in writing the result.
            }
            v => panic!("invalid state: {:?}", v),
        }
        drop(state);

        unsafe {
            ManuallyDrop::drop(&mut self.inner);
        }
        forget(self);
    }
}

impl<T> Drop for Promise<T> {
    fn drop(&mut self) {
        let mut state = self.inner.state.lock();
        match replace(&mut *state, Poisoned) {
            Empty => {
                self.inner.cond.notify_all();
            }
            Callback(fun) => fun(Err(FutureError)),
            Poisoned => {
                // The consumer dropped the future, there is no point in doing anything.
            }
            v => panic!("invalid state: {:?}", v),
        }
        drop(state);

        unsafe {
            ManuallyDrop::drop(&mut self.inner);
        }
    }
}

/// Error indicating that a promise was dropped before it was completed.
#[derive(Error, Debug)]
#[error("promise dropped while waiting for future to complete")]
pub struct FutureError;

impl<T> Future<T> {
    pub fn ready(value: T) -> Self {
        let inner = Arc::new(Inner {
            cond: Condvar::new(),
            state: Mutex::new(Completed(value)),
        });

        Future {
            inner: ManuallyDrop::new(inner),
        }
    }

    /// Blocks the caller until the value of `T` is available.
    ///
    /// Returns an error if the associated [`Promise`] was dropped without completing it.
    pub fn is_ready(&self) -> bool {
        let mut state = self.inner.state.lock();
        match replace(&mut *state, Empty) {
            Empty => false,
            Completed(_) | Poisoned => true,
            _ => panic!("invalid state"),
        }
    }

    /// Blocks the caller until the value of `T` is available.
    ///
    /// # Panics
    /// Panic if the associated [`Promise`] was dropped without completing it.
    pub fn wait(self) -> T {
        self.wait_or_err().expect("failed to get value from future")
    }

    /// Blocks the caller until the value of `T` is available.
    ///
    /// Returns an error if the associated [`Promise`] was dropped without completing it.
    pub fn wait_or_err(mut self) -> Result<T, FutureError> {
        let mut state = self.inner.state.lock();
        let result = loop {
            match replace(&mut *state, Empty) {
                Empty => self.inner.cond.wait(&mut state),
                Completed(val) => {
                    break Ok(val);
                }
                Poisoned => {
                    break Err(FutureError);
                }
                _ => panic!("invalid state"),
            }
        };

        drop(state);
        unsafe {
            ManuallyDrop::drop(&mut self.inner);
        }
        forget(self);

        result
    }

    /// Maps an [`Future<T>`] to [`Future<R>`] using the supplied function `FnOnce(T) -> R`.
    pub fn map<F, R>(self, fun: F) -> Future<R>
    where
        F: FnOnce(T) -> R + Send + 'static,
        R: Send + 'static,
    {
        let (promise, future) = Promise::new();
        self.attach_callback(move |result| {
            if let Ok(value) = result {
                promise.complete(fun(value))
            }
        });

        future
    }

    /// Attach a callback which will be called with `Ok(T)` when the value is available. The
    /// callback is called with `Err` if the associated [`Promise`] was dropped without
    /// completing it.
    ///
    /// Note that there are two scenarios:
    /// * The callback is called _immediately_ in the _current_ thread if the value is available.
    /// * The callback is called _later_ in a _different_ thread when [`Promise::complete`] is called.
    ///
    /// Due to the second case, it is important that the provided callback does not block or
    /// panic since that would propagate to the other thread.
    pub fn attach_callback<F>(self, fun: F)
    where
        F: FnOnce(Result<T, FutureError>) + Send + 'static,
    {
        self.attach_boxed_callback(Box::new(fun))
    }

    fn attach_boxed_callback(mut self, fun: Box<dyn FnOnce(Result<T, FutureError>) + Send>) {
        let mut state = self.inner.state.lock();
        match replace(&mut *state, Empty) {
            Empty => {
                *state = Callback(fun);
            }
            Completed(val) => {
                (fun)(Ok(val));
            }
            Poisoned => {
                (fun)(Err(FutureError));
            }
            _ => panic!("invalid state"),
        };

        drop(state);
        unsafe {
            ManuallyDrop::drop(&mut self.inner);
        }
        forget(self);
    }
}

impl<T> Drop for Future<T> {
    fn drop(&mut self) {
        let mut state = self.inner.state.lock();
        *state = Poisoned;
        drop(state);

        unsafe {
            ManuallyDrop::drop(&mut self.inner);
        }
    }
}
