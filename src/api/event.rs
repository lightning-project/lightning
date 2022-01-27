use crate::driver::DriverEvent;
use crate::prelude::*;
use lightning_core::util::Promise;

/// Run-time event.
///
/// Each kernel invocation returns an event which can be queried for completion.
pub struct Event {
    pub(crate) future: DriverEvent,
}

impl Event {
    /// Block until the events completes.
    pub fn wait(self) -> Result {
        let (promise, future) = Promise::new();
        self.then(|result| promise.complete(result));
        future.wait()
    }

    /// Query whether the event has completed.
    pub fn query(&self) -> bool {
        self.future.is_ready()
    }

    /// Calls the provided callback when the event completes (or immediately if the event has
    /// already completed).
    pub fn then<F>(self, fun: F)
    where
        F: FnOnce(Result) + Send + 'static,
    {
        self.future.attach_callback(|result| fun(result))
    }
}
