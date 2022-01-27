//! Management of CUDA events.

use crate::{cuda_call, cuda_check, Error, Result, Stream};
use cuda_driver_sys::*;
use std::os::raw::c_uint;
use std::time::Duration;
use std::{fmt, mem};

/// CUDA Event.
///
/// CUDA uses events to organize synchronization of [`Stream`]s. Events can record onto a compute
/// stream and will fire once all all previously enqueued work onto the stream has completed.
/// This object wraps a `CUevent`.
///
/// [`Stream`]: struct.Stream.html
#[derive(PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Event(CUevent);

// Events are thread-safe since all calls are passed ot the CUDA driver API which is thread-safe.
unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    /// Create a new event with timing disabled.
    pub fn new() -> Result<Self> {
        Event::with_flags(EventFlags::DISABLE_TIMING)
    }

    /// Create a new event with timing enabled.
    pub fn with_timing() -> Result<Self> {
        Event::with_flags(EventFlags::empty())
    }

    /// Create a new event with custom flags.
    pub fn with_flags(flags: EventFlags) -> Result<Self> {
        unsafe { cuda_call(|event| cuEventCreate(event, flags.bits)).map(Self) }
    }

    /// Record this event on the given stream.
    ///
    /// The event fires once all previously enqueued work on the stream has completed.
    pub fn record(&self, stream: &Stream) -> Result {
        unsafe { cuda_check(cuEventRecord(self.0, stream.raw())) }
    }

    /// Returns `true` if all the event has fired and `false` otherwise.
    pub fn query(&self) -> Result<bool> {
        unsafe {
            match cuEventQuery(self.0) {
                CUresult::CUDA_SUCCESS => Ok(true),
                CUresult::CUDA_ERROR_NOT_READY => Ok(false),
                err => Err(Error::from_raw(err)),
            }
        }
    }

    /// Block until this event fires.
    pub fn synchronize(&self) -> Result {
        unsafe { cuda_check(cuEventSynchronize(self.0)) }
    }

    /// Compute the time elapsed between two events.
    ///
    /// See `cuEventElapsedTime` for details. Returns `None` if the duration is negative, meaning event
    /// `start` actually fired later than `end`.
    pub fn elapsed(start: &Event, end: &Event) -> Result<Option<Duration>> {
        let time_ms = Self::elapsed_ms(start, end)?;
        let time_sec = time_ms / 1000.0;

        if f32::is_finite(time_sec) && time_sec >= 0.0 {
            Ok(Some(Duration::from_secs_f32(time_sec)))
        } else {
            Ok(None)
        }
    }

    /// Compute the time elapsed between two events in milliseconds.
    ///
    /// See `cuEventElapsedTime` for details. Time can be negative if `start` fired later than `end`.
    pub fn elapsed_ms(start: &Event, end: &Event) -> Result<f32> {
        unsafe { cuda_call(|t| cuEventElapsedTime(t, start.0, end.0)) }
    }

    /// Alias for [`Event::elapsed(start, self)`](#method.elapsed).
    pub fn elapsed_since(&self, start: &Event) -> Result<Option<Duration>> {
        Self::elapsed(start, self)
    }

    /// Returns the underlying `CUevent` object and consume this object, preventing the destructor
    /// from being run.
    #[inline(always)]
    pub fn into_raw(self) -> CUevent {
        let out = self.0;
        mem::forget(self);
        out
    }

    /// Construct from a `CUevent` object.
    ///
    /// # Safety
    /// The given `CUevent` object should be a valid CUDA event object.
    #[inline(always)]
    pub unsafe fn from_raw(evt: CUevent) -> Self {
        Self(evt)
    }

    /// Returns the underlying `CUevent` object.
    #[inline(always)]
    pub fn raw(&self) -> CUevent {
        self.0
    }
}

impl fmt::Debug for Event {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("CudaEvent").field(&self.0).finish()
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            cuEventDestroy_v2(self.0);
        }
    }
}

bitflags::bitflags! {
    /// Flags for configuring CUDA events.
    ///
    /// Represent the `CU_EVENT_*` flags from the CUDA API.
    pub struct EventFlags: c_uint {
        const BLOCKING_SYNC = CUevent_flags_enum::CU_EVENT_BLOCKING_SYNC as c_uint;
        const DISABLE_TIMING = CUevent_flags_enum::CU_EVENT_DISABLE_TIMING as c_uint;
        const INTERPROCESS = CUevent_flags_enum::CU_EVENT_INTERPROCESS as c_uint;
    }
}
