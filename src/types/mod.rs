//! Common types used throughout the lightning runtime system.
mod chunk;
mod config;
pub(crate) mod dag;
mod tasklet;

use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug, Display};
use std::num::NonZeroU64;

pub use lightning_core::accessor::*;
pub use lightning_core::info::*;
pub use lightning_core::*;
pub use lightning_memops::{Reduction, ReductionFunction};

pub use self::chunk::*;
pub use self::config::*;
pub use self::tasklet::*;

#[derive(Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Debug, Hash)]
pub struct SyncId(pub(crate) NonZeroU64);

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct EventId(NonZeroU64);

const EVENT_ID_BITS: u64 = 56;
impl EventId {
    pub(crate) fn new(node_id: WorkerId, index: NonZeroU64) -> Self {
        assert_eq!(index.get() & ((!0) << EVENT_ID_BITS), 0);
        let lhs = node_id.0 as u64;
        let rhs = unsafe { NonZeroU64::new_unchecked(index.get() << (64 - EVENT_ID_BITS)) };

        Self(lhs | rhs)
    }

    pub const fn null(node_id: WorkerId) -> Self {
        let lhs = node_id.0 as u64;
        let rhs = 1 << (64 - EVENT_ID_BITS);
        Self(unsafe { NonZeroU64::new_unchecked(lhs | rhs) })
    }

    pub fn get(&self) -> u64 {
        self.0.get() >> (64 - EVENT_ID_BITS)
    }

    pub fn owner(&self) -> WorkerId {
        WorkerId(self.0.get() as u8)
    }
}

impl Debug for EventId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("EventId")
            .field(&format_args!("{}:{}", self.owner(), self.get()))
            .finish()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CudaKernelId(pub(crate) u64);

impl Display for CudaKernelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "kernel{}", self.0)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum CudaArg {
    Value(DataValue),
    Array(CudaArgArray),
}

impl CudaArg {
    pub fn value(value: DataValue) -> Self {
        Self::Value(value)
    }

    pub fn array(ndims: usize, array_index: usize, domain: Rect, transform: Affine) -> Self {
        Self::Array(CudaArgArray {
            ndims,
            array_index,
            domain,
            transform,
            per_block: None,
        })
    }

    pub fn array_per_block(
        ndims: usize,
        array_index: usize,
        per_block: Transform,
        domain: Rect,
        transform: Affine,
    ) -> Self {
        Self::Array(CudaArgArray {
            ndims,
            array_index,
            domain,
            transform,
            per_block: Some(Box::new(per_block)),
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CudaArgArray {
    pub array_index: usize,
    pub ndims: usize,
    pub domain: Rect,
    pub per_block: Option<Box<Transform>>,
    pub transform: Affine,
}
