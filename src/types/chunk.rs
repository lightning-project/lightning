use lightning_core::util::array;
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug};
use std::num::NonZeroU64;

use crate::types::{DataType, Dim, MemoryKind, Strides, WorkerId, MAX_DIMS};

const PREFERRED_ALIGNMENT: usize = 256;

#[derive(Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Hash, PartialOrd, Ord)]
pub struct ChunkId(NonZeroU64);

const CHUNK_ID_BITS: u64 = 56;
impl ChunkId {
    pub(crate) fn new(node_id: WorkerId, index: NonZeroU64) -> Self {
        assert_eq!(index.get() & ((!0) << CHUNK_ID_BITS), 0);
        Self((node_id.0 as u64) << CHUNK_ID_BITS | index)
    }

    pub(crate) fn get(&self) -> u64 {
        self.0.get() & ((1 << CHUNK_ID_BITS) - 1)
    }

    pub fn owner(&self) -> WorkerId {
        WorkerId((self.0.get() >> CHUNK_ID_BITS) as u8)
    }
}

impl Debug for ChunkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ChunkId")
            .field(&format_args!("{}:{}", self.owner(), self.get()))
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct ChunkLayoutBuilder {
    padding_fraction: u8,
    axes: [u8; MAX_DIMS],
}

impl ChunkLayoutBuilder {
    pub fn row_major() -> Self {
        Self::with_order([2, 1, 0])
    }

    pub fn column_major() -> Self {
        Self::with_order([0, 1, 2])
    }

    pub fn with_order(axes: [usize; MAX_DIMS]) -> Self {
        Self::new(axes, 8)
    }

    pub fn new(axes: [usize; MAX_DIMS], padding_fraction: u8) -> Self {
        let mut axes_sorted = axes;
        axes_sorted.sort_unstable();
        assert_eq!(axes_sorted, [0, 1, 2]);

        Self {
            axes: array::generate(|i| axes[i] as u8),
            padding_fraction,
        }
    }

    pub fn axes_order(&self) -> [usize; MAX_DIMS] {
        array::generate(|i| self.axes[i] as usize)
    }

    pub fn build(&self, data_type: DataType, mut size: Dim, affinity: MemoryKind) -> ChunkLayout {
        let elem_size = data_type.layout().pad_to_align().size() as i64;
        let mut alignment = data_type.alignment();
        let mut stride = 1;
        let mut strides = [0; MAX_DIMS];

        if size.is_empty() {
            size = Dim::empty();
        }

        for &i in &self.axes {
            let n = size[i as usize] as i64;
            let max_n = (n * self.padding_fraction as i64) / (self.padding_fraction as i64 - 1);
            let mut current_n = n;
            let mut padded_n = n;

            while alignment < PREFERRED_ALIGNMENT {
                if current_n > max_n {
                    break;
                }

                if stride * elem_size * current_n % (2 * alignment as i64) == 0 {
                    padded_n = current_n;
                    alignment *= 2;
                }

                current_n += 1;
            }

            strides[i as usize] = stride;
            stride *= padded_n;
        }

        ChunkLayout {
            size,
            strides: Strides::from(strides),
            data_type,
            alignment,
            affinity: Some(affinity),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunkLayout {
    pub size: Dim,
    pub data_type: DataType,
    pub strides: Strides,
    pub alignment: usize,
    pub affinity: Option<MemoryKind>,
}

impl ChunkLayout {
    pub fn to_contiguous(&self, size: Dim) -> Self {
        let mut strides = [0; MAX_DIMS];
        let mut stride = 1;

        for &i in &self.strides.order() {
            strides[i] = stride;
            stride *= size[i] as i64;
        }

        Self {
            size,
            strides: Strides::from(strides),
            data_type: self.data_type,
            alignment: self.alignment,
            affinity: self.affinity,
        }
    }

    pub fn size_in_bytes(&self) -> Option<usize> {
        let elem_align = self.data_type.alignment();
        let mut line = 1;

        if self.alignment % elem_align != 0 {
            return None; // Invalid alignment
        }

        for &i in &self.strides.order() {
            if self.size[i] == 0 {
                return Some(0);
            }

            if self.size[i] == 1 {
                continue;
            }

            // TODO: Overflow check for cast?
            if self.strides[i] < line as i64 {
                return None;
            }

            // TODO: Overflow check for mult?
            line = self.strides[i] as usize * self.size[i] as usize;
        }

        Some(line * elem_align)
    }
}
