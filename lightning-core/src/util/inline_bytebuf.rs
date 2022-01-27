use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::{self, Debug};
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;

const INLINE_CAPACITY: usize = 16;

#[derive(Copy, Clone)]
struct DataHeap {
    capacity: usize,
    ptr: NonNull<u8>,
}

union Data {
    inline: [u8; INLINE_CAPACITY],
    heap: DataHeap,
}

unsafe impl Send for InlineByteBuf {}
unsafe impl Sync for InlineByteBuf {}

pub struct InlineByteBuf {
    len: usize,
    data: Data,
}

impl InlineByteBuf {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_inline(&self) -> bool {
        self.len <= INLINE_CAPACITY
    }

    pub fn as_ptr(&self) -> *const u8 {
        unsafe {
            if self.is_inline() {
                self.data.inline.as_ptr()
            } else {
                self.data.heap.ptr.as_ptr()
            }
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        unsafe {
            if self.is_inline() {
                self.data.inline.as_mut_ptr()
            } else {
                self.data.heap.ptr.as_ptr()
            }
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }
}

impl Drop for InlineByteBuf {
    fn drop(&mut self) {
        if !self.is_inline() {
            unsafe {
                Vec::from_raw_parts(
                    self.data.heap.ptr.as_ptr(),
                    self.len,
                    self.data.heap.capacity,
                );
            }
        }
    }
}

impl From<&[u8]> for InlineByteBuf {
    fn from(buffer: &[u8]) -> Self {
        let len = buffer.len();
        let data = if len <= INLINE_CAPACITY {
            let mut inline = [0; INLINE_CAPACITY];
            inline[..len].copy_from_slice(buffer);

            Data { inline }
        } else {
            let mut buffer = ManuallyDrop::new(Vec::<u8>::from(buffer));

            Data {
                heap: DataHeap {
                    capacity: buffer.capacity(),
                    ptr: unsafe { NonNull::new_unchecked(buffer.as_mut_ptr()) },
                },
            }
        };

        Self { len, data }
    }
}

impl From<Vec<u8>> for InlineByteBuf {
    fn from(buffer: Vec<u8>) -> Self {
        let len = buffer.len();
        let data = if len <= INLINE_CAPACITY {
            let mut inline = [0; INLINE_CAPACITY];
            inline[..len].copy_from_slice(&buffer);

            Data { inline }
        } else {
            let mut buffer = ManuallyDrop::new(buffer);

            Data {
                heap: DataHeap {
                    capacity: buffer.capacity(),
                    ptr: unsafe { NonNull::new_unchecked(buffer.as_mut_ptr()) },
                },
            }
        };

        Self { len, data }
    }
}

impl From<Box<[u8]>> for InlineByteBuf {
    fn from(buffer: Box<[u8]>) -> Self {
        buffer.into_vec().into()
    }
}

impl Clone for InlineByteBuf {
    fn clone(&self) -> Self {
        Self::from(self.as_slice())
    }
}

impl Debug for InlineByteBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut l = f.debug_list();
        let mut count = 0;

        for v in self.as_slice() {
            l.entry(&v);

            count += 1;
            if count > 64 {
                l.entry(&format_args!("... {} bytes in total", self.len()));
                break;
            }
        }

        l.finish()
    }
}

impl Deref for InlineByteBuf {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl DerefMut for InlineByteBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl Serialize for InlineByteBuf {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(self.as_slice())
    }
}

impl<'de> Deserialize<'de> for InlineByteBuf {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;

        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = InlineByteBuf;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "a byte buffer")
            }

            fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(InlineByteBuf::from(v))
            }

            fn visit_borrowed_bytes<E>(self, v: &'de [u8]) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(InlineByteBuf::from(v))
            }

            fn visit_byte_buf<E>(self, v: Vec<u8>) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(InlineByteBuf::from(v))
            }
        }

        deserializer.deserialize_bytes(Visitor)
    }
}
