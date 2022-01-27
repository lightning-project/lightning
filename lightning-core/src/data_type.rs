//! Defines primitive data types and container to hold arbitrary values.

use crate::prelude::*;
use either::Either;
use lazy_static::lazy_static;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::alloc::{Layout, LayoutError};
use std::any::Any;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::os::raw::*;
use std::{mem, str};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DataType(usize);

#[derive(
    IntoPrimitive, TryFromPrimitive, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug,
)]
#[repr(usize)]
pub enum PrimitiveType {
    // Signed integers
    I8,
    I16,
    I32,
    I64,

    // Unsigned integers
    U8,
    U16,
    U32,
    U64,

    // Floats
    F32,
    F64,

    // Vector types
    I8x2,
    I8x3,
    I8x4,
    U8x2,
    U8x3,
    U8x4,

    I16x2,
    I16x3,
    I16x4,
    U16x2,
    U16x3,
    U16x4,

    I32x2,
    I32x3,
    I32x4,
    U32x2,
    U32x3,
    U32x4,

    I64x2,
    I64x3,
    I64x4,
    U64x2,
    U64x3,
    U64x4,

    F32x2,
    F32x3,
    F32x4,

    F64x2,
    F64x3,
    F64x4,
}

#[derive(Debug, Serialize, Deserialize, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComplexType {
    #[doc(hidden)]
    KeyValue { key: DataType, value: DataType },

    #[doc(hidden)]
    Custom {
        size: usize,
        alignment: usize,
        name: String,
    },
}

impl DataType {
    pub fn of<T: HasDataType>() -> Self {
        T::data_type()
    }

    /// Create custom data type with the given `alignment` and `size`. The `name` should correspond
    /// to the type name used in the C source code.
    ///
    /// There are several restrictions:
    /// - `size` cannot exceeds 255 bytes.
    /// - `alignment` must be power of two.
    ///
    /// # Example
    /// ```
    /// # use lightning_core::*;
    /// // The DataType that corresponds to this C struct:
    /// //   struct Point2D {
    /// //     float x;
    /// //     float y;
    /// //   };
    /// let point2d = DataType::custom(
    ///     "Point2D",
    ///     DTYPE_FLOAT.alignment(),
    ///     DTYPE_FLOAT.size_in_bytes() * 2,
    /// ).unwrap();
    /// ```
    pub fn custom(name: &str, alignment: usize, size: usize) -> Result<Self, LayoutError> {
        let layout = Layout::from_size_align(size, alignment)?.pad_to_align();

        Ok(Self::from_complex(ComplexType::Custom {
            name: name.to_string(),
            size: layout.size(),
            alignment: layout.align(),
        }))
    }

    pub fn key_value_pair(key: DataType, value: DataType) -> Self {
        Self::from_complex(ComplexType::KeyValue { key, value })
    }

    pub fn index_value_pair(dtype: Self) -> Self {
        Self::key_value_pair(dtype, DTYPE_U64)
    }

    pub fn array_name(&self, ndims: usize) -> String {
        let mut result = self.name().to_string();
        for _i in 0..ndims {
            result.push_str("[]");
        }
        result
    }

    pub fn name(&self) -> Cow<'static, str> {
        match self.unpack() {
            Either::Left(x) => Cow::Borrowed(x.name()),
            Either::Right(x) => Cow::Owned(x.name()),
        }
    }

    pub fn ctype(&self) -> Cow<'static, str> {
        match self.unpack() {
            Either::Left(x) => Cow::Borrowed(x.ctype()),
            Either::Right(x) => Cow::Owned(x.ctype()),
        }
    }

    pub fn layout(&self) -> Layout {
        match self.unpack() {
            Either::Left(x) => x.layout(),
            Either::Right(x) => x.layout(),
        }
    }

    /// Size of this data type in bytes.
    pub fn size_in_bytes(&self) -> usize {
        self.layout().size()
    }

    /// Alignment of this data type in bytes.
    pub fn alignment(&self) -> usize {
        self.layout().align()
    }

    fn unpack(&self) -> Either<PrimitiveType, &'static ComplexType> {
        if let Ok(x) = PrimitiveType::try_from(self.0) {
            Either::Left(x)
        } else {
            Either::Right(unsafe { mem::transmute(self.0) })
        }
    }

    pub fn is_primitive(&self) -> bool {
        self.unpack().is_left()
    }

    pub fn is_complex(&self) -> bool {
        self.unpack().is_right()
    }

    pub fn from_primitive(dtype: PrimitiveType) -> Self {
        Self(dtype as usize)
    }

    pub fn from_complex(dtype: ComplexType) -> Self {
        let dtype = dtype.intern();
        assert!(PrimitiveType::try_from(dtype as *const _ as usize).is_err());

        Self(dtype as *const _ as usize)
    }

    pub fn to_primitive(&self) -> Option<PrimitiveType> {
        PrimitiveType::try_from(self.0).ok()
    }

    pub fn to_complex(&self) -> Option<&'static ComplexType> {
        if self.is_complex() {
            Some(unsafe { mem::transmute(self.0) })
        } else {
            None
        }
    }
}

impl PrimitiveType {
    /// User-facing name of type. Used in errors and debug messages.
    ///
    /// # Examples
    /// ```
    /// # use lightning_core::*;
    /// assert_eq!(DTYPE_I8.name(), "int8");
    /// assert_eq!(DTYPE_U32.name(), "uint32");
    /// ```
    pub fn name(&self) -> &'static str {
        use PrimitiveType::*;
        match self {
            I8 => "int8",
            I16 => "int16",
            I32 => "int32",
            I64 => "int64",
            U8 => "uint8",
            U16 => "uint16",
            U32 => "uint32",
            U64 => "uint64",
            F32 => "float32",
            F64 => "float64",
            I8x2 => "int8x2",
            I8x3 => "int8x3",
            I8x4 => "int8x4",
            U8x2 => "uint8x2",
            U8x3 => "uint8x3",
            U8x4 => "uint8x4",
            I16x2 => "int16x2",
            I16x3 => "int16x3",
            I16x4 => "int16x4",
            U16x2 => "uint16x2",
            U16x3 => "uint16x3",
            U16x4 => "uint16x4",
            I32x2 => "int32x2",
            I32x3 => "int32x3",
            I32x4 => "int32x4",
            U32x2 => "uint32x2",
            U32x3 => "uint32x3",
            U32x4 => "uint32x4",
            I64x2 => "int64x2",
            I64x3 => "int64x3",
            I64x4 => "int64x4",
            U64x2 => "uint64x2",
            U64x3 => "uint64x3",
            U64x4 => "uint64x4",
            F32x2 => "float32x2",
            F32x3 => "float32x3",
            F32x4 => "float32x4",
            F64x2 => "float64x2",
            F64x3 => "float64x3",
            F64x4 => "float64x4",
        }
    }

    /// Type name used in C source code.
    ///
    /// # Examples
    /// ```
    /// # use lightning_core::*;
    /// assert_eq!(DTYPE_I8.ctype(), "int8_t");
    /// assert_eq!(DTYPE_F32.ctype(), "float");
    /// ```
    pub fn ctype(&self) -> &'static str {
        use PrimitiveType::*;

        match self {
            I8 => "int8_t",
            I16 => "int16_t",
            I32 => "int32_t",
            I64 => "int64_t",
            U8 => "uint8_t",
            U16 => "uint16_t",
            U32 => "uint32_t",
            U64 => "uint64_t",
            F32 => "float",
            F64 => "double",
            I8x2 => "char2",
            I8x3 => "char3",
            I8x4 => "char4",
            U8x2 => "uchar2",
            U8x3 => "uchar3",
            U8x4 => "uchar4",
            I16x2 => "short2",
            I16x3 => "short1",
            I16x4 => "short4",
            U16x2 => "ushort2",
            U16x3 => "ushort3",
            U16x4 => "ushort4",
            I32x2 => "int2",
            I32x3 => "int3",
            I32x4 => "int4",
            U32x2 => "uint2",
            U32x3 => "uint3",
            U32x4 => "uint4",
            I64x2 => "longlong2",
            I64x3 => "longlong3",
            I64x4 => "longlong4",
            U64x2 => "ulonglong2",
            U64x3 => "ulonglong3",
            U64x4 => "ulonglong4",
            F32x2 => "float2",
            F32x3 => "float3",
            F32x4 => "float4",
            F64x2 => "double2",
            F64x3 => "double3",
            F64x4 => "double4",
        }
    }

    #[inline(always)]
    pub fn layout(&self) -> Layout {
        use PrimitiveType::*;

        // Size and alignments taken from the CUDA programming guide.
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types
        match self {
            I8 | U8 => Layout::new::<i8>(),
            I16 | U16 => Layout::new::<i16>(),
            I32 | U32 => Layout::new::<i32>(),
            I64 | U64 => Layout::new::<i64>(),
            F32 => Layout::new::<f32>(),
            F64 => Layout::new::<f64>(),
            I8x2 | U8x2 => unsafe { Layout::from_size_align_unchecked(2, 2) },
            I8x3 | U8x3 => unsafe { Layout::from_size_align_unchecked(3, 1) },
            I8x4 | U8x4 => unsafe { Layout::from_size_align_unchecked(4, 4) },
            I16x2 | U16x2 => unsafe { Layout::from_size_align_unchecked(4, 4) },
            I16x3 | U16x3 => unsafe { Layout::from_size_align_unchecked(6, 2) },
            I16x4 | U16x4 => unsafe { Layout::from_size_align_unchecked(8, 8) },
            I32x2 | U32x2 | F32x2 => unsafe { Layout::from_size_align_unchecked(8, 8) },
            I32x3 | U32x3 | F32x3 => unsafe { Layout::from_size_align_unchecked(12, 4) },
            I32x4 | U32x4 | F32x4 => unsafe { Layout::from_size_align_unchecked(16, 16) },
            I64x2 | U64x2 | F64x2 => unsafe { Layout::from_size_align_unchecked(16, 16) },
            I64x3 | U64x3 | F64x3 => unsafe { Layout::from_size_align_unchecked(24, 8) },
            I64x4 | U64x4 | F64x4 => unsafe { Layout::from_size_align_unchecked(32, 16) },
        }
    }
}

impl ComplexType {
    fn intern(self) -> &'static Self {
        lazy_static! {
            static ref INTERNER: RwLock<HashSet<&'static ComplexType>> = RwLock::default();
        };

        if let Some(ptr) = INTERNER.read().get(&self) {
            return ptr;
        }

        let mut guard = INTERNER.write();
        if let Some(ptr) = guard.get(&self) {
            return ptr;
        }

        let kind = Box::leak(Box::new(self));
        guard.insert(kind);
        kind
    }

    pub fn name(&self) -> String {
        self.ctype()
    }

    pub fn ctype(&self) -> String {
        use ComplexType::*;

        match self {
            KeyValue { key, value } => {
                return format!(
                    "{NS}::KeyValuePair<{}, {}>",
                    key.ctype(),
                    value.ctype(),
                    NS = "::lightning",
                );
            }
            Custom { name, .. } => name.to_string(),
        }
    }

    pub fn layout(&self) -> Layout {
        use ComplexType::*;

        match self {
            KeyValue { key, value } => Layout::extend(&key.layout(), value.layout()).unwrap().0,
            &Custom {
                size, alignment, ..
            } => Layout::from_size_align(size, alignment).unwrap(),
        }
    }
}

impl serde::Serialize for DataType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let result: Either<PrimitiveType, &'static ComplexType> = self.unpack();
        result.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for DataType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let result = Either::<PrimitiveType, ComplexType>::deserialize(deserializer)?;

        Ok(match result {
            Either::Left(x) => DataType::from_primitive(x),
            Either::Right(x) => DataType::from_complex(x),
        })
    }
}

impl fmt::Debug for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.unpack() {
            Either::Left(x) => Debug::fmt(&x, f),
            Either::Right(x) => Debug::fmt(x, f),
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

/// Alias for [`DataType::new(DataTypeKind::I8)`].
pub const DTYPE_I8: DataType = DataType(PrimitiveType::I8 as usize);
/// Alias for [`DataType::new(DataTypeKind::I16)`].
pub const DTYPE_I16: DataType = DataType(PrimitiveType::I16 as usize);
/// Alias for [`DataType::new(DataTypeKind::I32)`].
pub const DTYPE_I32: DataType = DataType(PrimitiveType::I32 as usize);
/// Alias for [`DataType::new(DataTypeKind::I64)`].
pub const DTYPE_I64: DataType = DataType(PrimitiveType::I64 as usize);
/// Alias for [`DataType::new(DataTypeKind::U8)`].
pub const DTYPE_U8: DataType = DataType(PrimitiveType::U8 as usize);
/// Alias for [`DataType::new(DataTypeKind::U16)`].
pub const DTYPE_U16: DataType = DataType(PrimitiveType::U16 as usize);
/// Alias for [`DataType::new(DataTypeKind::U32)`].
pub const DTYPE_U32: DataType = DataType(PrimitiveType::U32 as usize);
/// Alias for [`DataType::new(DataTypeKind::U64)`].
pub const DTYPE_U64: DataType = DataType(PrimitiveType::U64 as usize);
/// Alias for [`DataType::new(DataTypeKind::F32)`].
pub const DTYPE_F32: DataType = DataType(PrimitiveType::F32 as usize);
/// Alias for [`DataType::new(DataTypeKind::F64)`].
pub const DTYPE_F64: DataType = DataType(PrimitiveType::F64 as usize);

/// Alias for [`DataType::new(DataTypeKind::U8)`] (`char` in C).
pub const DTYPE_BYTE: DataType = DTYPE_U8;
/// Alias for [`DataType::new(DataTypeKind::I16)`] (`short` in C).
pub const DTYPE_SHORT: DataType = DTYPE_I16;
/// Alias for [`DataType::new(DataTypeKind::I32)`] (`int` in C).
pub const DTYPE_INT: DataType = DTYPE_I32;
/// Alias for [`DataType::new(DataTypeKind::I64)`] (`long long` in C).
pub const DTYPE_LONG: DataType = DTYPE_I64;
/// Alias for [`DataType::new(DataTypeKind::F32)`] (`float` in C).
pub const DTYPE_FLOAT: DataType = DTYPE_F32;
/// Alias for [`DataType::new(DataTypeKind::F64)`] (`double` in C).
pub const DTYPE_DOUBLE: DataType = DTYPE_F64;

#[cfg(target_pointer_width = "64")]
/// Alias for `size_t` in C.
pub static DTYPE_SIZE_T: DataType = DTYPE_U64;
#[cfg(target_pointer_width = "32")]
/// Alias for `size_t` in C.
pub static DTYPE_SIZE: DataType = DTYPE_U32;

/// Alias for `float2` in CUDA.
pub const DTYPE_FLOAT2: DataType = DataType(PrimitiveType::F32x2 as usize);
/// Alias for `float3` in CUDA.
pub const DTYPE_FLOAT3: DataType = DataType(PrimitiveType::F32x3 as usize);
/// Alias for `float4` in CUDA.
pub const DTYPE_FLOAT4: DataType = DataType(PrimitiveType::F32x4 as usize);
/// Alias for `double2` in CUDA.
pub const DTYPE_DOUBLE2: DataType = DataType(PrimitiveType::F64x2 as usize);

/// Alias for `short` in C.
#[allow(non_camel_case_types)]
pub type short = c_short;
/// Alias for `int` in C.
#[allow(non_camel_case_types)]
pub type int = c_int;
/// Alias for `long` in C.
#[allow(non_camel_case_types)]
pub type long = c_long;
/// Alias for `float` in C.
#[allow(non_camel_case_types)]
pub type float = c_float;
/// Alias for `double` in C.
#[allow(non_camel_case_types)]
pub type double = c_double;
/// Alias for `float2` in C.
#[allow(non_camel_case_types)]
pub type float2 = [c_float; 2];
/// Alias for `double2` in C.
#[allow(non_camel_case_types)]
pub type double2 = [c_double; 2];
/// Alias for `float4` in C.
#[allow(non_camel_case_types)]
pub type float4 = [c_float; 4];
/// Alias for `double4` in C.
#[allow(non_camel_case_types)]
pub type double4 = [c_double; 4];

/// Pair of index and value V. Ordering is by value.
pub type IndexValuePair<V> = KeyValuePair<u64, V>;

/// Pair of some key K and value V. Ordering is by value.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct KeyValuePair<K, V> {
    pub key: K,
    pub value: V,
}

impl<K: PartialEq, V: PartialOrd> PartialOrd for KeyValuePair<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.value, &other.value)
    }
}

impl<K: HasDataType, V: HasDataType> HasDataType for KeyValuePair<K, V> {
    fn data_type() -> DataType {
        DataType::key_value_pair(K::data_type(), V::data_type())
    }
}

impl<K: HasDataType, V: HasDataType> From<KeyValuePair<K, V>> for DataValue {
    fn from(this: KeyValuePair<K, V>) -> DataValue {
        DataValue::from(&this)
    }
}

impl<K: HasDataType, V: HasDataType> From<&KeyValuePair<K, V>> for DataValue {
    fn from(this: &KeyValuePair<K, V>) -> DataValue {
        use std::mem::size_of;
        use std::slice::from_raw_parts;

        Self {
            dtype: KeyValuePair::<K, V>::data_type(),
            data: Box::from(unsafe {
                from_raw_parts(
                    this as *const KeyValuePair<K, V> as *const u8,
                    size_of::<KeyValuePair<K, V>>(),
                )
            }),
        }
    }
}

impl<K: HasDataType, V: HasDataType> TryFrom<DataValue> for KeyValuePair<K, V> {
    type Error = CastError;

    fn try_from(this: DataValue) -> Result<Self, CastError> {
        <Self as TryFrom<&DataValue>>::try_from(&this)
    }
}

impl<K: HasDataType, V: HasDataType> TryFrom<&DataValue> for KeyValuePair<K, V> {
    type Error = CastError;

    fn try_from(this: &DataValue) -> Result<Self, CastError> {
        use std::mem::size_of;
        use std::ptr::read_unaligned;

        if this.dtype == Self::data_type() && this.data.len() == size_of::<Self>() {
            Ok(unsafe { read_unaligned(this.data.as_ptr() as *const Self) })
        } else {
            Err(CastError)
        }
    }
}

/// Types which correspond to [`DataType`] variants.
///
/// These type should satisify the following requirements.
/// - [`Send`] + [`Sync`]: thread safe.
/// - [`Copy`]: Trivial copyable.
/// - [`Into<DataValue>`]: Can be converted into [`DataValue`].
/// - [`TryFrom<DataValue>`]: Can be converted from a [`DataValue`].
/// - [`Any`]: Allows for dynamic casting.
pub trait HasDataType:
    Send + Sync + Copy + Default + Debug + Into<DataValue> + TryFrom<DataValue, Error = CastError> + Any
{
    fn data_type() -> DataType;
}

/// Type-erased value for a given [`DataType`].
///
/// Use any of the many `From` impls to convert a value into [`DataValue`] and use many of the
/// many `TryFrom` impls to convert back into a value.
///
/// # Example
/// ```
/// # use lightning_core::*;
/// # use std::convert::TryInto;
/// let x: i32 = 123;
///
/// let y = DataValue::from(&x);
///
/// // Data type should be i32.
/// assert_eq!(y.data_type(), DTYPE_I32);
///
/// // Data is internally stored in native byte order.
/// assert_eq!(y.as_raw_data(), &x.to_ne_bytes());
///
/// // Convert back to i32 using `TryInto`.
/// let z: i32 = y.try_into().unwrap();
/// assert_eq!(x, z);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct DataValue {
    dtype: DataType,
    data: Box<[u8]>,
}

impl DataValue {
    /// Returns [`DataType`] of value.
    pub fn data_type(&self) -> DataType {
        self.dtype
    }

    pub fn into_raw_data(self) -> Box<[u8]> {
        assert_eq!(self.dtype.size_in_bytes(), self.data.len());
        self.data
    }

    pub fn as_raw_data(&self) -> &[u8] {
        assert_eq!(self.dtype.size_in_bytes(), self.data.len());
        &self.data
    }

    pub fn from_raw_data(data: &[u8], dtype: DataType) -> Self {
        assert_eq!(dtype.size_in_bytes(), data.len());

        Self {
            dtype,
            data: Box::from(data),
        }
    }

    pub fn cast(&self, dtype: DataType) -> Result<Self, CastError> {
        if self.data_type() == dtype {
            return Ok(self.clone());
        }

        macro_rules! rules {
            ($x:ident => $($y:ident),*) => {
                if self.dtype == <$x>::data_type() {
                    if let Ok(x) = <$x>::try_from(self) {
                        $(
                            if dtype == <$y>::data_type() {
                                let y: $y = x.try_into().map_err(|_| CastError)?;
                                return Ok(y.into());
                            }
                        )*
                    }
                }
            };
        }

        // TODO: this needs more love. There is no way to cast from float to integer for example
        // even though that would be lossless for certain floats.
        rules!(u8 => u16, u32, u64, i8, i16, i32, i64, f32, f64);
        rules!(u16 => u8, u32, u64, i8, i16, i32, i64, f32, f64);
        rules!(u32 => u8, u16, u64, i8, i16, i32, i64, f64);
        rules!(u64 => u8, u16, u32, i8, i16, i32, i64);
        rules!(i8 => u8, u16, u32, u64, i16, i32, i64, f32, f64);
        rules!(i16 => u8, u16, u32, u64, i8, i32, i64, f32, f64);
        rules!(i32 => u8, u16, u32, u64, i8, i16, i64, f64);
        rules!(i64 => u8, u16, u32, u64, i8, i16, i32);
        rules!(f32 => f64);

        Err(CastError)
    }
}

/// Error indicating failure to cast instance of [`DataValue`] to a rust type.
#[derive(Error, Debug)]
#[error("failed to cast value")]
pub struct CastError;

macro_rules! impl_data_type {
    ($($typ:ty as $name:expr),*) => {
        $(
            impl HasDataType for $typ {
                fn data_type() -> DataType {
                    $name
                }
            }

            impl From<&$typ> for DataValue {
                fn from(this: &$typ) -> DataValue {
                    use std::slice::from_raw_parts;
                    use std::mem::size_of;

                    Self {
                        dtype: <$typ>::data_type(),
                        data: Box::from(unsafe { from_raw_parts(
                            this as *const $typ as *const u8,
                            size_of::<$typ>(),
                        )})
                    }
                }
            }

            impl From<$typ> for DataValue {
                fn from(this: $typ) -> DataValue {
                    <DataValue as From<&$typ>>::from(&this)
                }
            }

            impl TryFrom<&DataValue> for $typ {
                type Error = CastError;

                fn try_from(this: &DataValue) -> Result<$typ, CastError> {
                    use std::mem::size_of;
                    use std::ptr::read_unaligned;

                    if this.dtype == <$typ>::data_type() && this.data.len() == size_of::<$typ>() {
                        Ok(unsafe {
                            read_unaligned(this.data.as_ptr() as *const $typ)
                        })
                    } else {
                        Err(CastError)
                    }
                }
            }

            impl TryFrom<DataValue> for $typ {
                type Error = CastError;

                fn try_from(this: DataValue) -> Result<$typ, CastError> {
                    (&this).try_into()
                }
            }
        )*
    };
}

impl_data_type!(
    i8 as DTYPE_I8,
    i16 as DTYPE_I16,
    i32 as DTYPE_I32,
    i64 as DTYPE_I64,
    u8 as DTYPE_U8,
    u16 as DTYPE_U16,
    u32 as DTYPE_U32,
    u64 as DTYPE_U64,
    f32 as DTYPE_F32,
    f64 as DTYPE_F64,
    [i8; 2] as DataType(PrimitiveType::I8x2 as usize),
    [i8; 3] as DataType(PrimitiveType::I8x3 as usize),
    [i8; 4] as DataType(PrimitiveType::I8x4 as usize),
    [u8; 2] as DataType(PrimitiveType::U8x2 as usize),
    [u8; 3] as DataType(PrimitiveType::U8x3 as usize),
    [u8; 4] as DataType(PrimitiveType::U8x4 as usize),
    [i32; 2] as DataType(PrimitiveType::I32x2 as usize),
    [i32; 3] as DataType(PrimitiveType::I32x3 as usize),
    [i32; 4] as DataType(PrimitiveType::I32x4 as usize),
    [u32; 2] as DataType(PrimitiveType::U32x2 as usize),
    [u32; 3] as DataType(PrimitiveType::U32x3 as usize),
    [u32; 4] as DataType(PrimitiveType::U32x4 as usize),
    [f32; 2] as DataType(PrimitiveType::F32x2 as usize),
    [f32; 3] as DataType(PrimitiveType::F32x3 as usize),
    [f32; 4] as DataType(PrimitiveType::F32x4 as usize),
    [f64; 2] as DataType(PrimitiveType::F64x2 as usize)
);

impl From<&DataValue> for DataValue {
    fn from(this: &DataValue) -> Self {
        this.clone()
    }
}
