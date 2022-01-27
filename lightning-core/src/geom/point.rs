use super::dim::Dim;
use super::rect::Rect;
use super::{One, Zero};
use crate::prelude::*;
use crate::util::array;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::convert::TryInto;
use std::fmt::{self, Debug, Display};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{self, Deref, DerefMut};

/// Multi-dimensional index point having [`MAX_DIMS`] dimensions.
///
/// Note that indices can be negative. This type is essentially a wrapper around `[u64; MAX_DIMS]`
/// while providing many useful operations.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Point<T, const N: usize> {
    pub dims: [T; N],
}

impl<T, const N: usize> Point<T, N> {
    /// Returns the point `[f(0), f(1), f(2), ...]`
    #[inline(always)]
    pub fn gen<F>(f: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        Self {
            dims: array::generate(f),
        }
    }

    /// Interpret this point as an [`Extent`]
    #[inline(always)]
    pub fn to_dim(self) -> Dim<T, N> {
        Dim { dims: self.dims }
    }
    /// Returns the point `[0, 0, 0, ...]`
    #[inline(always)]
    pub fn zeros() -> Self
    where
        T: Zero,
    {
        Self {
            dims: array::generate(|_| T::zero()),
        }
    }

    /// Returns the point `[1, 1, 1, ...]`
    #[inline(always)]
    pub fn ones() -> Self
    where
        T: One,
    {
        Self {
            dims: array::generate(|_| T::one()),
        }
    }

    #[inline(always)]
    pub fn zip<F>(self, rhs: Self, fun: F) -> Self
    where
        F: FnMut(T, T) -> T,
    {
        Self {
            dims: array::zip(self.dims, rhs.dims, fun),
        }
    }

    #[inline(always)]
    pub fn try_zip<F>(self, rhs: Self, mut fun: F) -> Option<Self>
    where
        F: FnMut(T, T) -> Option<T>,
    {
        struct Error;

        Some(Self {
            dims: array::try_zip(self.dims, rhs.dims, |a, b| fun(a, b).ok_or(Error)).ok()?,
        })
    }
}

impl<T: Clone, const N: usize> Point<T, N> {
    /// Returns the point `[v, v, v, ...]`
    #[inline(always)]
    pub fn repeat(v: T) -> Self {
        Self::gen(|_| v.clone())
    }

    pub fn resize<const M: usize>(self, default: T) -> Point<T, M> {
        self.to_dim().resize::<M>(default).to_point()
    }
}

impl<T: PartialOrd, const N: usize> Point<T, N> {
    pub fn element_min(self, rhs: Self) -> Self {
        Self::zip(self, rhs, |a, b| if a < b { a } else { b })
    }

    pub fn element_max(self, rhs: Self) -> Self {
        Self::zip(self, rhs, |a, b| if a > b { a } else { b })
    }
}

impl<const N: usize> Point<u64, N> {
    /// Add two points using saturating arithmetic.
    pub fn saturating_sub(self, rhs: Self) -> Self {
        self.zip(rhs, u64::saturating_sub)
    }

    /// Add two points using saturating arithmetic.
    pub fn saturating_add(self, rhs: Self) -> Self {
        self.zip(rhs, u64::saturating_add)
    }

    /// Element-wise addition while checking for overflows.
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        self.try_zip(rhs, u64::checked_add)
    }

    /// Element-wise multiplication while checking for overflows.
    pub fn checked_mul(self, rhs: Self) -> Option<Self> {
        self.try_zip(rhs, u64::checked_mul)
    }

    /// Element-wise subtraction while checking for overflows.
    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        self.try_zip(rhs, u64::checked_sub)
    }

    /// Returns a [`Rect`] where the lower bounds are zero and upper bounds correspond to this
    /// point.
    #[inline(always)]
    pub fn to_bounds(self) -> Rect<u64, N> {
        self.to_dim().to_bounds()
    }

    #[inline(always)] // This will become a noop if usize == u64, so inline to allow optimizations
    pub fn to_usize(self) -> Option<[usize; N]> {
        array::try_map(self.dims, |v| v.try_into()).ok()
    }
}

impl<T> Point<T, 1> {
    /// Returns the point `[x, y]`.
    #[inline(always)]
    pub const fn new(x: T) -> Self {
        Self { dims: [x] }
    }
}

impl<T> Point<T, 2> {
    /// Returns the point `[x, y]`.
    #[inline(always)]
    pub const fn new(x: T, y: T) -> Self {
        Self { dims: [x, y] }
    }
}

impl<T> Point<T, 3> {
    /// Returns the point `[x, y, z]`.
    #[inline(always)]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { dims: [x, y, z] }
    }

    pub fn x(self) -> Point<T, 1> {
        let [x, _, _] = self.dims;
        Point::<T, 1>::new(x)
    }

    pub fn xy(self) -> Point<T, 2> {
        let [x, y, _] = self.dims;
        Point::<T, 2>::new(x, y)
    }
}

impl<T, const N: usize> From<Point<T, N>> for [T; N] {
    fn from(this: Point<T, N>) -> Self {
        this.dims
    }
}

impl<T, const N: usize> From<[T; N]> for Point<T, N> {
    fn from(dims: [T; N]) -> Self {
        Self { dims }
    }
}

impl<T: Zero> From<()> for Point<T, 1> {
    fn from(_: ()) -> Self {
        Self::zeros()
    }
}

impl<T> From<T> for Point<T, 1> {
    fn from(x: T) -> Self {
        Self::new(x)
    }
}

impl<T> From<(T,)> for Point<T, 1> {
    fn from((x,): (T,)) -> Self {
        Self::new(x)
    }
}

impl<T> From<(T, T)> for Point<T, 2> {
    fn from((x, y): (T, T)) -> Self {
        Self::new(x, y)
    }
}

impl<T> From<(T, T, T)> for Point<T, 3> {
    fn from((x, y, z): (T, T, T)) -> Self {
        Self::new(x, y, z)
    }
}

impl<H, T: Zero> From<H> for Point<T, 2>
where
    H: Into<Point<T, 1>>,
{
    fn from(that: H) -> Self {
        let [x] = that.into().dims;
        Self::new(x, T::zero())
    }
}

impl<H, T: Zero> From<H> for Point<T, 3>
where
    H: Into<Point<T, 2>>,
{
    fn from(that: H) -> Self {
        let [x, y] = that.into().dims;
        Self::new(x, y, T::zero())
    }
}

impl<T: Zero, const N: usize> Default for Point<T, N> {
    fn default() -> Self {
        Self::zeros()
    }
}

impl<const N: usize> ops::Add<Point<u64, N>> for Point<u64, N> {
    type Output = Point<u64, N>;

    #[inline(always)]
    fn add(self, rhs: Point<u64, N>) -> Self::Output {
        self.zip(rhs, u64::add)
    }
}

impl<const N: usize> ops::Sub<Point<u64, N>> for Point<u64, N> {
    type Output = Point<u64, N>;

    #[inline(always)]
    fn sub(self, rhs: Point<u64, N>) -> Self::Output {
        self.zip(rhs, u64::sub)
    }
}

impl<const N: usize> ops::Mul<Point<u64, N>> for Point<u64, N> {
    type Output = Point<u64, N>;

    #[inline(always)]
    fn mul(self, rhs: Point<u64, N>) -> Self::Output {
        self.zip(rhs, u64::mul)
    }
}

impl<T, const N: usize> Deref for Point<T, N> {
    type Target = [T; N];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.dims
    }
}

impl<T, const N: usize> DerefMut for Point<T, N> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.dims
    }
}

impl<T: Debug, const N: usize> Debug for Point<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.dims.fmt(f)
    }
}

impl<T: Debug, const N: usize> Display for Point<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.dims.fmt(f)
    }
}

impl<T: Serialize, const N: usize> Serialize for Point<T, N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeTuple;

        let mut tup = serializer.serialize_tuple(N)?;
        for v in &self.dims {
            tup.serialize_element(&v)?;
        }
        tup.end()
    }
}

impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for Point<T, N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{Error, SeqAccess};

        struct Visitor<T, const N: usize>(PhantomData<fn() -> T>);

        impl<'de, T: Deserialize<'de>, const N: usize> serde::de::Visitor<'de> for Visitor<T, N> {
            type Value = Point<T, N>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(formatter, "an {}-dimensional point", N)
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut items: MaybeUninit<[T; N]> = MaybeUninit::uninit();

                for i in 0..N {
                    if let Some(item) = seq.next_element()? {
                        unsafe {
                            (items.as_mut_ptr() as *mut T).add(i).write(item);
                        }
                    } else {
                        return Err(A::Error::invalid_length(i, &self));
                    }
                }

                let dims = unsafe { items.assume_init() };
                Ok(Point { dims })
            }
        }

        deserializer.deserialize_tuple(N, Visitor(PhantomData))
    }
}
