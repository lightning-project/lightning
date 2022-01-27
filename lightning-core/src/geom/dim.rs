use super::point::Point;
use super::rect::PointInRectIter;
use super::rect::Rect;
use super::{One, Zero};
use crate::util::array;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::convert::TryInto;
use std::fmt::{self, Debug, Display};
use std::ops::{self};

/// Represents the dimensions of an volume in [`MAX_DIMS`] dimensional space.
///
/// The difference between [`Point`] and [`Extent`] is that [`Point`] represents some arbitrary
/// point in space (which means coordinates could be negative) while [`Extent`] represents the
/// dimensions (i.e., width, height, depth, ...) of some volume (which means the coordinates
/// must be nonnegative, but could be zero).
///
/// This type is essentially a wrapper around `[u64; MAX_DIMS]` while providing many useful
/// operations.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Dim<T, const N: usize> {
    pub dims: [T; N],
}

impl<T, const N: usize> Dim<T, N> {
    #[inline(always)]
    pub fn to_point(self) -> Point<T, N> {
        Point { dims: self.dims }
    }

    /// Returns the extent `[f(0), f(1), f(2), ...]`
    #[inline(always)]
    pub fn gen<F>(f: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        Self {
            dims: array::generate(f),
        }
    }

    /// Returns an empty extent (`[0, 0, 0, ...]`)
    #[inline(always)]
    pub fn empty() -> Self
    where
        T: Zero,
    {
        Self {
            dims: array::generate(|_| T::zero()),
        }
    }

    /// Returns the extent of a single point (`[1, 1, 1, ...]`)
    #[inline(always)]
    pub fn one() -> Self
    where
        T: One,
    {
        Self {
            dims: array::generate(|_| T::one()),
        }
    }

    #[inline]
    fn zip<F>(self, rhs: Self, fun: F) -> Self
    where
        F: FnMut(T, T) -> T,
    {
        Self {
            dims: array::zip(self.dims, rhs.dims, fun),
        }
    }

    #[inline]
    fn try_zip<F>(self, rhs: Self, fun: F) -> Option<Self>
    where
        F: FnMut(T, T) -> Option<T>,
    {
        Some(self.to_point().try_zip(rhs.to_point(), fun)?.to_dim())
    }
}

impl<T: Clone, const N: usize> Dim<T, N> {
    /// Returns the extent `[v, v, v, ...]`
    #[inline(always)]
    pub fn repeat(v: T) -> Self {
        Self {
            dims: array::generate(|_| v.clone()),
        }
    }

    /// Returns the exists where the first values are taken from the given slice, padded with
    /// `1` to reach `N` elements.
    ///
    /// # Panics
    /// If `slice.len() > N`.
    #[inline]
    pub fn from_slice(slice: &[T]) -> Self
    where
        T: One,
    {
        assert!(slice.len() <= N);
        Dim {
            dims: array::generate(|i| {
                if let Some(v) = slice.get(i) {
                    v.clone()
                } else {
                    T::one()
                }
            }),
        }
    }

    #[inline(always)]
    pub fn resize<const M: usize>(self, default: T) -> Dim<T, M> {
        Dim {
            dims: array::generate(|i| {
                if i < N {
                    self.dims[i].clone()
                } else {
                    default.clone()
                }
            }),
        }
    }
}

impl<const N: usize> Dim<u64, N> {
    #[inline]
    pub fn div_ceil(self, rhs: Self) -> Self {
        const fn div_ceil(p: u64, q: u64) -> u64 {
            (p / q) + (if p % q == 0 { 0 } else { 1 })
        }

        self.zip(rhs, div_ceil)
    }

    /// Element-wise division while checking for overflows.
    #[inline]
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        self.try_zip(rhs, u64::checked_div)
    }

    /// Element-wise modulo while checking for overflows.
    #[inline]
    pub fn checked_rem(self, rhs: Self) -> Option<Self> {
        self.try_zip(rhs, u64::checked_rem)
    }

    /// Element-wise addition while checking for overflows.
    #[inline]
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        self.try_zip(rhs, u64::checked_add)
    }

    /// Element-wise multiplication while checking for overflows.
    #[inline]
    pub fn checked_mul(self, rhs: Self) -> Option<Self> {
        self.try_zip(rhs, u64::checked_mul)
    }

    /// Element-wise subtraction while checking for overflows.
    #[inline]
    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        self.try_zip(rhs, u64::checked_sub)
    }

    /// Element-wise subtraction using saturating arithmetic.
    pub fn saturating_sub(self, rhs: Self) -> Self {
        self.zip(rhs, u64::saturating_sub)
    }

    /// Returns the volume of this extent (width x height x depth)
    #[inline(always)]
    pub fn volume(self) -> u64 {
        let mut volume = 1;

        for i in 0..N {
            volume *= self[i];
        }

        volume
    }

    /// Returns true if this extent is empty.
    #[inline(always)]
    pub fn is_empty(self) -> bool {
        let mut empty = false;

        for i in 0..N {
            empty |= self[i] == 0;
        }

        empty
    }

    /// Returns a rectangle where upper bounds are given by this extent and the lower bounds are zero.
    #[inline(always)]
    pub fn to_bounds(self) -> Rect<u64, N> {
        Rect::new(Point::zeros(), self)
    }

    #[inline]
    pub fn to_usize(self) -> Option<[usize; N]> {
        array::try_map(self.dims, |v| v.try_into()).ok()
    }

    #[inline]
    pub fn to_i64(self) -> Option<[i64; N]> {
        array::try_map(self.dims, |v| v.try_into()).ok()
    }
}

impl<T> Dim<T, 1> {
    /// Returns the point `[x, y]`.
    #[inline(always)]
    pub const fn new(x: T) -> Self {
        Self { dims: [x] }
    }
}

impl<T> Dim<T, 2> {
    /// Returns the point `[x, y]`.
    #[inline(always)]
    pub const fn new(x: T, y: T) -> Self {
        Self { dims: [x, y] }
    }
}

impl<T> Dim<T, 3> {
    /// Returns the point `[x, y, z]`.
    #[inline(always)]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { dims: [x, y, z] }
    }

    pub fn xy(self) -> Dim<T, 2> {
        let [x, y, _] = self.dims;
        Dim::<T, 2>::new(x, y)
    }
}

impl<T> Dim<T, 4> {
    /// Returns the point `[x, y, z, v]`.
    #[inline(always)]
    pub const fn new(x: T, y: T, z: T, v: T) -> Self {
        Self { dims: [x, y, z, v] }
    }

    pub fn xyz(self) -> Dim<T, 3> {
        let [x, y, z, _] = self.dims;
        Dim::<T, 3>::new(x, y, z)
    }
}

impl<T, const N: usize> From<Dim<T, N>> for [T; N] {
    fn from(this: Dim<T, N>) -> Self {
        this.dims
    }
}

impl<T, const N: usize> From<[T; N]> for Dim<T, N> {
    fn from(dims: [T; N]) -> Self {
        Self { dims }
    }
}

impl<T: One> From<()> for Dim<T, 1> {
    fn from(_: ()) -> Self {
        Self::one()
    }
}

impl<T> From<T> for Dim<T, 1> {
    fn from(x: T) -> Self {
        Self::new(x)
    }
}

impl<T> From<(T,)> for Dim<T, 1> {
    fn from((x,): (T,)) -> Self {
        Self::new(x)
    }
}

impl<T> From<(T, T)> for Dim<T, 2> {
    fn from((x, y): (T, T)) -> Self {
        Self::new(x, y)
    }
}

impl<T> From<(T, T, T)> for Dim<T, 3> {
    fn from((x, y, z): (T, T, T)) -> Self {
        Self::new(x, y, z)
    }
}

impl<T: One, H> From<H> for Dim<T, 2>
where
    H: Into<Dim<T, 1>>,
{
    fn from(that: H) -> Self {
        let [x] = that.into().dims;
        Self::new(x, T::one())
    }
}

impl<T: One, H> From<H> for Dim<T, 3>
where
    H: Into<Dim<T, 2>>,
{
    fn from(that: H) -> Self {
        let [x, y] = that.into().dims;
        Self::new(x, y, T::one())
    }
}

impl<T: One, H> From<H> for Dim<T, 4>
where
    H: Into<Dim<T, 3>>,
{
    fn from(that: H) -> Self {
        let [x, y, z] = that.into().dims;
        Self::new(x, y, z, T::one())
    }
}

impl<T: Debug, const N: usize> Debug for Dim<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.dims[..], f)
    }
}

impl<T: Debug, const N: usize> Display for Dim<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.dims[..], f)
    }
}

impl<const N: usize> ops::Add for Dim<u64, N> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Dim<u64, N>) -> Self::Output {
        self.zip(rhs, u64::add)
    }
}

impl<const N: usize> ops::Add<Point<u64, N>> for Dim<u64, N> {
    type Output = Point<u64, N>;

    #[inline(always)]
    fn add(self, rhs: Point<u64, N>) -> Self::Output {
        self.to_point() + rhs
    }
}

impl<const N: usize> ops::Add<Dim<u64, N>> for Point<u64, N> {
    type Output = Point<u64, N>;

    #[inline(always)]
    fn add(self, rhs: Dim<u64, N>) -> Self::Output {
        self + rhs.to_point()
    }
}

impl<const N: usize> ops::Sub for Dim<u64, N> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Dim<u64, N>) -> Self::Output {
        self.zip(rhs, u64::sub)
    }
}

impl<const N: usize> ops::Sub<Point<u64, N>> for Dim<u64, N> {
    type Output = Point<u64, N>;

    #[inline(always)]
    fn sub(self, rhs: Point<u64, N>) -> Self::Output {
        self.to_point() - rhs
    }
}

impl<const N: usize> ops::Sub<Dim<u64, N>> for Point<u64, N> {
    type Output = Point<u64, N>;

    #[inline(always)]
    fn sub(self, rhs: Dim<u64, N>) -> Self::Output {
        self - rhs.to_point()
    }
}

impl<const N: usize> ops::Mul for Dim<u64, N> {
    type Output = Self;

    fn mul(self, rhs: Dim<u64, N>) -> Self::Output {
        self.zip(rhs, u64::mul)
    }
}

impl<const N: usize> ops::Mul<u64> for Dim<u64, N> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: u64) -> Self::Output {
        self * Dim::<u64, N>::repeat(rhs)
    }
}

impl<const N: usize> ops::Mul<Dim<u64, N>> for u64 {
    type Output = Dim<u64, N>;

    #[inline(always)]
    fn mul(self, rhs: Dim<u64, N>) -> Self::Output {
        rhs * Dim::<u64, N>::repeat(self)
    }
}

impl<const N: usize> ops::Mul<Dim<u64, N>> for Point<u64, N> {
    type Output = Point<u64, N>;

    #[inline(always)]
    fn mul(self, rhs: Dim<u64, N>) -> Self::Output {
        self * rhs.to_point()
    }
}

impl<const N: usize> ops::Mul<Point<u64, N>> for Dim<u64, N> {
    type Output = Point<u64, N>;

    #[inline(always)]
    fn mul(self, rhs: Point<u64, N>) -> Self::Output {
        self.to_point() * rhs
    }
}

impl<const N: usize> ops::Div<Dim<u64, N>> for Dim<u64, N> {
    type Output = Dim<u64, N>;

    #[inline(always)]
    fn div(self, rhs: Dim<u64, N>) -> Self::Output {
        self.zip(rhs, u64::div)
    }
}

impl<const N: usize> ops::Div<Dim<u64, N>> for Point<u64, N> {
    type Output = Point<u64, N>;

    #[inline(always)]
    fn div(self, rhs: Dim<u64, N>) -> Self::Output {
        (self.to_dim() / rhs).to_point()
    }
}

impl<const N: usize> ops::Div<u64> for Dim<u64, N> {
    type Output = Dim<u64, N>;

    #[inline(always)]
    fn div(self, rhs: u64) -> Self::Output {
        self / Dim::repeat(rhs)
    }
}

impl<const N: usize> ops::Div<Dim<u64, N>> for u64 {
    type Output = Dim<u64, N>;

    #[inline(always)]
    fn div(self, rhs: Dim<u64, N>) -> Self::Output {
        Dim::repeat(self) / rhs
    }
}

impl<const N: usize> ops::Rem<Dim<u64, N>> for Dim<u64, N> {
    type Output = Dim<u64, N>;

    #[inline(always)]
    fn rem(self, rhs: Dim<u64, N>) -> Self::Output {
        self.zip(rhs, u64::rem)
    }
}

impl<const N: usize> ops::Rem<Dim<u64, N>> for Point<u64, N> {
    type Output = Point<u64, N>;

    #[inline(always)]
    fn rem(self, rhs: Dim<u64, N>) -> Self::Output {
        (self.to_dim() % rhs).to_point()
    }
}

impl<const N: usize> ops::Rem<u64> for Dim<u64, N> {
    type Output = Dim<u64, N>;

    #[inline(always)]
    fn rem(self, rhs: u64) -> Self::Output {
        self % Dim::repeat(rhs)
    }
}

impl<const N: usize> ops::Rem<Dim<u64, N>> for u64 {
    type Output = Dim<u64, N>;

    #[inline(always)]
    fn rem(self, rhs: Dim<u64, N>) -> Self::Output {
        Dim::repeat(self) % rhs
    }
}

impl<T, const N: usize> ops::Deref for Dim<T, N> {
    type Target = [T; N];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.dims
    }
}

impl<T, const N: usize> ops::DerefMut for Dim<T, N> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.dims
    }
}

impl<const N: usize> IntoIterator for Dim<u64, N> {
    type Item = Point<u64, N>;
    type IntoIter = PointInRectIter<N>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.to_bounds().into_iter()
    }
}

impl<T: Serialize + Copy, const N: usize> Serialize for Dim<T, N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_point().serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for Dim<T, N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Point::deserialize(deserializer).map(|p| p.to_dim())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_i64() {
        let x = Dim::<u64, 3>::new(1, 2, 3);
        assert_eq!(x.to_i64(), Some([1, 2, 3]));
    }
}
