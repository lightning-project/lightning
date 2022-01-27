use super::dim::Dim;
use super::point::Point;
use super::MAX_DIMS;
use crate::geom::{One, Zero};
use crate::prelude::{reversed, TryInto};
use serde::{Deserialize, Serialize};
use std::cmp::{max, min};
use std::fmt::{self, Debug};
use std::iter::FusedIterator;
use std::ops::Range;

/// Axis aligned bounding box having `MAX_DIMS` dimensions.
///
/// This box is defined by a [lower bound](Rect::low()) (inclusive) and an
/// [upper bound](Rect::high()) (exclusive) along each axis. The upper bound is exclusive to
/// allow zero-volume boxed to be defined as the lower bound equalling the upper bound.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Rect<T, const N: usize> {
    pub lo: Point<T, N>,
    pub hi: Point<T, N>,
}

impl<const N: usize> Rect<u64, N> {
    /// Returns a rectangle having lower bounds `offset` and upper bounds `offset + extents`.
    #[inline(always)]
    pub fn new(offset: Point<u64, N>, extents: Dim<u64, N>) -> Self {
        Self::from_bounds(offset, offset + extents)
    }

    /// Returns a rectangle having lower bounds `lo` and upper bounds `hi`.
    ///
    /// # Panics
    /// Panics if `lo[i] > hi[i]` for any `i`
    pub fn from_bounds(lo: Point<u64, N>, hi: Point<u64, N>) -> Self {
        assert!({
            let mut valid = true;

            for i in 0..N {
                valid &= lo[i] <= hi[i];
            }

            valid
        });

        Self { lo, hi }
    }

    ///
    pub fn from_point(p: Point<u64, N>) -> Self {
        Self::new(p, Dim::one())
    }

    #[inline(always)]
    pub fn offset_add(self, p: Point<u64, N>) -> Self {
        Self::from_bounds(self.lo + p, self.hi + p)
    }

    #[inline(always)]
    pub fn offset_sub(self, p: Point<u64, N>) -> Self {
        Self::from_bounds(self.lo - p, self.hi - p)
    }
    /// The size along each axis (height, width, depth, etc.).
    #[inline(always)]
    pub fn extents(self) -> Dim<u64, N> {
        let mut dims = [0; N];

        for i in 0..N {
            dims[i] = u64::wrapping_sub(self.hi[i], self.lo[i])
        }

        Dim::from(dims)
    }
}

impl<T, const N: usize> Rect<T, N> {
    /// Returns the lower bounds of this rectangle. Note that the invariant `low[i] <= high[i]` for
    /// all `i` holds.
    #[inline(always)]
    pub fn low(self) -> Point<T, N> {
        self.lo
    }

    /// Returns the upper bounds of this rectangle (exclusive, the point returned by this function
    /// is itself not considered to be inside the rectangle). Note that the invariant
    /// `low[i] <= high[i]` for all `i` holds.
    #[inline(always)]
    pub fn high(self) -> Point<T, N> {
        self.hi
    }

    /// Swap axes `i` and `j`.
    #[inline(always)]
    pub fn swap(mut self, i: usize, j: usize) -> Self {
        self.lo.swap(i, j);
        self.hi.swap(i, j);
        self
    }
}

impl<T: Copy + Zero + One, const N: usize> Rect<T, N> {
    // Resize an N-dimensional rectangle to an M-dimensional rectangle. If `M > N` then `M-N`
    // dimensions will be added each consisting of the range `0..1`. If `M <N`, then last `N-M`
    // dimensions will be discarded.
    pub fn resize<const M: usize>(self) -> Rect<T, M> {
        Rect {
            lo: self.lo.resize(T::zero()),
            hi: self.hi.resize(T::one()),
        }
    }
}

impl<T: Ord + Copy, const N: usize> Rect<T, N> {
    /// Returns `true` if the given point lies within this rectangle (note that the upper bounds of
    /// a rectangle are exclusive and thus not considered to lie within).
    pub fn contains_point(self, p: Point<T, N>) -> bool {
        let (lo, hi) = (self.lo.dims, self.hi.dims);
        let mut contains = true;

        for i in 0..N {
            contains &= (p[i] >= lo[i]) & (p[i] < hi[i]);
        }

        contains
    }

    /// Returns `true` if all points represented by `other` lie within this rectangle.
    pub fn contains(self, other: Rect<T, N>) -> bool {
        let (lo, hi) = (self.lo.dims, self.hi.dims);
        let (a, b) = (other.lo.dims, other.hi.dims);
        let mut contains = true;

        for i in 0..N {
            contains &= (a[i] >= lo[i]) & (b[i] <= hi[i]);
        }

        contains
    }

    /// Returns `true` if this rectangle intersects `other` (i.e., they overlap).
    pub fn intersects(self, other: Rect<T, N>) -> bool {
        let mut intersects = true;

        for i in 0..N {
            let l = max(self.lo[i], other.lo[i]);
            let h = min(self.hi[i], other.hi[i]);
            intersects &= l < h;
        }

        intersects
    }

    /// Returns the rectangle representing the points which lie in the intersection of two
    /// rectangles, or `None` if the two rectangles do not intersect.
    pub fn intersection(self, other: Self) -> Option<Self> {
        let low = Point::zip(self.lo, other.lo, max);
        let high = Point::zip(self.hi, other.hi, min);
        let mut non_empty = true;

        for i in 0..N {
            non_empty &= low[i] < high[i];
        }

        if non_empty {
            Some(Self { lo: low, hi: high })
        } else {
            None
        }
    }

    /// Returns true if this rectangle contains no points.
    #[inline(always)]
    pub fn is_empty(self) -> bool {
        (0..N).any(|i| self.lo[i] == self.hi[i])
    }

    /// Returns the smallest rectangle which contains both this rectangle and `other`.
    pub fn union(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }

        if other.is_empty() {
            return self;
        }

        let low = Point::zip(self.lo, other.lo, min);
        let high = Point::zip(self.hi, other.hi, max);

        Self { lo: low, hi: high }
    }
}

impl<T> Rect<T, 3> {
    pub fn xy(self) -> Rect<T, 2> {
        Rect {
            lo: self.lo.xy(),
            hi: self.hi.xy(),
        }
    }
}

impl From<Range<u64>> for Rect<u64, MAX_DIMS> {
    fn from(r: Range<u64>) -> Self {
        (r, 0..1).into()
    }
}

impl From<(Range<u64>, Range<u64>)> for Rect<u64, MAX_DIMS> {
    fn from((a, b): (Range<u64>, Range<u64>)) -> Self {
        (a, b, 0..1).into()
    }
}

impl From<(Range<u64>, Range<u64>, Range<u64>)> for Rect<u64, MAX_DIMS> {
    fn from((a, b, c): (Range<u64>, Range<u64>, Range<u64>)) -> Self {
        let start = Point::from((a.start, b.start, c.start));
        let end = Point::from((a.end, b.end, c.end));
        Self::from_bounds(start, end)
    }
}

impl From<Range<Point<u64, MAX_DIMS>>> for Rect<u64, MAX_DIMS> {
    fn from(r: Range<Point<u64, MAX_DIMS>>) -> Self {
        Rect::from_bounds(r.start, r.end)
    }
}

impl<const N: usize> Default for Rect<u64, N> {
    fn default() -> Self {
        Self {
            lo: Point::zeros(),
            hi: Point::zeros(),
        }
    }
}

impl<const N: usize> Debug for Rect<u64, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut f = f.debug_tuple("Rect");

        for i in 0..N {
            f.field(&(&self.lo[i]..&self.hi[i]));
        }

        f.finish()
    }
}

impl<const N: usize> IntoIterator for Rect<u64, N> {
    type Item = Point<u64, N>;
    type IntoIter = PointInRectIter<N>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        PointInRectIter {
            exhausted: self.is_empty(),
            cursor: self.lo,
            bounds: self,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PointInRectIter<const N: usize> {
    exhausted: bool,
    cursor: Point<u64, N>,
    bounds: Rect<u64, N>,
}

impl<const N: usize> PointInRectIter<N> {
    fn checked_count(&self) -> Option<usize> {
        if self.exhausted {
            return Some(0);
        }

        let mut count = 1;
        let mut line_size = 1;

        for i in reversed(0..N) {
            let remaining = self.bounds.hi[i] - self.cursor[i] - 1;
            count = u64::checked_add(count, u64::checked_mul(remaining, line_size)?)?;
            line_size = u64::checked_mul(line_size, self.bounds.hi[i] - self.bounds.lo[i])?;
        }

        count.try_into().ok()
    }
}

impl<const N: usize> Iterator for PointInRectIter<N> {
    type Item = Point<u64, N>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let cursor = self.cursor;
        for i in reversed(0..N) {
            self.cursor[i] += 1;

            if self.cursor[i] < self.bounds.hi[i] {
                return Some(cursor);
            }

            self.cursor[i] = self.bounds.lo[i];
        }

        self.exhausted = true;
        return Some(cursor);
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.checked_count() {
            Some(c) => (c, Some(c)),
            None => (usize::MAX, None),
        }
    }

    fn count(self) -> usize {
        self.checked_count().expect("count exceeds usize::MAX")
    }

    #[inline(always)]
    fn fold<B, F>(self, mut accum: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        if self.exhausted {
            return accum;
        }

        let mut cursor = self.cursor;

        'a: loop {
            accum = f(accum, cursor);

            for i in reversed(0..N) {
                cursor[i] += 1;

                if cursor[i] < self.bounds.hi[i] {
                    continue 'a;
                }

                cursor[i] = self.bounds.lo[i];
            }

            break;
        }

        accum
    }
}

impl<const N: usize> FusedIterator for PointInRectIter<N> {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_iterator() {
        for &bounds in &[
            Rect::from((4..10, 8..8)),
            Rect::from((0..1, 0..1, 0..1)),
            Rect::from((0..1, 0..0, 10..100)),
        ] {
            let mut points = vec![];

            for p in bounds {
                let p = Point::from(p);
                assert!(bounds.contains_point(p), "{:?} not in {:?}", p, bounds);
                assert!(!points.contains(&p), "{:?} appears twice", p);
                points.push(p);
            }

            assert_eq!(points.len() as u64, bounds.extents().volume());

            // check if for_each yields same output.
            let mut index = 0;
            bounds.into_iter().for_each(|p| {
                assert_eq!(points[index], p);
                index += 1;
            });

            assert_eq!(points.len(), index);

            // Check if count/size_hint is valid
            let mut count = points.len();
            let mut iter = bounds.into_iter();
            loop {
                assert_eq!(count, iter.clone().count());
                assert_eq!((count, Some(count)), iter.size_hint());

                if count == 0 {
                    assert!(iter.next().is_none());
                    break;
                } else {
                    assert!(iter.next().is_some());
                    count -= 1;
                }
            }
        }
    }
}
