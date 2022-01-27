use super::point::Point;
use super::rect::Rect;
use crate::prelude::*;
use crate::util::array;
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug};
use std::ops::{Deref, DerefMut};

#[derive(Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Translate<const N: usize> {
    #[serde(with = "serde_arrays")]
    translate: [i64; N],
}

impl<const N: usize> Translate<N> {
    pub fn new(translate: [i64; N]) -> Self {
        Self { translate }
    }

    pub fn identity() -> Self {
        Self::new([0; N])
    }

    pub fn add_offset(p: Point<u64, N>) -> Self {
        Self::new(array::map(*p, |v| v as i64))
    }

    pub fn sub_offset(p: Point<u64, N>) -> Self {
        Self::new(array::map(*p, |v| -(v as i64)))
    }

    pub fn is_identity(&self) -> bool {
        self.translate == [0; N]
    }

    pub fn translate(&self) -> [i64; N] {
        self.translate
    }

    pub fn combine(&self, other: &Translate<N>) -> Translate<N> {
        Self {
            translate: self.apply(other.translate()),
        }
    }

    pub fn apply(&self, input: [i64; N]) -> [i64; N] {
        array::generate(|i| self.translate[i] + input[i])
    }

    pub fn try_apply_point(&self, p: Point<u64, N>) -> Option<Point<u64, N>> {
        let mut output = [0u64; N];

        for i in 0..N {
            output[i] = (p[i] as i64 + self.translate[i]).try_into().ok()?;
        }

        Some(Point::from(output))
    }

    pub fn try_apply_bounds(&self, r: Rect<u64, N>) -> Option<Rect<u64, N>> {
        let lo = self.try_apply_point(r.low())?;
        let hi = self.try_apply_point(r.high())?;

        Some(Rect { lo, hi })
    }

    pub fn apply_point(&self, p: Point<u64, N>) -> Point<u64, N> {
        self.try_apply_point(p).expect("overflow")
    }

    pub fn apply_bounds(&self, r: Rect<u64, N>) -> Rect<u64, N> {
        self.try_apply_bounds(r).expect("overflow")
    }

    pub fn resize<const M: usize>(&self) -> Translate<M> {
        let mut translate = [0; M];

        for i in 0..usize::min(N, M) {
            translate[i] = self.translate[i];
        }

        Translate { translate }
    }
}

impl<const N: usize> Deref for Translate<N> {
    type Target = [i64; N];

    fn deref(&self) -> &[i64; N] {
        &self.translate
    }
}

impl<const N: usize> DerefMut for Translate<N> {
    fn deref_mut(&mut self) -> &mut [i64; N] {
        &mut self.translate
    }
}

impl<const N: usize> Debug for Translate<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.translate, f)
    }
}

impl<const N: usize> Default for Translate<N> {
    fn default() -> Self {
        Self::identity()
    }
}
