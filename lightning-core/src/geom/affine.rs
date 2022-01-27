use super::permutate::Permutation;
use super::point::Point;
use super::rect::Rect;
use super::transform::RegularTransform;
use super::transform::Transform;
use super::translate::Translate;
use crate::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug};

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Affine<const N: usize, const M: usize> {
    transform: Option<Box<Transform<N, M>>>,
    translate: Translate<M>,
}

impl<const N: usize, const M: usize> Affine<N, M> {
    pub fn new(transform: Transform<N, M>, translate: Translate<M>) -> Self {
        let transform = match transform.is_identity() {
            false => Some(Box::new(transform)),
            true => None,
        };

        Self {
            transform,
            translate,
        }
    }

    pub fn from_translate(translate: Translate<M>) -> Self {
        Self {
            transform: None,
            translate,
        }
    }

    pub fn identity() -> Self {
        Self::from_translate(Translate::identity())
    }

    pub fn from_scales(scales: [i64; N]) -> Self {
        Self::new(Transform::from_scales(scales), Translate::identity())
    }

    pub fn from_permutation(p: Permutation<M>) -> Self {
        Self::new(Transform::from_permutation(p), Translate::identity())
    }

    pub fn add_offset(p: Point<u64, M>) -> Self {
        Self::from_translate(Translate::add_offset(p))
    }

    pub fn sub_offset(p: Point<u64, M>) -> Self {
        Self::from_translate(Translate::sub_offset(p))
    }

    pub fn transform(&self) -> Option<&Transform<N, M>> {
        self.transform.as_deref()
    }

    pub fn translate(&self) -> Translate<M> {
        self.translate
    }

    pub fn is_identity(&self) -> bool {
        self.translate.is_identity() && self.is_translate()
    }

    pub fn is_translate(&self) -> bool {
        self.transform.is_none()
    }

    pub fn combine<const K: usize>(&self, other: &Affine<M, K>) -> Affine<N, K> {
        let (lhs, rhs) = (other, self);

        let transform = match (&lhs.transform, &rhs.transform) {
            (Some(lhs), Some(rhs)) => Transform::combine(&rhs, &lhs),
            (Some(lhs), None) => lhs.resize::<N, K>(),
            (None, Some(rhs)) => rhs.resize::<N, K>(),
            (None, None) => Transform::identity(),
        };

        let translate = match &lhs.transform {
            Some(lhs_matrix) => Translate::combine(
                &Translate::new(lhs_matrix.apply(*rhs.translate)),
                &lhs.translate,
            ),
            None => Translate::combine(&rhs.translate.resize(), &lhs.translate),
        };

        Affine::new(transform, translate)
    }

    pub fn try_apply_point(&self, p: Point<u64, N>) -> Option<Point<u64, M>> {
        let q = match self.transform() {
            Some(transform) => transform.try_apply_point(p)?,
            None => p.resize::<M>(0),
        };

        self.translate.try_apply_point(q)
    }

    pub fn apply_point(&self, p: Point<u64, N>) -> Point<u64, M> {
        self.try_apply_point(p).expect("overflow")
    }

    pub fn try_apply_bounds(&self, r: Rect<u64, N>) -> Option<Rect<u64, M>> {
        let q = match self.transform() {
            Some(transform) => transform.try_apply_bounds(r)?,
            None => Rect::from_bounds(
                Point::from(r.lo.resize::<M>(0)),
                Point::from(r.hi.resize::<M>(1)),
            ),
        };

        self.translate.try_apply_bounds(q)
    }

    pub fn apply_bounds(&self, r: Rect<u64, N>) -> Rect<u64, M> {
        self.try_apply_bounds(r).expect("overflow")
    }

    pub fn resize<const P: usize, const Q: usize>(&self) -> Affine<P, Q> {
        Affine {
            transform: self
                .transform
                .as_ref()
                .map(|e| Box::new(e.resize::<P, Q>())),
            translate: self.translate.resize::<Q>(),
        }
    }

    pub fn to_regular(&self) -> Option<RegularTransform<N, M>> {
        if let Some(transform) = &self.transform {
            RegularTransform::new(&transform, self.translate)
        } else {
            Some(RegularTransform::from_translate(self.translate))
        }
    }
}

impl<const N: usize, const M: usize> Debug for Affine<N, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Transform")
            .field("matrix", &self.transform)
            .field("translate", &self.translate)
            .finish()
    }
}

impl<const N: usize, const M: usize> Default for Affine<N, M> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<const N: usize, const M: usize> From<Transform<N, M>> for Affine<N, M> {
    fn from(t: Transform<N, M>) -> Self {
        Affine::new(t, default())
    }
}

impl<const N: usize, const M: usize> From<Translate<M>> for Affine<N, M> {
    fn from(t: Translate<M>) -> Self {
        Affine::new(default(), t)
    }
}
