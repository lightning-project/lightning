use super::affine::Affine;
use super::permutate::Permutation;
use super::point::Point;
use super::rect::Rect;
use super::translate::Translate;
use crate::prelude::*;
use crate::util::array;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::convert::TryInto;
use std::fmt::{self, Debug, Formatter};
use std::num::NonZeroI64;
use std::ops::{Deref, DerefMut};

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Transform<const N: usize, const M: usize> {
    matrix: [[i64; N]; M],
}

impl<const N: usize, const M: usize> Transform<N, M> {
    pub fn new(matrix: [[i64; N]; M]) -> Self {
        Self { matrix }
    }

    pub fn identity() -> Self {
        Self::from_scales([1; N])
    }

    pub fn from_scales(scales: [i64; N]) -> Self {
        let mut matrix = [[0; N]; M];
        for i in 0..usize::min(N, M) {
            matrix[i][i] = scales[i];
        }

        Self::new(matrix)
    }

    pub fn from_permutation(p: Permutation<M>) -> Self {
        Self::new(p.apply(*Self::identity()))
    }

    pub fn is_identity(&self) -> bool {
        let mut valid = true;

        for i in 0..M {
            for j in 0..N {
                valid &= self[i][j] == (i == j) as i64;
            }
        }

        valid
    }

    pub fn combine<const K: usize>(&self, other: &Transform<M, K>) -> Transform<N, K> {
        let (lhs, rhs) = (other, self);

        let mut matrix = [[0; N]; K];

        for i in 0..K {
            for j in 0..N {
                for k in 0..M {
                    matrix[i][j] += lhs[i][k] * rhs[k][j];
                }
            }
        }

        Transform { matrix }
    }

    pub fn apply(&self, input: [i64; N]) -> [i64; M] {
        let mut output = [0; M];

        for i in 0..M {
            let mut result = 0;

            for j in 0..N {
                result += input[j] * self[i][j];
            }

            output[i] = result;
        }

        output
    }

    pub fn try_apply_point(&self, p: Point<u64, N>) -> Option<Point<u64, M>> {
        let mut output = [0u64; M];

        for i in 0..M {
            let mut result = 0;

            for j in 0..N {
                result += p[j] as i64 * self[i][j];
            }

            output[i] = result.try_into().ok()?;
        }

        Some(Point::from(output))
    }

    pub fn try_apply_bounds(&self, r: Rect<u64, N>) -> Option<Rect<u64, M>> {
        if r.is_empty() {
            return Some(Rect::default());
        }

        let lo = r.low();
        let hi = r.high();

        let mut a = [0u64; M];
        let mut b = [0u64; M];

        for i in 0..M {
            let mut p = 0;
            let mut q = 0;

            for j in 0..N {
                let factor = self[i][j];
                p += i64::min(lo[j] as i64 * factor, (hi[j] as i64 - 1) * factor);
                q += i64::max(lo[j] as i64 * factor, (hi[j] as i64 - 1) * factor);
            }

            a[i] = p.try_into().ok()?;
            b[i] = (q + 1).try_into().ok()?;
        }

        Some(Rect::from_bounds(Point::from(a), Point::from(b)))
    }

    pub fn apply_point(&self, p: Point<u64, N>) -> Point<u64, M> {
        self.try_apply_point(p).expect("overflow")
    }

    pub fn apply_bounds(&self, r: Rect<u64, N>) -> Rect<u64, M> {
        self.try_apply_bounds(r).expect("overflow")
    }

    pub fn resize<const P: usize, const Q: usize>(&self) -> Transform<P, Q> {
        let mut matrix = [[0; P]; Q];

        for i in 0..Q {
            for j in 0..P {
                if i < M && j < N {
                    matrix[i][j] = self[i][j];
                } else {
                    matrix[i][j] = (i == j) as i64;
                }
            }
        }

        Transform { matrix }
    }
}

impl<const N: usize, const M: usize> Deref for Transform<N, M> {
    type Target = [[i64; N]; M];

    fn deref(&self) -> &[[i64; N]; M] {
        &self.matrix
    }
}

impl<const N: usize, const M: usize> DerefMut for Transform<N, M> {
    fn deref_mut(&mut self) -> &mut [[i64; N]; M] {
        &mut self.matrix
    }
}

impl<const N: usize, const M: usize> Debug for Transform<N, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.matrix, f)
    }
}

impl<const N: usize, const M: usize> Default for Transform<N, M> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<const N: usize, const M: usize> Serialize for Transform<N, M> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeTuple;
        let mut tup = serializer.serialize_tuple(N * M)?;

        for i in 0..M {
            for j in 0..N {
                tup.serialize_element(&self[i][j])?;
            }
        }

        tup.end()
    }
}

impl<'de, const N: usize, const M: usize> Deserialize<'de> for Transform<N, M> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{Error, SeqAccess};

        struct Vistor<const N: usize, const M: usize>;
        impl<'de, const N: usize, const M: usize> serde::de::Visitor<'de> for Vistor<N, M> {
            type Value = Transform<N, M>;

            fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                write!(formatter, "matrix of dimensions {} x {}", N, M)
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut matrix = [[0; N]; M];

                for i in 0..M {
                    for j in 0..N {
                        matrix[i][j] = seq
                            .next_element()?
                            .ok_or_else(|| Error::invalid_length(N * M, &self))?;
                    }
                }

                Ok(Transform::new(matrix))
            }
        }

        deserializer.deserialize_tuple(N * M, Vistor)
    }
}

pub struct RegularTransform<const N: usize, const M: usize> {
    axes: [u8; M],
    scales: [NonZeroI64; M],
    translate: Translate<M>,
}

impl<const N: usize, const M: usize> RegularTransform<N, M> {
    pub fn new(transform: &Transform<N, M>, translate: Translate<M>) -> Option<Self> {
        assert!(N < u8::MAX as usize && M < u8::MAX as usize);
        const ONE: NonZeroI64 = unsafe { NonZeroI64::new_unchecked(1) };

        let mut axes = [u8::MAX; M];
        let mut scales = [ONE; M];

        for i in 0..M {
            let mut axis = None;
            let mut scale = ONE;

            for j in 0..N {
                if let Some(s) = NonZeroI64::new(transform[i][j]) {
                    axis = Some(j);
                    scale = s;
                }
            }

            if let Some(j) = axis {
                for k in 0..M {
                    if i != k && transform[k][j] != 0 {
                        return None;
                    }
                }

                for k in 0..N {
                    if j != k && transform[i][k] != 0 {
                        return None;
                    }
                }

                axes[i] = j as u8;
                scales[i] = scale;
            }
        }

        Some(Self {
            axes,
            translate,
            scales,
        })
    }

    pub fn from_translate(translate: Translate<M>) -> Self {
        const ONE: NonZeroI64 = unsafe { NonZeroI64::new_unchecked(1) };

        Self {
            axes: array::generate(|i| i as u8),
            scales: array::generate(|_| ONE),
            translate,
        }
    }

    pub fn identity() -> Self {
        assert!(N <= u8::MAX as usize && M <= u8::MAX as usize);
        const ONE: NonZeroI64 = unsafe { NonZeroI64::new_unchecked(1) };

        Self {
            axes: array::generate(|i| (i < N).then(|| i as u8).unwrap_or(u8::MAX)),
            scales: [ONE; M],
            translate: Translate::identity(),
        }
    }

    pub fn axes(&self) -> [Option<usize>; M] {
        array::map(self.axes, |i| ((i as usize) < N).then(|| i as usize))
    }

    pub fn to_affine(&self) -> Affine<N, M> {
        let mut matrix = [[0; N]; M];

        for i in 0..M {
            let j = self.axes[i] as usize;
            if j < N {
                matrix[i][j] = self.scales[i].get();
            }
        }

        Affine::new(Transform::new(matrix), self.translate)
    }

    pub fn apply_point(&self, input: Point<u64, N>) -> Point<u64, M> {
        let mut output = Point::zeros();

        for i in 0..M {
            let j = self.axes[i] as usize;
            let v = if j < N {
                (input[j] as i64) * self.scales[i].get()
            } else {
                0
            };

            output[i] = (v + self.translate[i]).try_into().expect("overflow");
        }

        output
    }

    pub fn inverse_point(&self, input: Point<u64, M>, missing: Point<u64, N>) -> Point<u64, N> {
        let mut output = missing;

        for i in 0..N {
            let j = self.axes[i] as usize;
            if j < N {
                let v = (input[i] as i64 - self.translate[i]) / self.scales[i].get();

                output[j] = v.try_into().expect("overflow");
            };
        }

        output
    }

    pub fn apply_bounds(&self, r: Rect<u64, N>) -> Rect<u64, M> {
        let mut lo = Point::zeros();
        let mut hi = Point::ones();

        for i in 0..N {
            let j = self.axes[i] as usize;
            let (a, b) = if j < N {
                let scale = self.scales[i].get();

                if scale > 0 {
                    ((r.lo[j] as i64) * scale, (r.hi[j] as i64) * scale)
                } else {
                    ((r.hi[j] as i64 - 1) * scale, (r.lo[j] as i64) * scale + 1)
                }
            } else {
                (0, 1)
            };

            lo[i] = (a + self.translate[i]).try_into().expect("overflow");
            hi[i] = (b + self.translate[i]).try_into().expect("overflow");
        }

        Rect::from_bounds(lo, hi)
    }

    pub fn inverse_bounds(&self, r: Rect<u64, M>, domain: Rect<u64, N>) -> Rect<u64, N> {
        let mut lo = domain.lo;
        let mut hi = domain.hi;

        for i in 0..N {
            let j = self.axes[i] as usize;
            if j < N {
                let scale = self.scales[i];
                let (a, b) = if scale.get() > 0 {
                    (
                        (r.lo[i] as i64 - self.translate[i]) / scale.get(),
                        (r.hi[i] as i64 - self.translate[i]) / scale.get(),
                    )
                } else {
                    (
                        (r.hi[i] as i64 - 1 - self.translate[i]) / scale.get(),
                        (r.lo[i] as i64 - self.translate[i]) / scale.get() + 1,
                    )
                };

                lo[j] = a.try_into().expect("overflow");
                hi[j] = b.try_into().expect("overflow");
            };
        }

        Rect::from_bounds(lo, hi)
    }
}

impl<const N: usize, const M: usize> Debug for RegularTransform<N, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegularTransform")
            .field("axes", &self.axes())
            .field("scales", &self.scales)
            .field("translate", &self.translate)
            .finish()
    }
}

impl<const N: usize, const M: usize> Default for RegularTransform<N, M> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<const N: usize, const M: usize> From<Translate<M>> for RegularTransform<N, M> {
    fn from(t: Translate<M>) -> Self {
        Self::from_translate(t)
    }
}
