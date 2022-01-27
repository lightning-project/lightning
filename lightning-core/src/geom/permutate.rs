use super::dim::Dim;
use super::point::Point;
use super::rect::Rect;
use crate::util::array;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::{self, Debug};
use std::mem::{self, transmute_copy, MaybeUninit};

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Permutation<const N: usize> {
    axes: [u8; N],
}

impl<const N: usize> Permutation<N> {
    pub fn try_new(axes: [u8; N]) -> Option<Self> {
        let mut seen = [false; N];

        for &ax in &axes {
            if ax as usize >= N || seen[ax as usize] {
                return None;
            }

            seen[ax as usize] = true;
        }

        Some(Self { axes })
    }

    pub fn new(axes: [u8; N]) -> Self {
        Self::try_new(axes).expect("invalid axes provided for permutation")
    }

    pub fn identity() -> Self {
        Self {
            axes: array::generate(|i| i as u8),
        }
    }

    pub fn is_identity(self) -> bool {
        self == Self::identity()
    }

    pub fn with_axis_removed(axis: usize) -> Self {
        assert!(axis < N);
        Self {
            axes: array::generate(|i| {
                if i < axis {
                    i as u8
                } else if i == axis {
                    (N - 1) as u8
                } else {
                    (i - 1) as u8
                }
            }),
        }
    }

    pub fn with_axes_swapped(a: usize, b: usize) -> Self {
        assert!(a < N && b < N);

        Self {
            axes: array::generate(|i| {
                if i == a {
                    b as u8
                } else if i == b {
                    a as u8
                } else {
                    i as u8
                }
            }),
        }
    }

    pub fn combine(self, that: Self) -> Self {
        let mut output = [!0; N];

        for i in 0..N {
            output[i] = that.axes[self.axes[i] as usize];
        }

        Self { axes: output }
    }

    /// Returns the mapping performed by this object. Axes `i` in the output maps to `mapping[i]`
    /// in the output.
    pub fn mapping(self) -> [usize; N] {
        array::generate(|i| self.axes[i] as usize)
    }

    pub fn apply<T>(self, input: [T; N]) -> [T; N] {
        let mut output = unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() };
        let mut input =
            unsafe { mem::transmute_copy::<_, [MaybeUninit<T>; N]>(&MaybeUninit::new(input)) };

        for (in_index, out_index) in self.axes.iter().copied().enumerate() {
            output[out_index as usize] = mem::replace(&mut input[in_index], MaybeUninit::uninit());
        }

        unsafe { transmute_copy::<[MaybeUninit<T>; N], [T; N]>(&output) }
    }

    pub fn inverse<T>(self, input: [T; N]) -> [T; N] {
        let mut output = unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() };
        let mut input =
            unsafe { mem::transmute_copy::<_, [MaybeUninit<T>; N]>(&MaybeUninit::new(input)) };

        for (out_index, in_index) in self.axes.iter().copied().enumerate() {
            output[out_index] = mem::replace(&mut input[in_index as usize], MaybeUninit::uninit());
        }

        unsafe { transmute_copy::<[MaybeUninit<T>; N], [T; N]>(&output) }
    }

    pub fn apply_point<T>(self, input: Point<T, N>) -> Point<T, N> {
        Point::from(self.apply(input.dims))
    }

    pub fn apply_extents<T>(self, input: Dim<T, N>) -> Dim<T, N> {
        Dim::from(self.apply(input.dims))
    }

    pub fn apply_bounds<T>(self, b: Rect<T, N>) -> Rect<T, N> {
        Rect {
            lo: self.apply_point(b.lo),
            hi: self.apply_point(b.hi),
        }
    }

    pub fn inverse_point<T>(self, input: Point<T, N>) -> Point<T, N> {
        Point::from(self.inverse(input.dims))
    }

    pub fn inverse_extents<T>(self, input: Dim<T, N>) -> Dim<T, N> {
        Dim::from(self.inverse(input.dims))
    }

    pub fn inverse_bounds<T>(self, b: Rect<T, N>) -> Rect<T, N> {
        Rect {
            lo: self.inverse_point(b.lo),
            hi: self.inverse_point(b.hi),
        }
    }

    pub fn invert(self) -> Self {
        let mut output = [0; N];

        for (in_index, out_index) in self.axes.iter().copied().enumerate() {
            output[out_index as usize] = in_index as u8;
        }

        Permutation { axes: output }
    }
}

impl<'de, const N: usize> Serialize for Permutation<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serde_arrays::serialize(&self.axes, serializer)
    }
}

impl<'de, const N: usize> Deserialize<'de> for Permutation<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let axes: [u8; N] = serde_arrays::deserialize(deserializer)?;

        Ok(Self { axes })
    }
}

impl<const N: usize> Default for Permutation<N> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<const N: usize> Debug for Permutation<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("AxesMapping").field(&self.mapping()).finish()
    }
}
