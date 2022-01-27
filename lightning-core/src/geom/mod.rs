use crate::util::array;

mod affine;
mod dim;
mod permutate;
mod point;
mod rect;
mod transform;
mod translate;

pub type PointN<T, const N: usize> = point::Point<T, N>;
pub type Point1<T = u64> = point::Point<T, 1>;
pub type Point2<T = u64> = point::Point<T, 2>;
pub type Point3<T = u64> = point::Point<T, 3>;
pub type Point4<T = u64> = point::Point<T, 4>;
pub type Point<T = u64> = point::Point<T, MAX_DIMS>;

pub type DimN<T, const N: usize> = dim::Dim<T, N>;
pub type Dim1 = dim::Dim<u64, 1>;
pub type Dim2 = dim::Dim<u64, 2>;
pub type Dim3 = dim::Dim<u64, 3>;
pub type Dim4 = dim::Dim<u64, 4>;
pub type Dim = dim::Dim<u64, MAX_DIMS>;

pub type RectN<const N: usize> = rect::Rect<u64, N>;
pub type Rect1 = rect::Rect<u64, 1>;
pub type Rect2 = rect::Rect<u64, 2>;
pub type Rect3 = rect::Rect<u64, 3>;
pub type Rect4 = rect::Rect<u64, 4>;
pub type Rect = rect::Rect<u64, MAX_DIMS>;

pub type TranslateN<const N: usize> = translate::Translate<N>;
pub type Translate1<const N: usize> = TranslateN<1>;
pub type Translate2<const N: usize> = TranslateN<2>;
pub type Translate3<const N: usize> = TranslateN<3>;
pub type Translate4<const N: usize> = TranslateN<4>;
pub type Translate = TranslateN<MAX_DIMS>;

pub type TransformNM<const N: usize, const M: usize> = transform::Transform<N, M>;
pub type TransformN<const N: usize> = TransformNM<N, N>;
pub type Transform = TransformN<MAX_DIMS>;

pub type AffineNM<const N: usize, const M: usize> = affine::Affine<N, M>;
pub type AffineN<const N: usize> = AffineNM<N, N>;
pub type Affine = AffineN<MAX_DIMS>;

pub type RegularTransformNM<const N: usize, const M: usize> = transform::RegularTransform<N, M>;
pub type RegularTransformN<const N: usize> = RegularTransformNM<N, N>;
pub type RegularTransform = RegularTransformN<MAX_DIMS>;

pub type PermutationN<const N: usize> = permutate::Permutation<N>;
pub type Permutation1 = PermutationN<1>;
pub type Permutation2 = PermutationN<2>;
pub type Permutation3 = PermutationN<3>;
pub type Permutation4 = PermutationN<4>;
pub type Permutation = PermutationN<MAX_DIMS>;

pub trait Zero {
    fn zero() -> Self;
}

pub trait One {
    fn one() -> Self;
}

macro_rules! impl_one_zero {
    ($($t:ident)*) => {
        $(
        impl One for $t {
            fn one() -> Self { 1 as $t }
        }

        impl Zero for $t {
            fn zero() -> Self { 0 as $t }
        }
        )*
    }
}

impl_one_zero!(i8 i16 i32 i64 isize u8 u16 u32 u64 usize f32 f64);

impl One for bool {
    fn one() -> Self {
        true
    }
}

impl Zero for bool {
    fn zero() -> Self {
        false
    }
}

impl<T: One, const N: usize> One for [T; N] {
    fn one() -> Self {
        array::generate(|_| T::one())
    }
}

impl<T: Zero, const N: usize> Zero for [T; N] {
    fn zero() -> Self {
        array::generate(|_| T::zero())
    }
}

/// The maximum number of dimensions allowed for an array.
///
/// Limiting the number of dimensions simplify the internals since, for example, indices into an
/// n-dimensional array can be  represented as just `[usize; MAX_DIMS]` where only the first
/// n dimensions array actually used.
pub const MAX_DIMS: usize = 3;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_zero_one() {
        assert_eq!(usize::zero(), 0);
        assert_eq!(usize::one(), 1);

        assert_eq!(<[i64; 5]>::zero(), [0, 0, 0, 0, 0]);
        assert_eq!(<[i64; 5]>::one(), [1, 1, 1, 1, 1]);
    }
}
