//! Collection of utility functions.

pub use self::as_any::*;
pub use self::counter::*;
pub use self::drop_guard::*;
pub use self::future::*;
pub use self::group_by::*;
pub use self::inline_bytebuf::*;
pub use self::ordered_queue::*;
pub use self::tcell::*;
use crate::geom::{One, Zero};
use std::ops::{Add, Div, Mul, Rem};

pub mod array;
mod as_any;
mod counter;
mod drop_guard;
mod future;
mod group_by;
mod inline_bytebuf;
mod ordered_queue;
mod tcell;

/// Divide `x` by `y` and round up towards infinity. For example, `21/5 =  4.2` thus
/// `div_ceil(21,5) == 5`.
pub fn div_ceil<T>(x: T, y: T) -> T
where
    T: One + Zero + Div<Output = T> + Rem<Output = T> + Add<Output = T> + PartialOrd + Copy,
{
    let (q, r) = (x / y, x % y);
    if r != T::zero() && (x >= T::zero()) == (y >= T::zero()) {
        q + T::one()
    } else {
        q
    }
}

pub fn round_up<T>(x: T, y: T) -> T
where
    T: One
        + Zero
        + Div<Output = T>
        + Rem<Output = T>
        + Add<Output = T>
        + PartialOrd
        + Copy
        + Mul<Output = T>,
{
    div_ceil(x, y) * y
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_div_ceil() {
        // Div ceil is tricky to get right for negative numbers.
        assert_eq!(div_ceil(-1, 5), 0);
        assert_eq!(div_ceil(0, 5), 0);
        assert_eq!(div_ceil(1, 5), 1);

        assert_eq!(div_ceil(-1, -5), 1);
        assert_eq!(div_ceil(0, -5), 0);
        assert_eq!(div_ceil(1, -5), 0);

        assert_eq!(div_ceil(9, 5), 2);
        assert_eq!(div_ceil(10, 5), 2);
        assert_eq!(div_ceil(11, 5), 3);

        assert_eq!(div_ceil(-9, -5), 2);
        assert_eq!(div_ceil(-10, -5), 2);
        assert_eq!(div_ceil(-11, -5), 3);

        assert_eq!(div_ceil(9, -5), -1);
        assert_eq!(div_ceil(10, -5), -2);
        assert_eq!(div_ceil(11, -5), -2);

        assert_eq!(div_ceil(-9, 5), -1);
        assert_eq!(div_ceil(-10, 5), -2);
        assert_eq!(div_ceil(-11, 5), -2);
    }
}
