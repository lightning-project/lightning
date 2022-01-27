pub use cuda::copy::cuda_copy;
pub use cuda::fill::cuda_fill;
pub use cuda::reduce::cuda_fold;
pub use cuda::{cuda_reduce, MemOpsKernelsCache};
pub use host::fill::host_fill;
pub use host::reduce::{host_copy, host_fold, host_reduce};
pub use host::{Policy, RayonPolicy, SequentialPolicy};
pub use reduction::{Reduction, ReductionFunction};

mod cuda;
mod host;
mod reduction;

// This function simplifies the strides of arrays when applying element-wise functions to them.
// It takes two inputs:
//  * An array of D strides for N arrays. It is assumed that strides[0] is the `leading' stride, i.e.
//    the array for which it is most important that access is coalesced.
//  * An array of D counts (i.e, number of elements along each axis)
//
// It performs the following operations
//  * Negative value of the leading strides are inverted
//  * Axes are according to the leading strides in descending order (Fortran order, lowest stride first)
//  * Axes are merged if possible.
//
// Merge is possible if two axes are contiguous. Consider this example
//
// ```
//     for i in 0..10 {
//         for j in 0..10 {
//             foo(ptr[i * 40 + j + 4])
//         }
//     }
// ```
//
// Which can be simplified into
//
// ```
//     for i in 0..10 {
//         foo(ptr[i + 4])
//     }
// ```
//
// The function returns
//  * The number of dimensions that result after merging (in range 1..D)
//  * The offsets that must be added to each pointer to compensate for correcting negative strides.
fn simplify_strides<const N: usize, const D: usize>(
    strides: [&mut [i64; D]; N],
    counts: &mut [i64; D],
) -> (usize, [i64; N]) {
    assert!(D > 0);

    // If any count is zero, early exit
    for i in 0..D {
        if counts[i] <= 0 {
            *counts = [0; D];
            return (0, [0; N]);
        }
    }

    // Convert strides to positive
    let mut ptr_offsets = [0; N];
    for i in 0..D {
        if strides[0][i] < 0 && counts[i] > 0 {
            for j in 0..N {
                ptr_offsets[j] += strides[j][i] * (counts[i] - 1);
                strides[j][i] *= -1;
            }
        }
    }

    // Set stride to zero for unit dimensions.
    for i in 0..D {
        if counts[i] == 1 {
            for j in 0..N {
                strides[j][i] = 0;
            }
        }
    }

    // Sort strides from lowest to highest
    for _ in 0..D {
        for i in 1..D {
            if (strides[0][i - 1] > strides[0][i] || counts[i - 1] == 1) && counts[i] != 1 {
                counts.swap(i - 1, i);

                for j in 0..N {
                    strides[j].swap(i - 1, i);
                }
            }
        }
    }

    // Attempt to merge axes
    let mut ndims = 1;
    for i in 1..D {
        let mut mergeable = true;

        for j in 0..N {
            mergeable &= strides[j][ndims - 1] * counts[ndims - 1] == strides[j][i];
        }

        if mergeable {
            counts[ndims - 1] *= counts[i];
            counts[i] = 1;

            for j in 0..N {
                strides[j][i] = 0;
            }
        } else if counts[i] > 1 {
            counts.swap(ndims, i);

            for j in 0..N {
                strides[j].swap(ndims, i);
            }

            ndims += 1;
        }
    }

    (ndims, ptr_offsets)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_simplify_strides_randomize() {
        use rand::prelude::*;

        let mut rng = SmallRng::seed_from_u64(0);

        for _ in 0..1000 {
            let mut strides = [
                rng.gen_range(-100..=100),
                rng.gen_range(-100..=100),
                rng.gen_range(-100..=100),
            ];
            let mut counts = [
                rng.gen_range(1..=10),
                rng.gen_range(1..=10),
                rng.gen_range(1..=10),
            ];

            let original_strides = strides;
            let original_counts = counts;

            let (ndims, [offset]) = simplify_strides([&mut strides], &mut counts);

            // First ndims strides should be sorted low to high
            assert!(ndims >= 1);
            for i in 1..ndims {
                assert!(strides[i - 1] <= strides[i]);
            }

            // Remaining strides should be 0.
            for i in ndims..3 {
                assert_eq!(counts[i], 1);
                assert_eq!(strides[i], 0);
            }

            let mut expected = vec![];
            for i in 0..original_counts[0] {
                for j in 0..original_counts[1] {
                    for k in 0..original_counts[2] {
                        expected.push(
                            i * original_strides[0]
                                + j * original_strides[1]
                                + k * original_strides[2],
                        );
                    }
                }
            }

            let mut gotten = vec![];
            for i in 0..counts[0] {
                for j in 0..counts[1] {
                    for k in 0..counts[2] {
                        gotten.push(i * strides[0] + j * strides[1] + k * strides[2] + offset);
                    }
                }
            }

            expected.sort();
            gotten.sort();

            assert_eq!(expected, gotten);
        }
    }

    #[test]
    fn test_simplify_strides() {
        // Can merge all axes
        let mut a = [4, 40, 800];
        let mut b = [4, 40, 800];
        let mut n = [10, 20, 30];
        let (ndims, offsets) = simplify_strides([&mut a, &mut b], &mut n);
        assert_eq!(ndims, 1);
        assert_eq!(offsets, [0, 0]);
        assert_eq!(a, [4, 0, 0]);
        assert_eq!(b, [4, 0, 0]);
        assert_eq!(n, [6000, 1, 1]);

        // Can merge axes 0 and 1
        let mut a = [4, 40, 1000];
        let mut b = [4, 40, 800];
        let mut n = [10, 20, 30];
        let (ndims, offsets) = simplify_strides([&mut a, &mut b], &mut n);
        assert_eq!(ndims, 2);
        assert_eq!(offsets, [0, 0]);
        assert_eq!(a, [4, 1000, 0]);
        assert_eq!(b, [4, 800, 0]);
        assert_eq!(n, [200, 30, 1]);

        // Can merge axes 0 and 2
        let mut a = [1000, 40, 4];
        let mut b = [800, 40, 4];
        let mut n = [30, 20, 10];
        let (ndims, offsets) = simplify_strides([&mut a, &mut b], &mut n);
        assert_eq!(ndims, 2);
        assert_eq!(offsets, [0, 0]);
        assert_eq!(a, [4, 1000, 0]);
        assert_eq!(b, [4, 800, 0]);
        assert_eq!(n, [200, 30, 1]);

        // Can merge axes 1 and 2
        let mut a = [40, 4, 1000];
        let mut b = [40, 4, 800];
        let mut n = [20, 10, 30];
        let (ndims, offsets) = simplify_strides([&mut a, &mut b], &mut n);
        assert_eq!(ndims, 2);
        assert_eq!(offsets, [0, 0]);
        assert_eq!(a, [4, 1000, 0]);
        assert_eq!(b, [4, 800, 0]);
        assert_eq!(n, [200, 30, 1]);

        // Can merge no axes
        let mut a = [4, 40, 100];
        let mut b = [4, 44, 800];
        let mut n = [10, 20, 30];
        let (ndims, offsets) = simplify_strides([&mut a, &mut b], &mut n);
        assert_eq!(ndims, 3);
        assert_eq!(offsets, [0, 0]);
        assert_eq!(a, [4, 40, 100]);
        assert_eq!(b, [4, 44, 800]);
        assert_eq!(n, [10, 20, 30]);

        // Can merge no axes
        let mut a = [4, 40, 100];
        let mut b = [4, 44, 800];
        let mut n = [10, 1, 30];
        let (ndims, offsets) = simplify_strides([&mut a, &mut b], &mut n);
        assert_eq!(ndims, 2);
        assert_eq!(offsets, [0, 0]);
        assert_eq!(a, [4, 100, 0]);
        assert_eq!(b, [4, 800, 0]);
        assert_eq!(n, [10, 30, 1]);

        // Can merge no axes
        let mut a = [4, 40, 100];
        let mut b = [4, 44, 800];
        let mut n = [1, 20, 30];
        let (ndims, offsets) = simplify_strides([&mut a, &mut b], &mut n);
        assert_eq!(ndims, 2);
        assert_eq!(offsets, [0, 0]);
        assert_eq!(a, [40, 100, 0]);
        assert_eq!(b, [44, 800, 0]);
        assert_eq!(n, [20, 30, 1]);

        // Can merge no axes
        let mut a = [4, 40, 100];
        let mut b = [4, 44, 800];
        let mut n = [1, 20, 1];
        let (ndims, offsets) = simplify_strides([&mut a, &mut b], &mut n);
        assert_eq!(ndims, 1);
        assert_eq!(offsets, [0, 0]);
        assert_eq!(a, [40, 0, 0]);
        assert_eq!(b, [44, 0, 0]);
        assert_eq!(n, [20, 1, 1]);

        // Negative strides, can merge axes 1 and 2
        let mut a = [40, -4, 1000];
        let mut b = [40, -4, 800];
        let mut n = [20, 10, 30];
        let (ndims, offsets) = simplify_strides([&mut a, &mut b], &mut n);
        assert_eq!(ndims, 2);
        assert_eq!(offsets, [-(9 * 4), -(9 * 4)]);
        assert_eq!(a, [4, 1000, 0]);
        assert_eq!(b, [4, 800, 0]);
        assert_eq!(n, [200, 30, 1]);

        // Negative strides, merge no axes
        let mut a = [1000, -40, 4];
        let mut b = [800, -4, 40];
        let mut n = [30, 20, 10];
        let (ndims, offsets) = simplify_strides([&mut a, &mut b], &mut n);
        assert_eq!(ndims, 3);
        assert_eq!(offsets, [-(19 * 40), -(19 * 4)]);
        assert_eq!(a, [4, 40, 1000]);
        assert_eq!(b, [40, 4, 800]);
        assert_eq!(n, [10, 20, 30]);
    }
}
