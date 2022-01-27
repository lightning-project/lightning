use crate::host::{host_recur, Policy, UnsafeSendable};
use crate::{host, simplify_strides, Reduction, ReductionFunction};
use lightning_core::accessor::{ByteStrides, HostAccessor, HostMutAccessor};
use lightning_core::prelude::*;
use lightning_core::{HasDataType, MAX_DIMS};
use std::ptr;

macro_rules! impl_reductions {
    ($reduction: expr, $base:expr) => {
        let dtype = $reduction.data_type();
        match $reduction.function() {
            ReductionFunction::Sum => {
                impl_reductions!(@scalars, |x, y| x + y, dtype, $base);
            }
            ReductionFunction::Product => {
                impl_reductions!(@scalars, |x, y| x * y, dtype, $base);
            }
            ReductionFunction::Min => {
                impl_reductions!(@scalars, |x, y| if x < y { x } else { y }, dtype, $base);
            }
            ReductionFunction::Max => {
                impl_reductions!(@scalars, |x, y| if x > y { x } else { y }, dtype, $base);
            }
            ReductionFunction::And => {
                impl_reductions!(@integers, |x, y| x & y, dtype, $base);
            }
            ReductionFunction::Or => {
                impl_reductions!(@integers, |x, y| x | y, dtype, $base);
            }
        }
    };
    (@scalars, $combine: expr, $dtype: expr, $base: expr) => {
        impl_reductions!(f32, f64; $combine, $dtype, $base);
        impl_reductions!(@integers, $combine, $dtype, $base);
    };
    (@integers, $combine: expr, $dtype: expr, $base: expr) => {
        impl_reductions!(i8, i16, i32, i64, u8, u16, u32, u64; $combine, $dtype, $base);
    };
    ($($ty:ident),*; $combine: expr, $dtype: expr, $base: expr) => {
        match $dtype {
            $(
                x if x == <$ty as HasDataType>::data_type() => {
                    ($base)(|lhs: $ty, rhs: $ty| -> $ty { ($combine)(lhs, rhs) })
                }
            )*
            _ => {}
        }
    };
}

pub unsafe fn host_copy(policy: impl Policy, src: HostAccessor, dst: HostMutAccessor) {
    let src_size = src.extents();
    let dst_size = dst.extents();

    assert_eq!(src.data_type(), dst.data_type());
    assert_eq!(src_size, dst_size);

    let mut src_strides = src.strides_in_bytes();
    let mut dst_strides = dst.strides_in_bytes();
    let mut counts = dst_size.to_i64().unwrap();

    let (_, [dst_offset, src_offset]) =
        simplify_strides([&mut dst_strides, &mut src_strides], &mut counts);

    let src_ptr = UnsafeSendable(src.as_ptr().wrapping_offset(src_offset as isize));
    let dst_ptr = UnsafeSendable(dst.as_ptr_mut().wrapping_offset(dst_offset as isize));

    let mut elem_size = src.data_type().size_in_bytes();
    if dst_strides[0] == elem_size as i64 && src_strides[0] == elem_size as i64 {
        elem_size *= counts[0] as usize;
        counts[0] = 1;
    }

    host_recur(
        policy,
        host::count_bytes(&dst),
        [0; MAX_DIMS],
        counts,
        &move |p, q| {
            copy_leaf(
                p,
                q,
                src_ptr.0,
                src_strides,
                dst_ptr.0,
                dst_strides,
                elem_size,
            )
        },
    );
}

unsafe fn copy_leaf(
    start: [i64; MAX_DIMS],
    end: [i64; MAX_DIMS],
    src_ptr: *const u8,
    src_strides: ByteStrides,
    dst_ptr: *mut u8,
    dst_strides: ByteStrides,
    elem_size: usize,
) {
    macro_rules! spec {
        ($n:expr) => {
            if elem_size == $n {
                combine_spec(
                    start,
                    end,
                    src_ptr,
                    src_strides,
                    dst_ptr,
                    dst_strides,
                    |src, dst| unsafe { ptr::copy_nonoverlapping(src, dst, $n) },
                );
                return;
            }
        };
    }

    spec!(1);
    spec!(2);
    spec!(4);
    spec!(8);
    spec!(16);
    spec!(elem_size);
    unreachable!();
}

pub unsafe fn host_fold(
    policy: impl Policy,
    src: HostAccessor,
    dst: HostMutAccessor,
    reduction: Reduction,
) -> Result {
    let src_size = src.extents();
    let dst_size = dst.extents();

    let src_data = src.as_ptr();
    let dst_data = dst.as_ptr_mut();

    assert_eq!(src.data_type(), reduction.data_type());
    assert_eq!(src.data_type(), dst.data_type());
    assert_eq!(src_size, dst_size);

    let mut src_strides = src.strides_in_bytes();
    let mut dst_strides = dst.strides_in_bytes();
    let mut counts = dst_size.to_i64().unwrap();

    let (_, [dst_offset, src_offset]) =
        simplify_strides([&mut dst_strides, &mut src_strides], &mut counts);

    let src_ptr = UnsafeSendable(src_data.wrapping_offset(src_offset as isize));
    let dst_ptr = UnsafeSendable(dst_data.wrapping_offset(dst_offset as isize));

    host_recur(
        policy,
        host::count_bytes(&dst),
        [0; MAX_DIMS],
        counts,
        &move |p, q| {
            fold_leaf(
                p,
                q,
                src_ptr.0,
                src_strides,
                dst_ptr.0,
                dst_strides,
                reduction,
            )
        },
    );

    Ok(())
}

unsafe fn fold_leaf(
    start: [i64; MAX_DIMS],
    end: [i64; MAX_DIMS],
    src_ptr: *const u8,
    src_strides: ByteStrides,
    dst_ptr: *mut u8,
    dst_strides: ByteStrides,
    reduction: Reduction,
) {
    impl_reductions!(reduction, |fun| {
        fold_leaf_spec(start, end, src_ptr, src_strides, dst_ptr, dst_strides, fun)
    });
}

unsafe fn fold_leaf_spec<F: Fn(T, T) -> T, T: Copy>(
    start: [i64; MAX_DIMS],
    end: [i64; MAX_DIMS],
    src_ptr: *const u8,
    src_strides: ByteStrides,
    dst_ptr: *mut u8,
    dst_strides: ByteStrides,
    combine: F,
) {
    combine_spec(
        start,
        end,
        src_ptr,
        src_strides,
        dst_ptr,
        dst_strides,
        |src, dst| {
            let lhs = ptr::read(src as *const T);
            let rhs = ptr::read(dst as *const T);
            let result = (combine)(lhs, rhs);
            ptr::write(dst as *mut T, result);
        },
    );
}

unsafe fn combine_spec<F: Fn(*const u8, *mut u8)>(
    start: [i64; MAX_DIMS],
    end: [i64; MAX_DIMS],
    src_ptr: *const u8,
    src_strides: ByteStrides,
    dst_ptr: *mut u8,
    dst_strides: ByteStrides,
    combine: F,
) {
    for i2 in start[2]..end[2] {
        for i1 in start[1]..end[1] {
            for i0 in start[0]..end[0] {
                let src = src_ptr.wrapping_offset(src_strides.offset_in_bytes([i0, i1, i2]));
                let dst = dst_ptr.wrapping_offset(dst_strides.offset_in_bytes([i0, i1, i2]));

                combine(src, dst);
            }
        }
    }
}

pub unsafe fn host_reduce(
    policy: impl Policy,
    src: HostAccessor,
    dst: HostMutAccessor,
    axis: usize,
    reduction: Reduction,
) -> Result {
    let src_size = src.extents();
    let dst_size = dst.extents();

    assert_eq!(src.data_type(), reduction.data_type());
    assert_eq!(src.data_type(), dst.data_type());
    assert!(all(0..MAX_DIMS, |i| dst_size[i] == src_size[i] || i == axis));
    assert!(axis < MAX_DIMS);
    assert_eq!(dst_size[axis], 1);

    // If axis length is 1, we just as might well fold src into dst.
    if src_size[axis] == 1 {
        return host_fold(policy, src, dst, reduction);
    }

    let mut src_strides = src.strides_in_bytes();
    let mut dst_strides = dst.strides_in_bytes();
    let mut counts = dst.extents().to_i64().unwrap();

    src_strides.swap(0, axis);
    dst_strides.swap(0, axis);
    counts.swap(0, axis);

    // Dirty trick which allows us to simplify all strides after axis 0.
    let (_, [dst_offset, src_offset]) = {
        let [_, dst_strides @ ..] = &mut *dst_strides;
        let [_, src_strides @ ..] = &mut *src_strides;
        let [_, counts @ ..] = &mut counts;
        simplify_strides([dst_strides, src_strides], counts)
    };

    let axis_length = counts[0];
    let [_, counts @ ..] = counts;
    let src_ptr = UnsafeSendable(src.as_ptr().wrapping_offset(src_offset as isize));
    let dst_ptr = UnsafeSendable(dst.as_ptr_mut().wrapping_offset(dst_offset as isize));

    host_recur(
        policy,
        host::count_bytes(&dst),
        [0; 2],
        counts,
        &move |p, q| {
            reduce_leaf(
                p,
                q,
                axis_length,
                src_ptr.0,
                src_strides,
                dst_ptr.0,
                dst_strides,
                reduction,
            )
        },
    );

    Ok(())
}

unsafe fn reduce_leaf(
    start: [i64; 2],
    end: [i64; 2],
    axis_length: i64,
    src_ptr: *const u8,
    src_strides: ByteStrides,
    dst_ptr: *mut u8,
    dst_strides: ByteStrides,
    reduction: Reduction,
) {
    impl_reductions!(reduction, |combine| reduce_leaf_spec(
        start,
        end,
        axis_length,
        src_ptr,
        src_strides,
        dst_ptr,
        dst_strides,
        combine,
    ));
}

unsafe fn reduce_leaf_spec<F: Fn(T, T) -> T, T>(
    start: [i64; 2],
    end: [i64; 2],
    axis_length: i64,
    src_ptr: *const u8,
    src_strides: ByteStrides,
    dst_ptr: *mut u8,
    dst_strides: ByteStrides,
    combine: F,
) {
    for i2 in start[1]..end[1] {
        for i1 in start[0]..end[0] {
            let src_ptr = src_ptr.wrapping_offset(src_strides.offset_in_bytes([0, i1, i2]));
            let mut value = ptr::read(src_ptr as *const T);

            for i0 in 1..axis_length {
                let src_ptr = src_ptr.wrapping_offset(src_strides.offset_in_bytes([i0, i1, i2]));
                let reduce = ptr::read(src_ptr as *const T);
                value = combine(value, reduce);
            }

            let dst = dst_ptr.wrapping_offset(dst_strides.offset_in_bytes([0, i1, i2]));
            let reduce = ptr::read(dst as *const T);
            value = combine(reduce, value);
            ptr::write(dst as *mut T, value);
        }
    }
}
