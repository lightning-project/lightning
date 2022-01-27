use crate::host::{host_recur, Policy, UnsafeSendable};
use crate::{host, simplify_strides};
use lightning_core::accessor::{ByteStrides, HostMutAccessor};
use lightning_core::{DataValue, MAX_DIMS};
use std::convert::TryInto;
use std::ptr::copy_nonoverlapping;

pub unsafe fn host_fill(policy: impl Policy, dst: HostMutAccessor, value: DataValue) {
    assert_eq!(dst.data_type(), dst.data_type());
    let mut strides = dst.strides_in_bytes();
    let mut counts = dst.extents().to_i64().unwrap();

    let (_, [dst_offset]) = simplify_strides([&mut strides], &mut counts);
    let ptr = UnsafeSendable(dst.as_ptr_mut().wrapping_offset(dst_offset as isize));

    let value = value.as_raw_data();

    host_recur(
        policy,
        host::count_bytes(&dst),
        [0; MAX_DIMS],
        counts,
        &move |p, q| fill_leaf(p, q, ptr.0, strides, value),
    );
}

unsafe fn fill_leaf(
    start: [i64; MAX_DIMS],
    end: [i64; MAX_DIMS],
    dst: *mut u8,
    strides: ByteStrides,
    value: &[u8],
) {
    macro_rules! spec {
        ($n:expr) => {
            if let Ok(v) = value.try_into() {
                fill_leaf_spec::<[u8; $n]>(start, end, dst, strides, v);
                return;
            }
        };
    }

    spec!(1);
    spec!(2);
    spec!(4);
    spec!(8);
    spec!(16);

    fill_leaf_spec(start, end, dst, strides, value);
}

#[inline]
unsafe fn fill_leaf_spec<D: AsRef<[u8]>>(
    start: [i64; MAX_DIMS],
    end: [i64; MAX_DIMS],
    dst: *mut u8,
    strides: ByteStrides,
    value: D,
) {
    for i2 in start[2]..end[2] {
        for i1 in start[1]..end[1] {
            for i0 in start[0]..end[0] {
                let dst = dst.wrapping_offset(strides.offset_in_bytes([i0, i1, i2]));

                let value = value.as_ref();
                copy_nonoverlapping(value.as_ptr(), dst, value.len());
            }
        }
    }
}
