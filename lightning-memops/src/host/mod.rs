use lightning_core::accessor::HostMutAccessor;
use lightning_core::prelude::*;

pub(crate) mod copy;
pub(crate) mod fill;
pub(crate) mod reduce;

const WORK_SPLIT_THRESHOLD: usize = 1024 * 4;

#[derive(Copy, Clone)]
pub(super) struct UnsafeSendable<T>(pub(super) T);
unsafe impl Send for UnsafeSendable<*mut u8> {}
unsafe impl Send for UnsafeSendable<*const u8> {}
unsafe impl Sync for UnsafeSendable<*mut u8> {}
unsafe impl Sync for UnsafeSendable<*const u8> {}

fn count_bytes(buffer: &HostMutAccessor) -> usize {
    buffer.extents().volume() as usize * buffer.data_type().size_in_bytes()
}

pub trait Policy: Sized {
    fn join<A, B, RA, RB>(self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce(Self) -> RA + Send,
        B: FnOnce(Self) -> RB + Send,
        RA: Send,
        RB: Send;
}

#[derive(Debug, Clone, Copy)]
pub struct RayonPolicy;
impl Policy for RayonPolicy {
    fn join<A, B, RA, RB>(self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce(Self) -> RA + Send,
        B: FnOnce(Self) -> RB + Send,
        RA: Send,
        RB: Send,
    {
        rayon::join(|| left(Self), || right(Self))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SequentialPolicy;
impl Policy for SequentialPolicy {
    fn join<A, B, RA, RB>(self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce(Self) -> RA + Send,
        B: FnOnce(Self) -> RB + Send,
        RA: Send,
        RB: Send,
    {
        (left(Self), right(Self))
    }
}

fn host_recur<P: Policy, const D: usize>(
    policy: P,
    work_amount: usize,
    start: [i64; D],
    end: [i64; D],
    callback: &(dyn Fn([i64; D], [i64; D]) + Sync),
) {
    let split_axis = (0..D).map(|i| end[i] - start[i]).position_max().unwrap();
    let half = (end[split_axis] - start[split_axis]) / 2;

    if work_amount > WORK_SPLIT_THRESHOLD && half >= 1 {
        let mut middle = start;
        middle[split_axis] += half;

        policy.join(
            move |policy| host_recur(policy, work_amount / 2, start, middle, callback),
            move |policy| host_recur(policy, work_amount / 2, middle, end, callback),
        );

        return;
    }

    callback(start, end);
}
