use crossbeam::atomic::AtomicCell;
use std::cell::Cell;
use std::num::NonZeroU64;

pub trait Counter {
    type Output;

    fn get_and_increment(self) -> Self::Output;
}

impl Counter for &mut u64 {
    type Output = u64;

    fn get_and_increment(self) -> Self::Output {
        let current = *self;
        *self = u64::checked_add(current, 1).expect("attempt to add with overflow");
        current
    }
}

impl Counter for &mut NonZeroU64 {
    type Output = NonZeroU64;

    fn get_and_increment(self) -> Self::Output {
        let old_value = *self;
        let mut new_value = old_value.get();
        new_value.get_and_increment();
        *self = unsafe { NonZeroU64::new_unchecked(new_value) };
        old_value
    }
}

impl<T: Copy + Eq> Counter for &AtomicCell<T>
where
    for<'b> &'b mut T: Counter<Output = T>,
{
    type Output = T;

    fn get_and_increment(self) -> Self::Output {
        let mut old_value = self.load();

        loop {
            let mut new_value = old_value;
            new_value.get_and_increment();

            match self.compare_exchange(old_value, new_value) {
                Ok(_) => return old_value,
                Err(v) => old_value = v,
            }
        }
    }
}

impl<T: Copy> Counter for &Cell<T>
where
    for<'b> &'b mut T: Counter<Output = T>,
{
    type Output = T;

    fn get_and_increment(self) -> Self::Output {
        let mut value = self.get();
        value.get_and_increment();
        self.replace(value)
    }
}
