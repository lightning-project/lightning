use std::mem::{self, ManuallyDrop, MaybeUninit};

#[inline(always)]
pub fn generate<T, F, const N: usize>(mut fun: F) -> [T; N]
where
    F: FnMut(usize) -> T,
{
    // SAFETY: assume_init is safe since we convert "MaybeUninit<[T; N]>" to "[MaybeUninit<T>; N]"
    let mut output = unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() };

    for i in 0..N {
        // How to handle panics?
        output[i] = MaybeUninit::new(fun(i));
    }

    // SAFETY: transmute is safe since all items have been initialized
    unsafe { mem::transmute_copy(&output) }
}

#[inline(always)]
pub fn try_zip<L, R, T, F, E, const N: usize>(
    lhs: [L; N],
    rhs: [R; N],
    mut fun: F,
) -> Result<[T; N], E>
where
    F: FnMut(L, R) -> Result<T, E>,
{
    // SAFETY: transmute_copy to [ManuallyDrop<L>; N]" and forget lhs
    let mut lhs_items: [ManuallyDrop<L>; N] = unsafe { mem::transmute_copy(&lhs) };
    mem::forget(lhs);

    // SAFETY: transmute_copy to [ManuallyDrop<R>; N]" and forget rhs
    let mut rhs_items: [ManuallyDrop<R>; N] = unsafe { mem::transmute_copy(&rhs) };
    mem::forget(rhs);

    // SAFETY: assume_init is safe since we convert "MaybeUninit<[T; N]>" to "[MaybeUninit<T>; N]"
    let mut output = unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() };

    let mut len = 0;
    let mut result = Ok(());

    while len < N {
        // SAFETY: take is safe since must be initialized and we only read each entry once.
        let (l, r) = unsafe {
            (
                ManuallyDrop::take(&mut lhs_items[len]),
                ManuallyDrop::take(&mut rhs_items[len]),
            )
        };

        // How to handle panics?
        match fun(l, r) {
            Ok(val) => {
                output[len] = MaybeUninit::new(val);
                len += 1;
            }
            Err(e) => {
                result = Err(e);
                break;
            }
        }
    }

    match result {
        Ok(()) => {
            // SAFETY: transmute is safe since all items have been initialized
            Ok(unsafe { mem::transmute_copy(&output) })
        }
        Err(e) => {
            for i in 0..len {
                unsafe { output[i].as_ptr().read() };
            }

            for i in (len + 1)..N {
                unsafe { ManuallyDrop::drop(&mut lhs_items[i]) };
                unsafe { ManuallyDrop::drop(&mut rhs_items[i]) };
            }

            Err(e)
        }
    }
}

#[inline(always)]
pub fn zip<L, R, T, F, const N: usize>(lhs: [L; N], rhs: [R; N], mut fun: F) -> [T; N]
where
    F: FnMut(L, R) -> T,
{
    enum Infallible {}

    match try_zip(lhs, rhs, |l, r| Result::<_, Infallible>::Ok(fun(l, r))) {
        Ok(v) => v,
        Err(e) => match e {},
    }
}

#[inline(always)]
pub fn map<L, T, F, const N: usize>(input: [L; N], mut fun: F) -> [T; N]
where
    F: FnMut(L) -> T,
{
    zip(input, [(); N], |x, ()| fun(x))
}

#[inline(always)]
pub fn try_map<L, T, F, E, const N: usize>(input: [L; N], mut fun: F) -> Result<[T; N], E>
where
    F: FnMut(L) -> Result<T, E>,
{
    try_zip(input, [(); N], |x, ()| fun(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_generate() {
        let values = generate(|i| i);
        assert_eq!(values, [0, 1, 2, 3, 4]);

        let values = generate(|i| i.to_string());
        assert_eq!(values, ["0", "1", "2", "3", "4"])
    }

    #[test]
    fn test_array_zip() {
        let values = zip([0, 1, 2], [10, 20, 30], |x, y| x + y);
        assert_eq!(values, [10, 21, 32]);

        let values = zip(["a", "b", "c"], ["0", "1", "2"], |x, y| {
            format!("{}{}", x, y)
        });
        assert_eq!(values, ["a0", "b1", "c2"]);
    }

    #[test]
    fn test_array_map() {
        let values = map([0, 1, 2], |x| x * 10);
        assert_eq!(values, [0, 10, 20]);

        let values = map(["a", "b", "c"], |x| format!("{}!", x));
        assert_eq!(values, ["a!", "b!", "c!"]);
    }

    #[test]
    fn test_array_try_zip() {
        #[derive(Debug, PartialEq)]
        struct E;

        let values = try_zip([0u32, 1, 2], [10, 20, 30], |x, y| {
            u32::checked_sub(x, y).ok_or(E)
        });
        assert_eq!(values, Err(E));

        let values = try_zip([0u32, 1, 2], [10, 20, 30], |x, y| {
            u32::checked_sub(y, x).ok_or(E)
        });
        assert_eq!(values, Ok([10, 19, 28]));
    }

    #[test]
    fn test_array_try_map() {
        #[derive(Debug, PartialEq)]
        struct E;

        let values = try_map([0i32, 1, 2], |x| x.checked_neg().ok_or(E));
        assert_eq!(values, Ok([-0, -1, -2]));

        let values = try_map([0u32, 1, 2], |x| x.checked_neg().ok_or(E));
        assert_eq!(values, Err(E));
    }
}
