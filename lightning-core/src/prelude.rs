pub use anyhow::{anyhow, bail, Context as _, Error};
pub use itertools::{all, any, enumerate, rev as reversed, zip, Itertools as _};
pub use log::{debug, error, info, trace, warn};
pub use parking_lot::Mutex;
pub use std::cmp::{max, min};
pub use std::convert::{TryFrom, TryInto};
pub use std::error::Error as StdError;
pub use std::iter::FromIterator as _;
pub use std::mem::{forget, replace, swap, take};
pub use thiserror::Error;

pub type HashSet<K> = std::collections::HashSet<K, fxhash::FxBuildHasher>;
pub type HashMap<K, V> = std::collections::HashMap<K, V, fxhash::FxBuildHasher>;
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, fxhash::FxBuildHasher>;
pub type IndexSet<K> = indexmap::IndexSet<K, fxhash::FxBuildHasher>;
pub type Result<T = (), E = Error> = std::result::Result<T, E>;

#[inline(always)]
pub fn default<T: Default>() -> T {
    T::default()
}

pub fn new_boxed_slice<T, F>(n: usize, fun: F) -> Box<[T]>
where
    F: FnMut() -> T,
{
    let mut result = Vec::with_capacity(n);
    result.resize_with(n, fun);
    result.into_boxed_slice()
}

#[macro_export]
macro_rules! try_block {
    { $($token:tt)* } => {{
        let l = || {
            $($token)*
        };
        l()
    }}
}
