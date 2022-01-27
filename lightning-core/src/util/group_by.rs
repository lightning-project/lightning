use std::marker::PhantomData;
use std::mem::replace;

pub struct GroupByKey<'a, T: 'a, F, K> {
    slice: &'a [T],
    extraction: F,
    phantom: PhantomData<fn(&'a T) -> K>,
}

impl<'a, T: 'a, F, K: 'a> Iterator for GroupByKey<'a, T, F, K>
where
    F: FnMut(&T) -> K,
    K: PartialEq,
{
    type Item = (K, &'a [T]);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let mut iter = self.slice.iter();

        if let Some(first) = iter.next() {
            let key = (self.extraction)(first);
            let mut len = 1;
            for item in iter {
                if key == (self.extraction)(item) {
                    len += 1;
                } else {
                    break;
                }
            }

            let (head, tail) = unsafe {
                (
                    self.slice.get_unchecked(..len),
                    self.slice.get_unchecked(len..),
                )
            };

            self.slice = tail;
            Some((key, head))
        } else {
            None
        }
    }
}

pub struct GroupByKeyMut<'a, T: 'a, F, K> {
    slice: &'a mut [T],
    extraction: F,
    phantom: PhantomData<fn(&'a T) -> K>,
}

impl<'a, T: 'a, F, K: 'a> Iterator for GroupByKeyMut<'a, T, F, K>
where
    F: FnMut(&T) -> K,
    K: PartialEq,
{
    type Item = (K, &'a mut [T]);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let mut iter = self.slice.iter();

        if let Some(first) = iter.next() {
            let key = (self.extraction)(first);
            let mut len = 1;
            for item in iter {
                if key == (self.extraction)(item) {
                    len += 1;
                } else {
                    break;
                }
            }

            let (head, tail) = replace(&mut self.slice, &mut []).split_at_mut(len);
            self.slice = tail;
            Some((key, head))
        } else {
            None
        }
    }
}

pub trait GroupByExt {
    type Item;

    /// Group consecutive elements of a slice which map to the same key.
    fn group_by_key<F, K>(&self, extraction: F) -> GroupByKey<'_, Self::Item, F, K>
    where
        F: FnMut(&Self::Item) -> K;

    /// Group consecutive elements of a mutable slice which map to the same key.
    fn group_by_key_mut<F, K>(&mut self, extraction: F) -> GroupByKeyMut<'_, Self::Item, F, K>
    where
        F: FnMut(&Self::Item) -> K;

    /// Sorts and then groups all elements which map to the same key.
    fn sort_and_group_by_key<F, K>(&mut self, extraction: F) -> GroupByKeyMut<'_, Self::Item, F, K>
    where
        F: FnMut(&Self::Item) -> K,
        K: Ord;
}

impl<T> GroupByExt for [T] {
    type Item = T;

    fn group_by_key<F, K>(&self, extraction: F) -> GroupByKey<'_, T, F, K>
    where
        F: FnMut(&T) -> K,
    {
        GroupByKey {
            slice: self,
            extraction,
            phantom: PhantomData,
        }
    }

    fn group_by_key_mut<F, K>(&mut self, extraction: F) -> GroupByKeyMut<'_, T, F, K>
    where
        F: FnMut(&T) -> K,
    {
        GroupByKeyMut {
            slice: self,
            extraction,
            phantom: PhantomData,
        }
    }

    fn sort_and_group_by_key<F, K>(&mut self, mut extraction: F) -> GroupByKeyMut<'_, T, F, K>
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.sort_by_key(&mut extraction);
        self.group_by_key_mut(extraction)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_group_by_key() {
        let items = &[3, 3, 3, 2, 2, 5, 5, 5, 5, 5, 1, 2, 2];
        let mut iter = items.group_by_key(|&v| v);

        assert_eq!(iter.next(), Some((3, &[3, 3, 3][..])));
        assert_eq!(iter.next(), Some((2, &[2, 2][..])));
        assert_eq!(iter.next(), Some((5, &[5, 5, 5, 5, 5][..])));
        assert_eq!(iter.next(), Some((1, &[1][..])));
        assert_eq!(iter.next(), Some((2, &[2, 2][..])));
        assert_eq!(iter.next(), None);

        let items = &mut [3, 3, 3, 2, 2, 5, 5, 5, 5, 5, 1, 2, 2];
        let mut iter = items.group_by_key_mut(|&v| v);

        assert_eq!(iter.next(), Some((3, &mut [3, 3, 3][..])));
        assert_eq!(iter.next(), Some((2, &mut [2, 2][..])));
        assert_eq!(iter.next(), Some((5, &mut [5, 5, 5, 5, 5][..])));
        assert_eq!(iter.next(), Some((1, &mut [1][..])));
        assert_eq!(iter.next(), Some((2, &mut [2, 2][..])));
        assert_eq!(iter.next(), None);

        let items = &mut [3, 3, 3, 2, 2, 5, 5, 5, 5, 5, 1, 2, 2];
        let mut iter = items.sort_and_group_by_key(|&v| v);

        assert_eq!(iter.next(), Some((1, &mut [1][..])));
        assert_eq!(iter.next(), Some((2, &mut [2, 2, 2, 2][..])));
        assert_eq!(iter.next(), Some((3, &mut [3, 3, 3][..])));
        assert_eq!(iter.next(), Some((5, &mut [5, 5, 5, 5, 5][..])));
        assert_eq!(iter.next(), None);
    }
}
