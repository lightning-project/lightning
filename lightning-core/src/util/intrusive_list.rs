use std::cell::Cell;
use std::fmt;
use std::mem::replace;
use std::ptr::NonNull;
use std::rc::Rc;

fn free_link<T>() -> Option<NonNull<T>> {
    Some(NonNull::dangling())
}

pub struct Link<T> {
    // Used as follows:
    // - Some(ptr): next item is `ptr`.
    // - None: no next item, this item is last in list
    // - Some(NonNull::dangling()): item is not part of any list.
    inner: Cell<Option<NonNull<T>>>,
}

impl<T> Link<T> {
    #[inline(always)]
    fn try_acquire(&self) -> bool {
        if self.inner.get() != free_link() {
            return false;
        }

        self.inner.set(None);
        true
    }

    #[inline(always)]
    fn release(&self) {
        self.inner.set(free_link());
    }

    #[inline(always)]
    fn set_next(&self, next: Option<NonNull<T>>) {
        self.inner.set(next);
    }

    #[inline(always)]
    fn get_next(&self) -> Option<NonNull<T>> {
        self.inner.get()
    }

    #[inline(always)]
    fn get_next_and_release(&self) -> Option<NonNull<T>> {
        self.inner.replace(free_link())
    }
}

impl<T> Default for Link<T> {
    fn default() -> Self {
        Self {
            inner: Cell::new(free_link()),
        }
    }
}

impl<T> fmt::Debug for Link<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<link>")
    }
}

pub struct List<A: LinkOp> {
    head: Option<NonNull<A::Item>>,
    tail: Option<NonNull<A::Item>>,
    adapter: A,
}

pub unsafe trait LinkOp {
    type Item;

    fn get_link<'a>(&self, this: &'a Self::Item) -> &'a Link<Self::Item>;
}

#[macro_export]
macro_rules! intrusive_list {
    ($name:ident for $item:ident.$field:ident) => {
        #[derive(Default)]
        struct $name;
        unsafe impl $crate::util::intrusive_list::LinkOp for $name {
            type Item = $item;

            #[inline(always)]
            fn get_link<'a>(
                &self,
                this: &'a Self::Item,
            ) -> &'a $crate::util::intrusive_list::Link<Self::Item> {
                &this.$field
            }
        }
    };
}

impl<T, A: LinkOp<Item = T>> List<A> {
    pub fn new(adapter: A) -> Self {
        Self {
            head: None,
            tail: None,
            adapter,
        }
    }

    pub fn push_back(&mut self, item: Rc<T>) {
        if self.try_push_back(item).is_err() {
            panic!("item cannot be added to linked list since it already part of another list")
        }
    }

    pub fn try_push_back(&mut self, item: Rc<T>) -> Result<(), Rc<T>> {
        if !self.adapter.get_link(&item).try_acquire() {
            return Err(item);
        }

        let ptr = NonNull::new(Rc::into_raw(item) as *mut T);
        let old_tail = replace(&mut self.tail, ptr);

        if let Some(old_tail) = old_tail {
            self.adapter
                .get_link(unsafe { old_tail.as_ref() })
                .set_next(ptr);
        } else {
            self.head = ptr;
        }

        Ok(())
    }

    pub fn pop_front(&mut self) -> Option<Rc<T>> {
        if let Some(head) = self.head {
            if let Some(new_head) = self
                .adapter
                .get_link(unsafe { head.as_ref() })
                .get_next_and_release()
            {
                self.head = Some(new_head);
            } else {
                self.head = None;
                self.tail = None;
            }

            let item = unsafe { Rc::from_raw(head.as_ptr()) };
            Some(item)
        } else {
            None
        }
    }

    pub fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    pub fn cursor(&self) -> Cursor<'_, A> {
        Cursor {
            list: self,
            cursor: self.head,
        }
    }

    pub fn iter(&self) -> Iter<'_, A> {
        Iter {
            cursor: self.cursor(),
        }
    }

    pub fn remove_if<F, G>(&mut self, mut condition: F, mut on_remove: G)
    where
        F: FnMut(&T) -> bool,
        G: FnMut(Rc<T>),
    {
        let mut prev: Option<NonNull<T>> = None;
        let mut cursor: Option<NonNull<T>> = self.head;

        while let Some(current) = cursor {
            let next = self
                .adapter
                .get_link(unsafe { current.as_ref() })
                .get_next();

            if condition(unsafe { current.as_ref() }) {
                if let Some(prev) = prev {
                    self.adapter
                        .get_link(unsafe { prev.as_ref() })
                        .set_next(next);
                } else {
                    self.head = next;
                }

                if next.is_none() {
                    self.tail = prev;
                }

                let item = unsafe { Rc::from_raw(current.as_ptr()) };
                self.adapter.get_link(&item).release();
                on_remove(item);
            } else {
                prev = Some(current);
            }

            cursor = next;
        }
    }

    pub fn remove(&mut self, needle: &Rc<T>) -> bool {
        let mut found = false;
        self.remove_if(
            move |other| std::ptr::eq(other, &**needle),
            |_| found = true,
        );
        found
    }
}

impl<A: LinkOp + Default> Default for List<A> {
    fn default() -> Self {
        Self {
            head: None,
            tail: None,
            adapter: A::default(),
        }
    }
}

impl<A: LinkOp> fmt::Debug for List<A>
where
    A::Item: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_list();
        for item in self {
            f.entry(item);
        }
        f.finish()
    }
}

impl<A: LinkOp> Drop for List<A> {
    fn drop(&mut self) {
        while let Some(_) = self.pop_front() {
            //
        }
    }
}

#[derive(Clone)]
pub struct Cursor<'a, A: LinkOp> {
    list: &'a List<A>,
    cursor: Option<NonNull<A::Item>>,
}

impl<'a, A: LinkOp> Cursor<'a, A> {
    fn get(&self) -> Option<&'a A::Item> {
        if let Some(ptr) = self.cursor {
            Some(unsafe { ptr.as_ref() })
        } else {
            None
        }
    }

    fn advance(&mut self) {
        if let Some(ptr) = self.cursor {
            self.cursor = self
                .list
                .adapter
                .get_link(unsafe { ptr.as_ref() })
                .get_next();
        }
    }
}

impl<'a, A: LinkOp> IntoIterator for &'a List<A> {
    type Item = &'a A::Item;
    type IntoIter = Iter<'a, A>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            cursor: self.cursor(),
        }
    }
}

pub struct Iter<'a, A: LinkOp> {
    cursor: Cursor<'a, A>,
}

impl<'a, A: LinkOp> Iterator for Iter<'a, A> {
    type Item = &'a A::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.cursor.get() {
            self.cursor.advance();
            Some(item)
        } else {
            None
        }
    }
}

pub struct IntoIter<A: LinkOp> {
    list: List<A>,
}

impl<A: LinkOp> Iterator for IntoIter<A> {
    type Item = Rc<A::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        self.list.pop_front()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    struct Foo {
        value: u32,
        next: Link<Foo>,
    }

    impl Foo {
        fn new(value: u32) -> Self {
            Self {
                value,
                next: Link::default(),
            }
        }
    }

    intrusive_list!(FooAdaptor for Foo.next);

    #[test]
    fn test_basic() {
        let mut list = List::new(FooAdaptor);

        list.push_back(Rc::new(Foo::new(123)));
        assert_eq!(list.pop_front().unwrap().value, 123);
        assert!(list.pop_front().is_none());
    }

    #[test]
    fn test_complex() {
        let mut list = List::new(FooAdaptor);

        list.push_back(Rc::new(Foo::new(1)));
        list.push_back(Rc::new(Foo::new(2)));
        list.push_back(Rc::new(Foo::new(3)));
        list.push_back(Rc::new(Foo::new(4)));
        assert_eq!(list.pop_front().unwrap().value, 1);

        list.push_back(Rc::new(Foo::new(5)));
        assert_eq!(list.pop_front().unwrap().value, 2);
        assert_eq!(list.pop_front().unwrap().value, 3);
        assert_eq!(list.pop_front().unwrap().value, 4);

        list.push_back(Rc::new(Foo::new(6)));
        assert_eq!(list.pop_front().unwrap().value, 5);
        assert_eq!(list.pop_front().unwrap().value, 6);

        assert!(list.pop_front().is_none());
    }

    #[test]
    fn test_iter() {
        let mut list = List::new(FooAdaptor);
        list.push_back(Rc::new(Foo::new(1)));
        list.push_back(Rc::new(Foo::new(2)));
        list.push_back(Rc::new(Foo::new(3)));
        list.push_back(Rc::new(Foo::new(4)));

        let mut cursor = list.cursor();
        let mut numbers = vec![];
        while let Some(v) = cursor.get() {
            cursor.advance();
            numbers.push(v.value);
        }
        assert_eq!(numbers, [1, 2, 3, 4]);

        let numbers: Vec<_> = list.iter().map(|r| r.value).collect();
        assert_eq!(numbers, [1, 2, 3, 4]);

        let numbers: Vec<_> = list.into_iter().map(|r| r.value).collect();
        assert_eq!(numbers, [1, 2, 3, 4]);
    }

    #[test]
    fn test_refcount() {
        let items = [
            Rc::new(Foo::new(1)),
            Rc::new(Foo::new(2)),
            Rc::new(Foo::new(3)),
        ];

        let check = |expected: [usize; 3]| {
            let gotten: Vec<_> = items.iter().map(|r| Rc::strong_count(r)).collect();
            assert_eq!(gotten, expected);
        };

        check([1, 1, 1]);

        {
            let mut list = List::new(FooAdaptor);
            list.push_back(Rc::clone(&items[0]));
            check([2, 1, 1]);

            list.push_back(Rc::clone(&items[1]));
            check([2, 2, 1]);

            let _ = list.pop_front();
            check([1, 2, 1]);

            list.push_back(Rc::clone(&items[2]));
            check([1, 2, 2]);

            let _ = list.pop_front();
            check([1, 1, 2]);

            list.push_back(Rc::clone(&items[0]));
            check([2, 1, 2]);
        }

        check([1, 1, 1]);
    }

    #[test]
    #[should_panic]
    fn test_panic() {
        let item = Rc::new(Foo::new(123));

        let mut list = List::new(FooAdaptor);
        list.push_back(Rc::clone(&item));
        list.push_back(Rc::clone(&item));
    }
}
