use std::cell::UnsafeCell;
use std::rc::{Rc, Weak};
use std::{mem, ptr};

#[derive(Debug)]
pub struct CloneCell<T: ?Sized> {
    inner: UnsafeCell<T>,
}

impl<T> CloneCell<T> {
    pub fn new(inner: T) -> Self {
        CloneCell {
            inner: UnsafeCell::new(inner),
        }
    }

    pub fn get(&self) -> T
    where
        T: CloneSafeCell,
    {
        unsafe { &*self.inner.get() }.clone()
    }

    pub fn get_mut(&mut self) -> &mut T {
        unsafe { &mut *self.inner.get() }
    }

    pub fn take(&self) -> T
    where
        T: Default,
    {
        self.replace(T::default())
    }

    pub fn swap(&self, other: &Self) {
        unsafe { ptr::swap(self.inner.get(), other.inner.get()) }
    }

    pub fn set(&self, value: T) {
        self.replace(value);
    }

    pub fn replace(&self, value: T) -> T {
        mem::replace(unsafe { &mut *self.inner.get() }, value)
    }

    pub fn into_inner(self) -> T {
        self.inner.into_inner()
    }
}

//unsafe impl<T: ?Sized>  !Sync for Cell<T> {}
unsafe impl<T: ?Sized + Send> Send for CloneCell<T> {}

impl<T> Clone for CloneCell<T>
where
    T: CloneSafeCell,
{
    fn clone(&self) -> Self {
        CloneCell::new(self.get())
    }
}

impl<T> Default for CloneCell<T>
where
    T: Default,
{
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T> From<T> for CloneCell<T> {
    fn from(inner: T) -> Self {
        Self::new(inner)
    }
}

impl<T> CloneCell<Vec<T>> {
    pub fn index(&self, index: usize) -> Option<T>
    where
        T: CloneSafeCell,
    {
        unsafe { &mut *self.inner.get() }.get(index).cloned()
    }

    #[inline]
    pub fn push(&self, item: T) {
        unsafe { &mut *self.inner.get() }.push(item)
    }

    #[inline]
    pub fn pop(&self) -> Option<T> {
        unsafe { &mut *self.inner.get() }.pop()
    }

    #[inline]
    pub fn len(&self) -> usize {
        unsafe { &*self.inner.get() }.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        unsafe { &*self.inner.get() }.is_empty()
    }
}

impl<T: ?Sized> CloneCell<Rc<T>> {
    pub fn downgrade(&self) -> Weak<T> {
        Rc::downgrade(unsafe { &*self.inner.get() })
    }
}

impl<T> CloneCell<Option<Rc<T>>> {
    pub fn downgrade(&self) -> Weak<T> {
        match unsafe { &*self.inner.get() } {
            Some(r) => Rc::downgrade(r),
            None => Weak::new(),
        }
    }
}

impl<T: ?Sized> CloneCell<Weak<T>> {
    pub fn upgrade(&self) -> Option<Rc<T>> {
        unsafe { &*self.inner.get() }.upgrade()
    }
}

impl<T> CloneCell<Option<T>> {
    pub fn is_some(&self) -> bool {
        unsafe { &*self.inner.get() }.is_some()
    }

    pub fn is_none(&self) -> bool {
        !self.is_some()
    }
}

pub unsafe trait CloneSafeCell: Clone {}
unsafe impl CloneSafeCell for String {}
unsafe impl<T> CloneSafeCell for Option<T> where T: CloneSafeCell {}
unsafe impl<T, E> CloneSafeCell for Result<T, E>
where
    T: CloneSafeCell,
    E: CloneSafeCell,
{
}
unsafe impl<T: ?Sized> CloneSafeCell for Rc<T> {}
unsafe impl<T: ?Sized> CloneSafeCell for Weak<T> {}
unsafe impl<T: ?Sized> CloneSafeCell for Box<T> where T: CloneSafeCell {}
unsafe impl<T: ?Sized> CloneSafeCell for Vec<T> where T: CloneSafeCell {}
