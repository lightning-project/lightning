use std::mem;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};

pub struct DropGuard<T, F: FnOnce(T)> {
    token: ManuallyDrop<T>,
    callback: ManuallyDrop<F>,
}

impl<T, F: FnOnce(T)> DropGuard<T, F> {
    pub fn new(token: T, dropper: F) -> Self {
        Self {
            token: ManuallyDrop::new(token),
            callback: ManuallyDrop::new(dropper),
        }
    }

    pub fn into_inner(mut self) -> T {
        let token = unsafe { ManuallyDrop::take(&mut self.token) };
        unsafe {
            ManuallyDrop::drop(&mut self.callback);
        }
        mem::forget(self);

        token
    }
}

impl<T, F: FnOnce(T)> Drop for DropGuard<T, F> {
    fn drop(&mut self) {
        let token = unsafe { ManuallyDrop::take(&mut self.token) };
        let callback = unsafe { ManuallyDrop::take(&mut self.callback) };

        (callback)(token);
    }
}

impl<T, F: FnOnce(T)> Deref for DropGuard<T, F> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.token
    }
}

impl<T, F: FnOnce(T)> DerefMut for DropGuard<T, F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.token
    }
}
