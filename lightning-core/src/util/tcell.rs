pub struct TCell<Q, T> {
    inner: qcell::TCell<Q, T>,
}

impl<Q, T> TCell<Q, T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: qcell::TCell::new(value),
        }
    }

    #[inline(always)]
    pub fn borrow<'a>(&'a self, owner: &'a TCellOwner<Q>) -> &'a T {
        owner.inner.ro(&self.inner)
    }

    #[inline(always)]
    pub fn borrow_mut<'a>(&'a self, owner: &'a mut TCellOwner<Q>) -> &'a mut T {
        owner.inner.rw(&self.inner)
    }
}

pub struct TCellOwner<Q: 'static> {
    inner: qcell::TCellOwner<Q>,
}

impl<Q: 'static> TCellOwner<Q> {
    pub fn new() -> Self {
        Self {
            inner: qcell::TCellOwner::new(),
        }
    }
}
