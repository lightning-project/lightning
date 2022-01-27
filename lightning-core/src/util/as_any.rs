use std::any::{type_name, Any, TypeId};

pub trait AsAny: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn type_name(&self) -> &'static str;
}

impl<T: Any> AsAny for T {
    #[inline(always)]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline(always)]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    #[inline(always)]
    fn type_name(&self) -> &'static str {
        type_name::<T>()
    }
}

pub trait Downcast: AsAny {
    #[inline(always)]
    fn is<T: Any>(&self) -> bool {
        self.type_id() == TypeId::of::<T>()
    }

    #[inline(always)]
    fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }

    #[inline(always)]
    fn downcast_mut<T: Any>(&mut self) -> Option<&mut T> {
        self.as_any_mut().downcast_mut::<T>()
    }
}
