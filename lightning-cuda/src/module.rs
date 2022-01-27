//! Loading CUDA modules and calling kernels.

use crate::{cuda_call, cuda_check, Result, Stream};
use cuda_driver_sys::*;
use std::convert::{TryFrom, TryInto};
use std::ffi::{c_void, CStr};
use std::fmt;
use std::marker::PhantomData;
use std::{mem, ptr};

#[repr(transparent)]
pub struct Module(CUmodule);

impl fmt::Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("CudaModule").field(&self.0).finish()
    }
}

impl Module {
    pub fn load_file(file_name: &CStr) -> Result<Self> {
        unsafe { cuda_call(|m| cuModuleLoad(m, file_name.as_ptr())).map(Self) }
    }

    pub unsafe fn load_image(image: *const c_void) -> Result<Self> {
        {
            cuda_call(|m| cuModuleLoadData(m, image)).map(Self)
        }
    }

    pub unsafe fn load_fatbinary(image: *const c_void) -> Result<Self> {
        {
            cuda_call(|m| cuModuleLoadFatBinary(m, image)).map(Self)
        }
    }

    pub fn function(&self, name: &CStr) -> Result<Function<'_>> {
        unsafe {
            let fun = cuda_call(|f| cuModuleGetFunction(f, self.0, name.as_ptr()))?;
            Ok(Function::from_raw(fun))
        }
    }

    #[inline(always)]
    pub fn into_raw(self) -> CUmodule {
        let out = self.0;
        mem::forget(self);
        out
    }

    #[inline(always)]
    pub unsafe fn from_raw(module: CUmodule) -> Self {
        Self(module)
    }

    #[inline(always)]
    pub fn raw(&self) -> CUmodule {
        self.0
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            cuModuleUnload(self.0);
        }
    }
}

#[repr(transparent)]
pub struct Function<'a> {
    fun: CUfunction,
    phantom: PhantomData<&'a Module>,
}

impl<'a> fmt::Debug for Function<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("CudaFunction").field(&self.fun).finish()
    }
}

impl<'a> Function<'a> {
    pub fn attribute(&self, attr: FunctionAttribute) -> Result<i32> {
        unsafe {
            let attr = mem::transmute::<_, CUfunction_attribute_enum>(attr);
            cuda_call(|v| cuFuncGetAttribute(v, attr, self.fun))
        }
    }

    pub fn set_attribute(&mut self, attr: FunctionAttribute, value: i32) -> Result {
        unsafe {
            let attr = mem::transmute::<_, CUfunction_attribute_enum>(attr);
            cuda_check(cuFuncSetAttribute(self.fun, attr, value))
        }
    }

    pub unsafe fn launch_async<G, B>(
        &self,
        stream: &Stream,
        grid: G,
        block: B,
        smem_size: u32,
        arguments: &[*const ()],
    ) -> Result
    where
        G: Into<Dim3>,
        B: Into<Dim3>,
    {
        let grid = grid.into();
        let block = block.into();
        let mut arguments = arguments.to_vec();
        arguments.push(std::ptr::null());

        cuda_check(cuLaunchKernel(
            self.fun,
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            smem_size,
            stream.raw(),
            arguments.as_mut_ptr() as *mut *mut c_void,
            ptr::null_mut(),
        ))
    }

    pub unsafe fn from_raw(fun: CUfunction) -> Self {
        Function {
            fun,
            phantom: PhantomData,
        }
    }

    pub fn raw(&self) -> CUfunction {
        self.fun
    }
}

#[repr(C)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, Debug)]
pub struct Dim3(pub u32, pub u32, pub u32);

impl Dim3 {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Dim3(x, y, z)
    }

    pub fn x(&self) -> u32 {
        self.0
    }

    pub fn y(&self) -> u32 {
        self.1
    }

    pub fn z(&self) -> u32 {
        self.2
    }
}

impl From<u32> for Dim3 {
    fn from(v: u32) -> Self {
        Dim3::new(v, 1, 1)
    }
}

impl From<&u32> for Dim3 {
    fn from(v: &u32) -> Self {
        Dim3::new(*v, 1, 1)
    }
}

impl From<(u32,)> for Dim3 {
    fn from(v: (u32,)) -> Self {
        Dim3::new(v.0, 1, 1)
    }
}

impl From<[u32; 1]> for Dim3 {
    fn from(v: [u32; 1]) -> Self {
        Dim3::new(v[0], 1, 1)
    }
}

impl From<(u32, u32)> for Dim3 {
    fn from(v: (u32, u32)) -> Self {
        Dim3::new(v.0, v.1, 1)
    }
}

impl From<[u32; 2]> for Dim3 {
    fn from(v: [u32; 2]) -> Self {
        Dim3::new(v[0], v[1], 1)
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    fn from(v: (u32, u32, u32)) -> Self {
        Dim3::new(v.0, v.1, v.2)
    }
}

impl From<[u32; 3]> for Dim3 {
    fn from(v: [u32; 3]) -> Self {
        Dim3::new(v[0], v[1], v[2])
    }
}

macro_rules! impl_try_into {
    ($T:ident) => {
        impl TryFrom<$T> for Dim3 {
            type Error = <$T as TryInto<u32>>::Error;

            fn try_from(x: $T) -> Result<Self, Self::Error> {
                match x.try_into() {
                    Ok(x) => Ok(Self(x, 1, 1)),
                    Err(e) => Err(e),
                }
            }
        }

        impl TryFrom<[$T; 1]> for Dim3 {
            type Error = <$T as TryInto<u32>>::Error;

            fn try_from([x]: [$T; 1]) -> Result<Self, Self::Error> {
                match x.try_into() {
                    Ok(x) => Ok(Self(x, 1, 1)),
                    Err(e) => Err(e),
                }
            }
        }

        impl TryFrom<[$T; 2]> for Dim3 {
            type Error = <$T as TryInto<u32>>::Error;

            fn try_from([x, y]: [$T; 2]) -> Result<Self, Self::Error> {
                match (x.try_into(), y.try_into()) {
                    (Ok(x), Ok(y)) => Ok(Self(x, y, 1)),
                    (Err(e), _) | (_, Err(e)) => Err(e),
                }
            }
        }

        impl TryFrom<[$T; 3]> for Dim3 {
            type Error = <$T as TryInto<u32>>::Error;

            fn try_from([x, y, z]: [$T; 3]) -> Result<Self, Self::Error> {
                match (x.try_into(), y.try_into(), z.try_into()) {
                    (Ok(x), Ok(y), Ok(z)) => Ok(Self(x, y, z)),
                    (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => Err(e),
                }
            }
        }
    };
}

impl_try_into!(i32);
impl_try_into!(i64);
impl_try_into!(isize);
//impl_try_into!(i32);
impl_try_into!(u64);
impl_try_into!(usize);

use CUfunction_attribute_enum::*;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, Debug)]
#[allow(non_camel_case_types)]
#[repr(u32)]
pub enum FunctionAttribute {
    MAX_THREADS_PER_BLOCK = CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK as u32,
    SHARED_SIZE_BYTES = CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES as u32,
    CONST_SIZE_BYTES = CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES as u32,
    LOCAL_SIZE_BYTES = CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES as u32,
    NUM_REGS = CU_FUNC_ATTRIBUTE_NUM_REGS as u32,
    PTX_VERSION = CU_FUNC_ATTRIBUTE_PTX_VERSION as u32,
    BINARY_VERSION = CU_FUNC_ATTRIBUTE_BINARY_VERSION as u32,
    CACHE_MODE_CA = CU_FUNC_ATTRIBUTE_CACHE_MODE_CA as u32,
    MAX_DYNAMIC_SHARED_SIZE_BYTES = CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES as u32,
    PREFERRED_SHARED_MEMORY_CARVEOUT = CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT as u32,
}
