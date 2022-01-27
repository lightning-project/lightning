/*!
This crate offers transparent rustic wrappers around the CUDA driver API.

# What is this?
This crate offers Rust versions of functions and types from the CUDA driver API. It assumes the user
is aware of how the CUDA API looks like, but wants slightly more convenience when working with CUDA.

The goal of this crate is to be as light-weight as possible: most types are a simple wrappers around
CUDA types and most functions simply forward the call directly to a CUDA function.

For example, [`Stream`] is a newtype around a `CUstream`, [`Stream::new`] simply calls
`cuStreamCreate`, and `Drop` simply calls `cuStreamDestroy`, and [`Stream::raw`] returns the underlying
 `CUstream` when needed for FFI. The should be no runtime overhead of using these methods opposed
 to the raw functions.

[`Stream`]: stream/struct.Stream.html
[`Stream::new`]: stream/struct.Stream.html#method.new

# Why choose cuba over other CUDA crates?
There exist many excellent CUDA crates already. The design of this crate is similar to that of
[RustaCUDA]. In fact, CUBA was born out of minor gripes with RustaCUDA's design, most notably related
to FFI interaction, thread-safety, and launching kernels. Although this code in this crate was
written from scratch, in some sense, it can be seen as a "spriritual" fork of RustaCUDA.

[RustaCUDA]: https://crates.io/crates/rustacuda

Some are other crates worth considering when working with CUDA.

* Low-level bindings
  * [cuda](https://crates.io/crates/cuda)
  * [cuda-driver-sys](https://crates.io/crates/cuda-driver-sys)
  * [cuda-runtime-sys](https://crates.io/crates/cuda-runtime-sys)
  * [cuda-sys](https://crates.io/crates/cuda-sys)
  * [cu-sys](https://crates.io/crates/cu-sys)
  * [cudart](https://crates.io/crates/cudart)
  * [cu](https://crates.io/crates/cu)
* High-level abstractions
  * [RustaCUDA](https://crates.io/crates/rustacuda)
  * [accel](https://crates.io/crates/accel)
  * [coaster](https://crates.io/crates/coaster)
  * [arrayfire](https://crates.io/crates/arrayfire)
  * [collenchyma](https://crates.io/crates/collenchyma)


# Unsafe and FFI
This crate allows safe functions where possible, but is not afraid to expose unsafe CUDA
functionality which cannot be made safe without additional overhead. For example, [`copy`]
allows copying between buffers if this can be done safely, while the unsafe variant [`copy_raw`]
performs no safety checks.

[`copy`]: copy/fn.copy.html
[`copy_raw`]: copy/fn.copy_raw.html


Additionally, most types have a `raw` method which exposes the underlying CUDA type (for example, [`Stream::raw`]
returns the `CUstream`) and an unsafe `from_raw` method which allows constructing a Rust type from
a CUDA type (for example, [`Stream::from_raw`] which takes a `CUstream`). This enables FFI
integration without existing libraries which work with raw CUDA types.

[`Stream::raw`]: stream/struct.Stream.html#method.raw
[`Stream::from_raw`]: stream/struct.Stream.html#method.from_raw


# Error handling
Nearly all functions in the CUDA driver API return an status code indicating whether the operation
was successful. This crate mimics this behavior by returning a [`CudaResult`] which can be either
`Ok(_)` on success or a [`CudaError`] on failure.

[`CudaResult`]: error/type.Result.html
[`CudaError`]: error/struct.Error.html


# Usage
The preferred way of using this crate is importing everything from [`prelude`]. The prelude
exposes all items prefixed with "Cuda" to prevent name collisions. (for example, [`Stream`]
becomes `CudaStream`).

[`prelude`]: prelude/index.html
[`Stream`]: stream/struct.Stream.html

```
use cuba::prelude::*;

let device = CudaDevice::nth(0)?;
let context = cuda_create_context(device, CudaContextFlags::empty())?;

// Use CudaStream, CudaEvent, CudaModule, CudaDeviceMem, etc...

unsafe { cuda_destroy_context(context)?; }
```

As an alternative, you could also use the `cuba` namespace directly, but this is more cluttered.

```
let device = cuba::Device::nth(0)?;
let context = cuba::create_context(device, cuba::ContextFlags::empty())?;

// Use cuba::Stream, cuba::Event, cuba::Module, cuba::DeviceMem, etc...

unsafe { cuba::destroy_context(context)?; }
```
*/

#![allow(dead_code)]
#![deny(
    missing_debug_implementations,
    bare_trait_objects,
    missing_copy_implementations,
    trivial_numeric_casts,
    unused_import_braces,
    unused_qualifications
)]
use cuda_driver_sys::*;
pub mod context;
pub mod copy;
pub mod device;
pub mod error;
pub mod event;
pub mod mem;
pub mod module;
pub mod prelude;
pub mod profiler;
pub mod stream;

pub use context::*;
pub use copy::*;
pub use device::*;
pub use error::*;
pub use event::*;
pub use mem::*;
pub use module::*;
pub use profiler::*;
pub use stream::*;

/// Initialize the CUDA runtime.
///
/// Must be called before calling any other CUDA function.
pub fn init() -> Result {
    unsafe { cuda_check(cuInit(0))? }

    for device in Device::all()? {
        if device.attribute(DeviceAttribute::UNIFIED_ADDRESSING)? == 0 {
            panic!(
                "device {:?} does not support UNIFIED_ADDRESSING, cuba assumes this is \
                    enabled across all CUDA-capable devices.",
                device
            );
        }
    }

    Ok(())
}

/// Returns the version of the CUDA driver API.
///
/// Returns a tuple `(major, minor)` indicating the version. For example, CUDA 9.2 would return `(9, 2)`.
pub fn version() -> Result<(u32, u32)> {
    let value = unsafe { cuda_call(|v| cuDriverGetVersion(v)) }? as u32;

    // According to the doc for cuDriverGetVersion, it returns 1000 * major + 10 * minor.
    Ok((value / 1000, (value % 1000) / 10))
}
