use super::{Key, ZEROS};
use crate::{simplify_strides, MemOpsKernelsCache};
use cuda_driver_sys::{cuMemsetD2D16Async, cuMemsetD2D32Async, cuMemsetD2D8Async};
use lightning_codegen::{make_valid_ident, KernelArg, KernelParam, CPP_NAMESPACE};
use lightning_core::accessor::{CudaMutAccessor, Strides};
use lightning_core::prelude::*;
use lightning_core::{prelude, DataType, DataValue, MAX_DIMS};
use lightning_cuda::{
    cuda_check, ContextHandle as CudaContextHandle, DevicePtr as CudaDevicePtr,
    Stream as CudaStream,
};

pub unsafe fn cuda_fill(
    context: CudaContextHandle,
    stream: &CudaStream,
    kernels: &MemOpsKernelsCache,
    dst: CudaMutAccessor,
    value: DataValue,
) -> prelude::Result {
    let dtype = value.data_type();
    assert_eq!(dst.data_type(), dtype);

    let elem_len = dtype.size_in_bytes();
    let mut strides = dst.strides();
    let mut counts = dst.extents().to_i64().unwrap();

    let (ndims, [dst_offset]) = simplify_strides([&mut strides], &mut counts);
    let dst_ptr = dst
        .as_ptr_mut()
        .offset_bytes(dst_offset as isize * elem_len as isize);

    let mut pattern = value.as_raw_data();
    let first_contiguous = strides[0] == 1;

    if ndims == 1 || (ndims == 2 && first_contiguous) {
        let byte_strides = strides.to_byte_strides(dtype);
        let stream = stream.raw();
        let mut width = counts[0] as usize;
        let mut height = counts[1] as usize;
        let mut pitch = byte_strides[1] as usize;

        if !first_contiguous {
            assert_eq!(ndims, 1);
            width = 1;
            height = counts[0] as usize;
            pitch = byte_strides[0] as usize;
        }

        let alignment = dtype.alignment();
        assert_eq!(pattern.len(), dtype.size_in_bytes());
        assert_eq!(pattern.len() % alignment, 0);

        for &n in &[1, 2, 4] {
            if alignment % n == 0 && all(enumerate(pattern), |(i, &v)| v == pattern[i % n]) {
                width *= pattern.len() / n;
                pattern = &pattern[..n];
                break;
            }
        }

        // Attempt 1 byte fill
        if let Ok(pattern) = pattern.try_into() {
            cuda_check(cuMemsetD2D8Async(
                dst_ptr.raw(),
                pitch,
                u8::from_ne_bytes(pattern),
                width,
                height,
                stream,
            ))?;

            return Ok(());
        }

        // Attempt 2 byte fill
        if let Ok(pattern) = pattern.try_into() {
            cuda_check(cuMemsetD2D16Async(
                dst_ptr.raw(),
                pitch,
                u16::from_ne_bytes(pattern),
                width,
                height,
                stream,
            ))?;

            return Ok(());
        }

        // Attempt 4 byte fill
        if let Ok(pattern) = pattern.try_into() {
            cuda_check(cuMemsetD2D32Async(
                dst_ptr.raw(),
                pitch,
                u32::from_ne_bytes(pattern),
                width,
                height,
                stream,
            ))?;

            return Ok(());
        }
    }

    fill_fallback(
        context, stream, kernels, ndims, counts, dst_ptr, strides, value,
    )
}

fn fill_fallback_kernel(dtype: DataType, ndims: usize) -> (String, String, Vec<KernelParam>) {
    let name = format!("fill_{}_{}d", make_valid_ident(&dtype.ctype()), ndims);
    let source = format!(
        "__device__ void {name}(
                {NS}::Point<{ndims}> index,
                {NS}::Array<{ty}, {ndims}> dst,
                {ty} value
            ) {{
                dst.get(index) = value;
            }}",
        name = name,
        ty = dtype.ctype(),
        ndims = ndims,
        NS = CPP_NAMESPACE,
    );

    let params = vec![
        KernelParam::array("dst", dtype, ndims, false),
        KernelParam::value("value", dtype),
    ];

    (name, source, params)
}

unsafe fn fill_fallback(
    context: CudaContextHandle,
    stream: &CudaStream,
    kernels: &MemOpsKernelsCache,
    ndims: usize,
    counts: [i64; MAX_DIMS],
    dst_ptr: CudaDevicePtr,
    dst_strides: Strides,
    value: DataValue,
) -> prelude::Result {
    let dtype = value.data_type();
    let args = [
        KernelArg::array_dyn(ndims, dtype, dst_ptr, &ZEROS, &counts, &*dst_strides),
        KernelArg::value(value),
    ];

    let key = Key::Fill {
        dtype,
        ndims: ndims as u8,
    };

    kernels.compile_and_launch_elementwise(
        context,
        stream,
        ndims,
        counts,
        &args,
        key,
        move || fill_fallback_kernel(dtype, ndims),
    )?;

    Ok(())
}
