use super::{Key, ZEROS};
use crate::{simplify_strides, MemOpsKernelsCache};
use cuda_driver_sys::CUmemorytype_enum::CU_MEMORYTYPE_UNIFIED;
use cuda_driver_sys::{cuMemcpy2DAsync_v2, CUDA_MEMCPY2D};
use lightning_codegen::{make_valid_ident, KernelArg, KernelParam, CPP_NAMESPACE};
use lightning_core::accessor::{CudaAccessor, CudaMutAccessor, Strides};
use lightning_core::prelude::*;
use lightning_core::{prelude, DataType, MAX_DIMS};
use lightning_cuda::{
    cuda_check, ContextHandle as CudaContextHandle, DevicePtr as CudaDevicePtr,
    Stream as CudaStream,
};
use std::ptr::null_mut;

pub unsafe fn cuda_copy(
    context: CudaContextHandle,
    stream: &CudaStream,
    kernels: &MemOpsKernelsCache,
    src: CudaAccessor,
    dst: CudaMutAccessor,
) -> prelude::Result {
    assert_eq!(src.data_type(), dst.data_type());
    assert_eq!(src.extents(), dst.extents());

    let dtype = src.data_type();
    let elem_len = dtype.size_in_bytes();

    let mut src_strides = src.strides();
    let mut dst_strides = dst.strides();
    let mut counts = dst.extents().to_i64().unwrap();

    let (ndims, [dst_offset, src_offset]) =
        simplify_strides([&mut dst_strides, &mut src_strides], &mut counts);
    let src_ptr = src
        .as_ptr()
        .offset_bytes(src_offset as isize * elem_len as isize);
    let dst_ptr = dst
        .as_ptr_mut()
        .offset_bytes(dst_offset as isize * elem_len as isize);

    let first_contiguous = src_strides[0] == 1 as i64 && dst_strides[0] == 1 as i64;
    let positive_strides = all(0..MAX_DIMS, |i| src_strides[i] >= 0 && dst_strides[i] >= 0);

    if positive_strides && (ndims == 1 || (ndims == 2 && first_contiguous)) {
        let src_byte_strides = src_strides.to_byte_strides(dtype);
        let dst_byte_strides = dst_strides.to_byte_strides(dtype);

        let stream = stream.raw();
        let mut width = elem_len * counts[0] as usize;
        let mut height = counts[1] as usize;
        let mut src_pitch = src_byte_strides[1] as usize;
        let mut dst_pitch = dst_byte_strides[1] as usize;

        if !first_contiguous {
            assert_eq!(ndims, 1);
            width = elem_len;
            height = counts[0] as usize;
            src_pitch = src_byte_strides[0] as usize;
            dst_pitch = dst_byte_strides[0] as usize;
        }

        cuda_check(cuMemcpy2DAsync_v2(
            &CUDA_MEMCPY2D {
                // unused
                srcArray: null_mut(),
                dstArray: null_mut(),
                srcXInBytes: 0,
                srcY: 0,
                dstXInBytes: 0,
                dstY: 0,

                srcMemoryType: CU_MEMORYTYPE_UNIFIED,
                srcDevice: src_ptr.raw(),
                srcHost: null_mut(),
                srcPitch: src_pitch,

                dstMemoryType: CU_MEMORYTYPE_UNIFIED,
                dstDevice: dst_ptr.raw(),
                dstHost: null_mut(),
                dstPitch: dst_pitch,

                WidthInBytes: width,
                Height: height,
            },
            stream,
        ))?;
        return Ok(());
    }

    copy_fallback(
        context,
        stream,
        kernels,
        dtype,
        ndims,
        counts,
        dst_ptr,
        dst_strides,
        src_ptr,
        src_strides,
    )
}

fn copy_fallback_kernel(dtype: DataType, ndims: usize) -> (String, String, Vec<KernelParam>) {
    let name = format!("copy_{}_{}d", make_valid_ident(&dtype.ctype()), ndims);
    let source = format!(
        "__device__ void {name}(
                {NS}::Point<{ndims}> index,
                {NS}::Array<{ty}, {ndims}> dst,
                const {NS}::Array<{ty}, {ndims}> src
            ) {{
                dst.get(index) = src.get(index);
            }}",
        name = name,
        ty = dtype.ctype(),
        ndims = ndims,
        NS = CPP_NAMESPACE,
    );

    let params = vec![
        KernelParam::array("dst", dtype, ndims, false),
        KernelParam::array("src", dtype, ndims, true),
    ];

    (name, source, params)
}

unsafe fn copy_fallback(
    context: CudaContextHandle,
    stream: &CudaStream,
    kernels: &MemOpsKernelsCache,
    dtype: DataType,
    ndims: usize,
    counts: [i64; MAX_DIMS],
    dst_ptr: CudaDevicePtr,
    dst_strides: Strides,
    src_ptr: CudaDevicePtr,
    src_strides: Strides,
) -> prelude::Result {
    let args = [
        KernelArg::array_dyn(ndims, dtype, dst_ptr, &ZEROS, &counts, &*dst_strides),
        KernelArg::array_dyn(ndims, dtype, src_ptr, &ZEROS, &counts, &*src_strides),
    ];

    let key = Key::Copy {
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
        move || copy_fallback_kernel(dtype, ndims),
    )?;

    Ok(())
}
