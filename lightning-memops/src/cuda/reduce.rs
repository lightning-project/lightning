use super::{Key, ZEROS};
use crate::{simplify_strides, MemOpsKernelsCache, Reduction};
use lightning_codegen::{make_valid_ident, KernelArg, KernelParam, CPP_NAMESPACE};
use lightning_core::accessor::{CudaAccessor, CudaMutAccessor};
use lightning_core::prelude::*;
use lightning_cuda::{ContextHandle as CudaContextHandle, Stream as CudaStream};

pub unsafe fn cuda_fold(
    context: CudaContextHandle,
    stream: &CudaStream,
    kernels: &MemOpsKernelsCache,
    src: CudaAccessor,
    dst: CudaMutAccessor,
    reduction: Reduction,
) -> Result {
    assert_eq!(reduction.data_type(), src.data_type());
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

    let args = [
        KernelArg::array_dyn(ndims, dtype, dst_ptr, &ZEROS, &counts, &*dst_strides),
        KernelArg::array_dyn(ndims, dtype, src_ptr, &ZEROS, &counts, &*src_strides),
    ];

    let key = Key::Fold {
        reduction,
        ndims: ndims as u8,
    };

    kernels.compile_and_launch_elementwise(
        context,
        stream,
        ndims,
        counts,
        &args,
        key,
        move || fold_kernel(reduction, ndims),
    )?;

    Ok(())
}

fn fold_kernel(reduction: Reduction, ndims: usize) -> (String, String, Vec<KernelParam>) {
    let dtype = reduction.data_type();
    let name = format!("fold_{}_{}d", make_valid_ident(&dtype.ctype()), ndims);

    let source = format!(
        "
            __device__ void {name}(
                {NS}::Point<{ndims}> index,
                {NS}::Array<{ty}, {ndims}> dst,
                const {NS}::Array<{ty}, {ndims}> src
            ) {{
                const {ty} lhs = src.get(index);
                const {ty} rhs = dst.get(index);
                dst.get(index) = {}(lhs, rhs);
            }}",
        reduction.csource_function_name(),
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
