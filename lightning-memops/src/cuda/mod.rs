use crate::{simplify_strides, Reduction};
use elementwise::{generate_elementwise_kernel, launch_elementwise_async};
use lightning_codegen::{
    make_valid_ident, Kernel, KernelArg, KernelParam, ModuleDef, CPP_NAMESPACE,
};
use lightning_core::accessor::{CudaAccessor4, CudaMutAccessor3};
use lightning_core::prelude::*;
use lightning_core::util::div_ceil;
use lightning_core::{DataType, Dim3, DTYPE_U64, MAX_DIMS};
use lightning_cuda::prelude::*;

pub(crate) mod copy;
pub(crate) mod elementwise;
pub(crate) mod fill;
pub(crate) mod reduce;

const ZEROS: [i64; 16] = [0; 16];

#[derive(Hash, PartialEq, Eq, Debug)]
enum Key {
    Fill {
        dtype: DataType,
        ndims: u8,
    },
    Copy {
        dtype: DataType,
        ndims: u8,
    },
    Fold {
        reduction: Reduction,
        ndims: u8,
    },
    Reduce {
        reduction: Reduction,
        ndims: u8,
        block_size_x: u32,
        block_size_y: u32,
    },
}

#[derive(Default, Debug)]
pub struct MemOpsKernelsCache {
    kernels: Mutex<HashMap<Key, Kernel>>,
}

impl MemOpsKernelsCache {
    unsafe fn compile_and_launch_elementwise<F: FnOnce() -> (String, String, Vec<KernelParam>)>(
        &self,
        handle: CudaContextHandle,
        stream: &CudaStream,
        ndims: usize,
        counts: [i64; MAX_DIMS],
        args: &[KernelArg],
        key: Key,
        gen: F,
    ) -> Result {
        use std::collections::hash_map::Entry;
        let mut cache = self.kernels.lock(); // Maybe a RwLock?

        let kernel = match cache.entry(key) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let (function_name, source, params) = gen();
                e.insert(generate_elementwise_kernel(
                    ndims,
                    &function_name,
                    source,
                    &params,
                    default(),
                ))
            }
        };

        launch_elementwise_async(kernel, ndims, handle, stream, counts, args)?;
        Ok(())
    }

    unsafe fn compile_and_launch<F: FnOnce() -> (String, String, Vec<KernelParam>)>(
        &self,
        handle: CudaContextHandle,
        stream: &CudaStream,
        block_size: Dim3,
        block_count: Dim3,
        args: &[KernelArg],
        key: Key,
        gen: F,
    ) -> Result {
        use std::collections::hash_map::Entry;
        let mut cache = self.kernels.lock(); // Maybe a RwLock?

        let kernel = match cache.entry(key) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let (function_name, source, params) = gen();
                let definition = ModuleDef::new(function_name, source.into_bytes(), params);
                let kernel = Kernel::new(definition, default());

                e.insert(kernel)
            }
        };

        kernel.launch_async(
            handle,
            stream,
            block_count,
            block_size,
            Dim3::new(0, 0, 0),
            0,
            args,
        )?;

        Ok(())
    }
}

pub unsafe fn cuda_reduce(
    context: CudaContextHandle,
    stream: &CudaStream,
    kernels: &MemOpsKernelsCache,
    mut src: CudaAccessor4,
    dst: CudaMutAccessor3,
    axis: usize,
    reduction: Reduction,
) -> Result {
    assert!(axis <= MAX_DIMS);
    for i in reversed(0..axis) {
        src = src.swap_axes(i, i + 1);
    }

    let src_size = src.extents();
    let dst_size = dst.extents();

    assert_eq!(reduction.data_type(), src.data_type());
    assert_eq!(src.data_type(), dst.data_type());
    assert_eq!(&src_size[1..], &dst_size[..]);

    let dtype = src.data_type();
    let elem_size = dtype.size_in_bytes();
    let mut src_strides = src.strides();
    let mut dst_strides = dst.strides();
    let mut counts = src_size.to_i64().unwrap();

    let [_, src_strides_except_first @ ..] = &mut *src_strides;
    let [_, counts_except_first @ ..] = &mut counts;

    let (ndims, [dst_offset, src_offset]) = simplify_strides(
        [&mut dst_strides, src_strides_except_first],
        counts_except_first,
    );

    let dst_ptr = dst
        .as_ptr_mut()
        .offset_bytes(dst_offset as isize * elem_size as isize);
    let src_ptr = src
        .as_ptr()
        .offset_bytes(src_offset as isize * elem_size as isize);

    let num_items = counts[1] * counts[2] * counts[3];
    let (block_size_x, block_size_y) = if num_items > 512 || counts[0] < 32 {
        (16, 16)
    } else if num_items > 32 || counts[0] < 512 {
        (1, 256)
    } else {
        (1, 1024)
    };

    let gen = move || {
        let name = format!("reduce_{}_{}d", make_valid_ident(&dtype.ctype()), ndims);
        let source = format!(
            "\
            __device__ void {name}(
                    dim3 blockIdx,
                    {NS}::Array<{ty}, {ndims}> dst,
                    {NS}::Array<{ty}, {ndims} + 1> src,
                    uint64_t count_reduce,
                    uint64_t count_first
            ) {{
                __shared__ {ty} shared_results[{block_size_y}][{block_size_x}];
                {ty} result = {identity};
                int tx = {block_size_x} > 1 ? threadIdx.x : 0;
                int ty = {block_size_y} > 1 ? threadIdx.y : 0;
                uint64_t index_first = blockIdx.x * {block_size_x} + tx;

                {NS}::Point<{ndims} + 1> src_index;
                if ({ndims} >= 1) src_index[1] = index_first;
                if ({ndims} >= 2) src_index[2] = blockIdx.y;
                if ({ndims} >= 3) src_index[3] = blockIdx.z;

                {NS}::Point<{ndims}> dst_index;
                if ({ndims} >= 1) dst_index[0] = index_first;
                if ({ndims} >= 2) dst_index[1] = blockIdx.y;
                if ({ndims} >= 3) dst_index[2] = blockIdx.z;

                if (index_first < count_first) {{
                    for (uint64_t i = ty; i < count_reduce; i += {block_size_y}) {{
                        src_index[0] = i;
                        {ty} partial = src.get(src_index);
                        result = {reduction}(result, partial);
                    }}
                }}

                if ({block_size_y} > 1) {{
                    shared_results[ty][tx] = result;

                    #pragma unroll(10)
                    for (int offset = {block_size_y}/ 2; offset > 0; offset /= 2) {{
                        __syncthreads();

                        if (ty < offset) {{
                            {ty} lhs = shared_results[ty][tx];
                            {ty} rhs = shared_results[ty + offset][tx];
                            result = {reduction}(lhs, rhs);
                            shared_results[ty][tx] = result;
                        }}
                    }}
                }}

                if (ty == 0 && index_first < count_first) {{
                    {ty} lhs = dst.get(dst_index);
                    {ty} rhs = result;
                    dst.get(dst_index) = {reduction}(lhs, rhs);
                }}
            }}
        ",
            reduction = reduction.csource_function_name(),
            name = name,
            block_size_x = block_size_x,
            block_size_y = block_size_y,
            ty = dtype.ctype(),
            ndims = ndims,
            identity = reduction.csource_identity_literal(),
            NS = CPP_NAMESPACE,
        );

        let parameters = vec![
            KernelParam::array("dst", dtype, ndims, false),
            KernelParam::array("src", dtype, ndims + 1, true),
            KernelParam::value("count_reduce", DTYPE_U64),
            KernelParam::value("count_first", DTYPE_U64),
        ];

        (name, source, parameters)
    };

    let args = [
        KernelArg::array_dyn(ndims, dtype, dst_ptr, &ZEROS, &counts[1..], &*dst_strides),
        KernelArg::array_dyn(ndims + 1, dtype, src_ptr, &ZEROS, &counts, &*src_strides),
        KernelArg::value(counts[0] as u64),
        KernelArg::value(counts[1] as u64),
    ];

    let key = Key::Reduce {
        reduction,
        ndims: ndims as u8,
        block_size_x,
        block_size_y,
    };

    kernels.compile_and_launch(
        context,
        stream,
        Dim3::new(block_size_x as u64, block_size_y as u64, 1),
        Dim3::new(
            div_ceil(counts[1] as u64, block_size_x as u64),
            counts[2] as u64,
            counts[3] as u64,
        ),
        &args,
        key,
        gen,
    )
}
