use lightning_codegen::{
    make_valid_ident, Kernel, KernelArg, KernelDef, KernelParam, KernelSpecializationPolicy,
    ModuleDef, CPP_NAMESPACE,
};
use lightning_core::prelude::*;
use lightning_core::util::div_ceil;
use lightning_core::{Dim3, DTYPE_U64, MAX_DIMS};
use lightning_cuda::prelude::*;
use std::fmt::Write as _;

pub(crate) fn generate_elementwise_kernel(
    ndims: usize,
    function: &str,
    source: String,
    params: &[KernelParam],
    policy: KernelSpecializationPolicy,
) -> Kernel {
    Kernel::new(generate_wrapper(ndims, function, source, &params), policy)
}

pub(crate) unsafe fn launch_elementwise_async(
    kernel: &mut Kernel,
    ndims: usize,
    handle: CudaContextHandle,
    stream: &CudaStream,
    counts: [i64; MAX_DIMS],
    inner_args: &[KernelArg],
) -> Result {
    for i in 0..MAX_DIMS {
        if i >= ndims {
            assert_eq!(counts[i], 1);
        } else if counts[i] <= 0 {
            return Ok(());
        }
    }

    let mut args = vec![];
    for i in 0..ndims {
        args.push(KernelArg::value(counts[i] as u64));
    }
    args.extend_from_slice(inner_args);

    let block_size = match ndims {
        0 => Dim3::new(1, 1, 1),
        1 => Dim3::new(512, 1, 1),
        2 => {
            let ysize = (counts[1].min(16) as u64).next_power_of_two();
            let xsize = 512 / ysize;
            Dim3::new(xsize, ysize, 1)
        }
        3 => Dim3::new(8, 8, 8),
        _ => unreachable!(),
    };

    let grid_size = Dim3::new(
        div_ceil(counts[0], block_size[0] as i64).try_into()?,
        div_ceil(counts[1], block_size[1] as i64).try_into()?,
        div_ceil(counts[2], block_size[2] as i64).try_into()?,
    );

    kernel.launch_async(
        handle,
        stream,
        grid_size,
        block_size,
        Dim3::new(0, 0, 0),
        0,
        &args,
    )?;

    Ok(())
}

#[allow(unused_must_use)]
fn generate_wrapper(
    ndims: usize,
    function: &str,
    mut source: String,
    inner_params: &[KernelParam],
) -> ModuleDef {
    let wrapper_name = format!("{}_wrapper", function);

    let mut header = String::new();
    write!(header, "dim3 blockIdx");

    let mut body = String::new();
    write!(body, "{NS}::Point<{}> index;\n", ndims, NS = CPP_NAMESPACE);

    let mut parameters = vec![];
    for i in 0..ndims {
        write!(header, ", uint64_t {}", ["nx", "ny", "nz"][i]);

        parameters.push(KernelParam::Value {
            name: ["nx", "ny", "nz"][i].to_string(),
            dtype: DTYPE_U64,
        });

        let v = ["x", "y", "z"][i];
        write!(
            body,
            "uint64_t {v} = (uint64_t) threadIdx.{v} + \
                        (uint64_t) blockIdx.{v} * (uint64_t) blockDim.{v};\n",
            v = v
        );

        write!(body, "if ({v} >= n{v}) return;\n", v = v);
        write!(body, "index[{i}] = {v};\n", i = i, v = v);
    }

    write!(body, "{}(index", function);

    for param in inner_params {
        parameters.push(param.clone());

        match param {
            &KernelParam::Array {
                ref name,
                ref dims,
                dtype,
                is_constant,
            } => {
                let name = make_valid_ident(name);
                let prefix = match is_constant {
                    true => "const ",
                    false => "",
                };

                write!(
                    header,
                    ", {} {NS}::Array<{}, {}> {}",
                    prefix,
                    dtype.ctype(),
                    dims.len(),
                    name,
                    NS = CPP_NAMESPACE,
                );
                write!(body, ", {}", param.name());
            }
            &KernelParam::Value {
                dtype, ref name, ..
            } => {
                let name = make_valid_ident(name);
                write!(header, ", {} {}", dtype.ctype(), name);
                write!(body, ", {}", param.name());
            }
        }
    }

    write!(body, ");\n");

    write!(
        source,
        "\n\n __device__ void {}({}) {{
        {}
    }}",
        wrapper_name, header, body
    );

    ModuleDef {
        source: source.into_bytes(),
        file_name: None,
        kernel: KernelDef {
            bounds_checking: false,
            function_name: wrapper_name,
            parameters,
        },
        compiler: default(),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use lightning_core::{DTYPE_F32, DTYPE_I8};

    #[test]
    fn test_wrap_kernel() {
        let params = vec![
            KernelParam::array("foo", DTYPE_I8, 1, false),
            KernelParam::array("bar", DTYPE_F32, 2, true),
            KernelParam::value("baz", DTYPE_U64),
        ];
        let inner = "";

        let kernel = generate_wrapper(1, "kernel", inner.to_string(), &params);
        let source = String::from_utf8(kernel.source).unwrap();

        let expected = "
        __device__ void kernel_wrapper(
            dim3 blockIdx,
            uint64_t nx,
            ::lightning::Array<int8_t, 1> foo,
            const ::lightning::Array<float, 2> bar,
            uint64_t baz
        ) {
                ::lightning::Point<1> index;
                uint64_t x = (uint64_t) threadIdx.x + (uint64_t) blockIdx.x * (uint64_t) blockDim.x;
                if (x >= nx) return;
                index[0] = x;
                kernel(index, foo, bar, baz);
            }
        ";

        let left = source.chars().filter(|&c| !c.is_whitespace());
        let right = expected.chars().filter(|&c| !c.is_whitespace());
        let equals = all(zip(left, right), |(x, y)| x == y);
        assert!(
            equals,
            "mismatch. \nGot: {}\nExpected: {}",
            source, expected
        );

        let kernel = generate_wrapper(3, "kernel", inner.to_string(), &params);
        let source = String::from_utf8(kernel.source).unwrap();

        let expected = "
        __device__ void kernel_wrapper(
            dim3 blockIdx,
            uint64_t nx,
            uint64_t ny,
            uint64_t nz,
            ::lightning::Array<int8_t, 1> foo,
            const ::lightning::Array<float, 2> bar,
            uint64_t baz
        ) {
                ::lightning::Point<3> index;

                uint64_t x = (uint64_t) threadIdx.x + (uint64_t) blockIdx.x * (uint64_t) blockDim.x;
                if (x >= nx) return;
                index[0] = x;

                uint64_t y = (uint64_t) threadIdx.y + (uint64_t) blockIdx.y * (uint64_t) blockDim.y;
                if (y >= ny) return;
                index[1] = y;

                uint64_t z = (uint64_t) threadIdx.z + (uint64_t) blockIdx.z * (uint64_t) blockDim.z;
                if (z >= nz) return;
                index[2] = z;

                kernel(index, foo, bar, baz);
            }
        ";

        let left = source.chars().filter(|&c| !c.is_whitespace());
        let right = expected.chars().filter(|&c| !c.is_whitespace());
        let equals = all(zip(left, right), |(x, y)| x == y);
        assert!(
            equals,
            "mismatch. \nGot: {}\nExpected: {}",
            source, expected
        );
    }
}
