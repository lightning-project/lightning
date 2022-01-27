use super::instance::{LaunchParam, LaunchParamArray};
use super::types::KernelConfig;
use super::{KernelArrayDimension, KernelDef, KernelParam, CPP_NAMESPACE};
use crate::types::ModuleDef;
use lightning_core::prelude::*;
use lightning_core::{DataType, DataValue, PrimitiveType, MAX_DIMS};
use rand::distributions::Alphanumeric;
use rand::prelude::*;
use std::borrow::Cow;
use std::ffi::CString;
use std::fmt::Write as _;
use std::io::Write as _;

const HEADER_SOURCE: &[u8] = include_bytes!("../../resources/lightning.h");

#[derive(Error, Debug)]
pub enum GenerateError {
    #[error("invalid configuration given")]
    InvalidConfig,
    //#[error("invalid identifier: {0}")]
    //InvalidIdentifier(String),
}

pub(super) struct KernelSource {
    pub(super) source: Vec<u8>,
    pub(super) symbol: CString,
    pub(super) params: Vec<LaunchParam>,
}

fn generate_unique_ident(prefix: &str) -> String {
    let mut symbol = make_valid_ident(prefix).into_owned();
    symbol.push('_');

    // Add some random stuff to ensure unique name
    symbol.extend(
        thread_rng()
            .sample_iter(&Alphanumeric)
            .map(|c| c as char)
            .take(32),
    );

    symbol
}

pub(crate) fn is_valid_ident(ident: &str) -> bool {
    // Cannot be empty
    if ident.is_empty() {
        return false;
    }

    for (i, c) in enumerate(ident.chars()) {
        // First character must be alphabetic
        if i == 0 && !c.is_ascii_alphabetic() {
            return false;
        }

        // Characters must be alphanumeric or '_'
        if !c.is_ascii_alphanumeric() && c != '_' {
            return false;
        }
    }

    true
}

pub fn make_valid_ident(ident: &str) -> Cow<'_, str> {
    if is_valid_ident(ident) {
        return Cow::Borrowed(ident);
    }

    let mut output = String::with_capacity(ident.len());
    for (i, c) in enumerate(ident.chars()) {
        if i == 0 && !c.is_ascii_alphabetic() {
            output.push('_');
        }

        if c.is_ascii_alphanumeric() || c == '_' {
            output.push(c);
        } else {
            write!(output, "_{}_", c as u32).unwrap();
        }
    }

    if output.is_empty() {
        output.push_str("empty");
    }

    Cow::Owned(output)
}

#[allow(unused_must_use)]
fn write_value_param(
    index: usize,
    constraints: &KernelConfig,
    name: &str,
    dtype: DataType,
    signature: &mut String,
    body: &mut String,
) -> Result<Vec<LaunchParam>, GenerateError> {
    use PrimitiveType::*;

    if let Some(s) = constraints
        .arguments
        .iter()
        .find(|s| s.param_index == index)
    {
        let value = &s.value;

        fn convert<T>(value: &DataValue) -> Option<i128>
        where
            for<'a> &'a DataValue: TryInto<T>,
            T: Into<i128>,
        {
            let x: Option<T> = value.try_into().ok();
            x.map(|v| v.into())
        }

        let value_opt = match dtype.to_primitive() {
            Some(I8) => convert::<i8>(value),
            Some(I16) => convert::<i16>(value),
            Some(I32) => convert::<i32>(value),
            Some(I64) => convert::<i64>(value),
            Some(U8) => convert::<u8>(value),
            Some(U16) => convert::<u16>(value),
            Some(U32) => convert::<u32>(value),
            Some(U64) => convert::<u64>(value),
            _ => None,
        };

        let value_int = match value_opt {
            Some(v) => v,
            None => return Err(GenerateError::InvalidConfig),
        };

        writeln!(body, "  {} {} = {};", dtype.ctype(), name, value_int);
        Ok(vec![])
    } else {
        write!(signature, "{} {}", dtype.ctype(), name);
        Ok(vec![LaunchParam::Value { index, dtype }])
    }
}

#[allow(unused_must_use)] // ignore errors of write!
fn write_array_param(
    index: usize,
    config: &KernelConfig,
    name: &str,
    dtype: DataType,
    dims: &[KernelArrayDimension],
    is_constant: bool,
    bounds_checking: bool,
    signature: &mut String,
    body: &mut String,
) -> Result<Vec<LaunchParam>, GenerateError> {
    let ndims = dims.len();
    let ctype = dtype.ctype();
    let ctype = &*ctype;
    let mut lparams = vec![];

    let prefix = match is_constant {
        true => "const",
        false => "",
    };

    let free_ndims = dims
        .iter()
        .filter(|e| matches!(e, KernelArrayDimension::Free))
        .count();

    lparams.push(LaunchParam::Array {
        index,
        kind: LaunchParamArray::Pointer {
            constant: is_constant,
            dtype,
        },
    });

    write!(signature, "{} *const {}_ptr", ctype, name);
    write!(
        body,
        "  {} {NS}::Array<{}, {}, {}> {name} = {NS}::Array<{}, {}, {}>({name}_ptr",
        prefix,
        ctype,
        free_ndims,
        bounds_checking,
        ctype,
        ndims,
        bounds_checking,
        name = name,
        NS = CPP_NAMESPACE,
    );

    write!(body, ", {NS}::Strides<{}> {{", ndims, NS = CPP_NAMESPACE);

    for axis in 0..ndims {
        if axis != 0 {
            write!(body, ", ");
        }

        if let Some(s) = config
            .strides
            .iter()
            .find(|s| s.param_index == index && s.axis == axis)
        {
            write!(body, "{}", s.stride);
        } else {
            lparams.push(LaunchParam::Array {
                index,
                kind: LaunchParamArray::Stride(axis),
            });

            write!(signature, ", int64_t {}_strides_{}", name, axis);
            write!(body, "{}_strides_{}", name, axis);
        }
    }

    if bounds_checking {
        write!(body, "}}, {NS}::Point<{}> {{", ndims, NS = CPP_NAMESPACE);

        for axis in 0..ndims {
            if axis != 0 {
                write!(body, ", ");
            }

            lparams.push(LaunchParam::Array {
                index,
                kind: LaunchParamArray::LowerBound(axis),
            });
            write!(signature, ", int64_t {}_lbnd_{}", name, axis);
            write!(body, "{}_lbnd_{}", name, axis);
        }

        write!(body, "}}, {NS}::Point<{}> {{", ndims, NS = CPP_NAMESPACE);

        for axis in 0..ndims {
            if axis != 0 {
                write!(body, ", ");
            }

            lparams.push(LaunchParam::Array {
                index,
                kind: LaunchParamArray::UpperBound(axis),
            });
            write!(signature, ", int64_t {}_ubnd_{}", name, axis);
            write!(body, "{}_ubnd_{}", name, axis);
        }
    }

    write!(body, "}})");

    let mut axis = 0;
    for dim in dims {
        use KernelArrayDimension::*;

        let var = match dim {
            BlockX => "blockIdx.x",
            BlockY => "blockIdx.y",
            BlockZ => "blockIdx.z",
            Free => {
                axis += 1;
                continue;
            }
        };

        write!(body, ".collapse_axis<{}>((int64_t) {})", axis, var);
    }

    write!(body, ";\n");

    Ok(lparams)
}

#[allow(unused_must_use)] // ignore errors of write!
fn generate_wrapper(
    kernel: &KernelDef,
    constraints: &KernelConfig,
    wrapper_name: &str,
    output: &mut String,
) -> Result<Vec<LaunchParam>, GenerateError> {
    const XYZ: [char; 3] = ['x', 'y', 'z'];
    let real_name = kernel.function_name.to_string();
    //if !is_valid_ident(&real_name) {
    //    return Err(CompilationError::InvalidIdentifier(real_name));
    //}

    let bounds_checking = kernel.bounds_checking;

    let mut signature = String::new();
    let mut body = String::new();
    let mut arguments = String::new();

    write!(signature, "uint32_t block_offset_x, ");
    write!(signature, "uint32_t block_offset_y, ");
    write!(signature, "uint32_t block_offset_z");

    let mut launch_params = vec![
        LaunchParam::BlockOffset(0),
        LaunchParam::BlockOffset(1),
        LaunchParam::BlockOffset(2),
    ];

    for i in 0..MAX_DIMS {
        if let Some(n) = constraints.block_count[i] {
            writeln!(body, "  LIGHTNING_ASSUME(blockIdx.{} < {});", XYZ[i], n);
            writeln!(body, "  LIGHTNING_ASSUME(gridDim.{} == {});", XYZ[i], n);
        }
    }

    if let Some(block_size) = constraints.block_size {
        for i in 0..MAX_DIMS {
            writeln!(
                body,
                "  LIGHTNING_ASSUME(blockDim.{} == {});",
                XYZ[i], block_size[i]
            );
            writeln!(
                body,
                "  LIGHTNING_ASSUME(threadIdx.{} < {});",
                XYZ[i], block_size[i]
            );
        }
    } else {
        for i in 0..MAX_DIMS {
            writeln!(body, "  LIGHTNING_ASSUME(blockDim.{} > 0);", XYZ[i]);
        }
    }

    writeln!(
        body,
        "  dim3 virtual_block_index(block_offset_x + blockIdx.x, \
                                    block_offset_y + blockIdx.y, \
                                    block_offset_z + blockIdx.z);"
    );

    write!(arguments, "virtual_block_index");

    let mut dtypes = vec![];

    for (index, param) in enumerate(&kernel.parameters) {
        write!(signature, ", ");
        write!(arguments, ", ");

        match param {
            &KernelParam::Array {
                dtype,
                ref name,
                ref dims,
                is_constant,
            } => {
                let name = make_valid_ident(name);

                let lparams = write_array_param(
                    index,
                    constraints,
                    &name,
                    dtype,
                    dims,
                    is_constant,
                    bounds_checking,
                    &mut signature,
                    &mut body,
                )?;

                write!(arguments, "{}", name);
                dtypes.push(dtype);
                launch_params.extend(lparams);
            }
            &KernelParam::Value { dtype, ref name } => {
                let name = make_valid_ident(name);
                let lparams =
                    write_value_param(index, constraints, &name, dtype, &mut signature, &mut body)?;

                write!(arguments, "{}", name);
                dtypes.push(dtype);
                launch_params.extend(lparams);
            }
        };
    }

    let dtypes = {
        let mut unique = vec![];
        for dtype in dtypes {
            if !unique.contains(&dtype) {
                unique.push(dtype);
            }
        }
        unique
    };

    for dtype in dtypes {
        write!(
            body,
            "  static_assert(sizeof({0}) == {1} && alignof({0}) <= {2}, \"invalid size/alignment \
            for {0} (size should be {1}, alignment should be at most {2})\");\n",
            dtype.ctype(),
            dtype.size_in_bytes(),
            dtype.alignment(),
        );
    }

    // Add __launch_bounds__ if block size is known
    let launch_bounds = match constraints.block_size {
        Some(b) => {
            format!("\n__launch_bounds__({})\n", b[0] * b[1] * b[2])
        }
        _ => String::new(),
    };

    write!(
        output,
        "\nextern \"C\" __global__ void {}{}({}) {{\n\
            {}\n\
            {}({});\n\
        }}\n",
        launch_bounds, wrapper_name, signature, body, real_name, arguments
    );

    Ok(launch_params)
}

#[allow(unused_must_use)] // ignore errors of write!
pub(super) fn generate_kernel_wrapper(
    definition: &ModuleDef,
    constraints: &KernelConfig,
) -> Result<KernelSource, GenerateError> {
    let mut source = vec![];

    source.extend("\n#line 1 \"lightning.h\"\n".as_bytes());
    source.extend(HEADER_SOURCE);

    let p = definition.file_name.as_deref().unwrap_or("<source>");
    write!(source, "\n#line 1 \"{}\"\n", p);
    source.extend(&definition.source);

    write!(
        source,
        "\n#line 1 \"<generated wrapper for {}>\"\n",
        definition.kernel.function_name,
    );

    let symbol = generate_unique_ident(&definition.kernel.function_name);

    let mut wrapper = String::new();
    let params = generate_wrapper(&definition.kernel, constraints, &symbol, &mut wrapper)?;
    source.extend(wrapper.bytes());

    Ok(KernelSource {
        symbol: CString::new(symbol).unwrap(),
        params,
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::types::KernelConfigStride;
    use lightning_core::{DTYPE_F32, DTYPE_I32};

    #[test]
    fn test_generate_wrapper() {
        let def = KernelDef {
            source: vec![],
            function_name: "bar".to_string(),
            parameters: vec![
                KernelParam::Value {
                    name: "a".to_string(),
                    dtype: DTYPE_F32,
                },
                KernelParam::Array {
                    name: "b".to_string(),
                    dtype: DTYPE_I32,
                    is_constant: false,
                    dims: vec![],
                },
                KernelParam::Array {
                    name: "c".to_string(),
                    dtype: DTYPE_I32,
                    is_constant: true,
                    dims: vec![KernelArrayDimension::Free, KernelArrayDimension::Free],
                },
                KernelParam::Array {
                    name: "d".to_string(),
                    dtype: DTYPE_I32,
                    is_constant: false,
                    dims: vec![KernelArrayDimension::BlockX, KernelArrayDimension::BlockY],
                },
            ],
            file_name: None,
            bounds_checking: false,
            compiler: Default::default(),
        };

        let config = KernelConfig {
            block_size: Some((16, 16, 4).into()),
            block_count: [None; 3],
            strides: vec![KernelConfigStride {
                param_index: 2,
                axis: 0,
                stride: 1,
            }],
            arguments: vec![],
        };

        let symbol = "example";
        let mut output = String::new();
        let params = generate_wrapper(&def, &config, &symbol, &mut output).unwrap();

        let expected = "
extern \"C\" __global__ void __launch_bounds__(1024) example(
    uint32_t block_offset_x,
    uint32_t block_offset_y,
    uint32_t block_offset_z,
    float a,
    int32_t *const b_ptr,
    int32_t *const c_ptr,
    int64_t c_strides_1,
    int32_t *const d_ptr,
    int64_t d_strides_0,
    int64_t d_strides_1
) {
    LIGHTNING_ASSUME(blockDim.x == 16);
    LIGHTNING_ASSUME(threadIdx.x < 16);
    LIGHTNING_ASSUME(blockDim.y == 16);
    LIGHTNING_ASSUME(threadIdx.y < 16);
    LIGHTNING_ASSUME(blockDim.z == 4);
    LIGHTNING_ASSUME(threadIdx.z < 4);

    dim3 virtual_block_index(
        block_offset_x + blockIdx.x,
        block_offset_y + blockIdx.y,
        block_offset_z + blockIdx.z
    );
   ::lightning::Array<int32_t, 0, false> b = ::lightning::Array<int32_t, 0, false>(
        b_ptr, ::lightning::Strides<0> {});
   const ::lightning::Array<int32_t, 2, false> c = ::lightning::Array<int32_t, 2, false>(
        c_ptr, ::lightning::Strides<2> {1, c_strides_1});
   ::lightning::Array<int32_t, 0, false> d = ::lightning::Array<int32_t, 2, false>(
        d_ptr, ::lightning::Strides<2> {d_strides_0, d_strides_1})
        .collapse_axis<0>((int64_t) blockIdx.x)
        .collapse_axis<0>((int64_t) blockIdx.y);

    static_assert(sizeof(float) == 4 && alignof(float) <= 4,
        \"invalid size/alignment for float (size should be 4, alignment should be at most 4)\");
    static_assert(sizeof(int32_t) == 4 && alignof(int32_t) <= 4,
        \"invalid size/alignment for int32_t (size should be 4, alignment should be at most 4)\");

    bar(virtual_block_index, a, b, c, d);
}";
        let output_stripped: String = output.chars().filter(|c| !c.is_whitespace()).collect();
        let expected_stripped: String = expected.chars().filter(|c| !c.is_whitespace()).collect();

        if output_stripped != expected_stripped {
            panic!(
                "expected != output\n expected={}\n output={}",
                expected, output
            );
        }
    }
}
