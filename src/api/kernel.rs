use lightning_codegen::{
    KernelArrayDimension, KernelDef as CodegenKernelDef, KernelParam, ModuleDef as CodegenModuleDef,
};
use smallvec::SmallVec;
use std::ffi::OsString;
use std::fmt::Display;
use std::fs::canonicalize;
use std::path::Path;

use crate::api::{Array, ArrayView, Context, Event};
use crate::planner::annotations::{
    parse_rules, AccessMode, AccessPattern, AffineAccessPattern, AnnotationRule,
};
use crate::planner::cuda::{CudaLauncher, CudaLauncherArg};
use crate::planner::distribution::IntoWorkDistribution;
use crate::prelude::*;
use crate::types::{
    CudaKernelId, DataType, DataValue, Dim, Dim3, ExecutorId, Point, Rect, DTYPE_I64, MAX_DIMS,
};

#[derive(Debug)]
struct ParamDef {
    name: String,
    dtype: DataType,
    is_array: bool,
}

/// Kernel definition.
pub struct CudaKernelBuilder {
    inner: CodegenModuleDef,
    constants: HashMap<String, i64>,
    num_arrays: usize,
    params: Vec<ParamDef>,
    rules: Vec<AnnotationRule>,
}

impl CudaKernelBuilder {
    pub fn new(source: impl Into<Vec<u8>>, function_name: impl Into<String>) -> Self {
        let inner = CodegenModuleDef {
            source: source.into(),
            file_name: None,
            compiler: default(),
            kernel: CodegenKernelDef {
                function_name: function_name.into(),
                parameters: vec![],
                bounds_checking: false,
            },
        };

        Self {
            inner,
            constants: default(),
            params: default(),
            rules: default(),
            num_arrays: 0,
        }
    }

    pub fn from_file(path: impl AsRef<Path>, function_name: impl Into<String>) -> Result<Self> {
        use std::{fs::File, io::Read};

        let path = path.as_ref();
        let path = canonicalize(path).with_context(|| format!("while opening {:?}", path))?;
        if path.extension().unwrap_or_default() != "cu" {
            warn!("expecting cu file, gotten {}", path.display());
        }

        let mut source = vec![];
        File::open(&path)
            .with_context(|| format!("while opening {:?}", path))?
            .read_to_end(&mut source)?;

        let mut result = Self::new(source, function_name);
        result.inner.compiler.working_dir = path.parent().map(Path::to_owned);
        result.inner.file_name = Some(path.to_string_lossy().into_owned());
        Ok(result)
    }

    pub fn define(&mut self, key: impl Display, value: impl Display) -> &mut Self {
        if let Ok(v) = value.to_string().parse() {
            self.add_constant(key.to_string(), v);
        }

        self.options(&["--define-macro", &format!("{}={}", key, value)])
    }

    pub fn add_constant(&mut self, key: impl Display, value: i64) -> &mut Self {
        let _ = self.constants.insert(key.to_string(), value);
        self
    }

    pub fn debugging(&mut self, flag: bool) -> &mut Self {
        self.inner.compiler.debugging = flag;
        self.inner.kernel.bounds_checking = flag;
        self
    }

    pub fn block_size(&mut self, _block_size: impl Into<Dim3>) -> Result<&mut Self> {
        //self.inner.block_size_hint = Some(block_size.into());
        Ok(self)
    }

    /// Add the given option which will be passed to the compiler verbatim.
    pub fn option<T>(&mut self, arg: T) -> &mut Self
    where
        T: Into<OsString>,
    {
        self.inner.compiler.options.push(arg.into());
        self
    }

    /// Add multiple options which will all be passed to the compiler verbatim.
    pub fn options<I>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator,
        I::Item: Into<OsString>,
    {
        for arg in args {
            self.option(arg);
        }

        self
    }

    /// Add options which will be passed to the host compiler/processor (via `-Xcompiler`)
    pub fn compiler_options<I>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator,
        I::Item: Into<OsString>,
    {
        for arg in args {
            self.option("--compiler-options");
            self.option(arg);
        }

        self
    }

    /// Add options which will be passed to the linker (via `-Xlinker`)
    pub fn linker_options<I>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator,
        I::Item: Into<OsString>,
    {
        for arg in args {
            self.option("--linker-options");
            self.option(arg);
        }

        self
    }

    pub fn param_value(&mut self, name: impl Into<String>, dtype: DataType) -> &mut Self {
        self.params.push(ParamDef {
            name: name.into(),
            dtype,
            is_array: false,
        });

        self
    }

    pub fn param_array(&mut self, name: impl Into<String>, dtype: DataType) -> &mut Self {
        self.params.push(ParamDef {
            name: name.into(),
            dtype,
            is_array: true,
        });

        self
    }

    pub fn annotate(&mut self, input: &str) -> Result<&mut Self> {
        for rule in parse_rules(input)? {
            self.annotate_rule(rule.name, rule.access_mode, rule.access_pattern);
        }

        Ok(self)
    }

    fn annotate_rule(
        &mut self,
        name: String,
        access_mode: AccessMode,
        access_pattern: AccessPattern,
    ) {
        self.rules.push(AnnotationRule {
            name,
            access_mode,
            access_pattern,
        })
    }

    pub fn compile(&mut self, context: &Context) -> Result<CudaKernel> {
        let mut raw_params: Vec<KernelParam> = vec![];
        let mut params = vec![];

        for param in &self.params {
            use AccessMode::*;
            if raw_params.iter().any(|e| &e.name() == &param.name) {
                bail!("param {:?} appears multiple times", param.name);
            }

            if !param.is_array {
                raw_params.push(KernelParam::Value {
                    name: param.name.clone(),
                    dtype: param.dtype,
                });

                params.push(Param::Value {
                    name: param.name.clone(),
                    dtype: param.dtype,
                });
                continue;
            }

            let index = match self.rules.iter().position(|e| &e.name == &param.name) {
                Some(i) => i,
                None => bail!("no annotation rule found for {:?}", param.name),
            };

            let rule = self.rules.swap_remove(index);
            let mode = rule.access_mode;
            let ndims = rule.access_pattern.len();
            let mut patterns = vec![rule.access_pattern];

            while let Some(i) = self.rules.iter().position(|e| &e.name == &param.name) {
                let rule = self.rules.swap_remove(i);

                // Only Read rules can appear multiple times.
                if mode != AccessMode::Read || rule.access_mode != mode {
                    bail!("rule for parameter {:?} appears multiple times", param.name);
                }

                if rule.access_pattern.len() != ndims {
                    bail!("dimensionality mismatch for parameter {:?}", param.name);
                }

                patterns.push(rule.access_pattern);
            }

            let is_constant = match mode {
                Read => true,
                Write | ReadWrite | Reduce(_) | Atomic(_) => false,
            };

            let mut dims = vec![KernelArrayDimension::Free; ndims];

            if matches!(mode, Reduce(_)) {
                dims.insert(0, KernelArrayDimension::BlockX);
                dims.insert(1, KernelArrayDimension::BlockY);
                dims.insert(2, KernelArrayDimension::BlockZ);
            }

            raw_params.push(KernelParam::Array {
                name: param.name.clone(),
                dtype: param.dtype,
                is_constant,
                dims,
            });

            params.push(Param::Array {
                name: param.name.clone(),
                dtype: param.dtype,
                access_mode: mode,
                access_patterns: patterns.into_boxed_slice(),
            });
        }

        if let Some(rule) = self.rules.pop() {
            bail!("rule found for unknown parameter {:?}", rule.name);
        }

        let mut definition = self.inner.clone();
        definition.kernel.parameters = raw_params;

        let name = definition.kernel.function_name.clone();
        let kernel_id = context.driver.compile_kernel(definition)?;

        Ok(CudaKernel {
            id: kernel_id,
            name,
            params,
            shared_memory: 0,
            context: Context::clone(context),
            constants: self.constants.clone(),
        })
    }
}

#[derive(Debug, Clone)]
enum Param {
    Value {
        name: String,
        dtype: DataType,
    },
    Array {
        name: String,
        dtype: DataType,
        access_mode: AccessMode,
        access_patterns: Box<[AccessPattern]>,
    },
}

#[derive(Debug)]
pub struct CudaKernel {
    pub(crate) id: CudaKernelId,
    pub(crate) name: String,
    pub(crate) shared_memory: u32,
    pub(crate) context: Context,
    pub(crate) constants: HashMap<String, i64>,
    params: Vec<Param>,
}

impl CudaKernel {
    pub fn id(&self) -> CudaKernelId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn constants(&self) -> &HashMap<String, i64> {
        &self.constants
    }

    pub fn constant(&self, key: &str) -> i64 {
        self.constants[key]
    }

    pub fn launch(
        &self,
        thread_domain: impl Into<Dim>,
        block_size: impl Into<Dim>,
        distribution: impl IntoWorkDistribution,
        args: impl KernelArgs,
    ) -> Result<Event> {
        let thread_domain = thread_domain.into();
        let distribution =
            distribution.into_work_distribution(self.context.system(), thread_domain)?;
        let superblocks = distribution.query_region(thread_domain.to_bounds());

        self._launch_distribution(
            thread_domain.to_bounds(),
            block_size.into(),
            &superblocks,
            &args,
        )
    }

    pub fn launch_one(
        &self,
        thread_domain: impl Into<Dim>,
        block_size: impl Into<Dim>,
        executor_id: ExecutorId,
        args: impl KernelArgs,
    ) -> Result<Event> {
        let thread_domain = thread_domain.into().to_bounds();

        self._launch_distribution(
            thread_domain,
            block_size.into(),
            &[(executor_id, thread_domain)],
            &args,
        )
    }

    pub fn launch_like<T>(
        &self,
        array: &Array<T>,
        block_size: impl Into<Dim>,
        args: impl KernelArgs,
    ) -> Result<Event> {
        self._launch_like(&array.inner, block_size.into(), &args)
    }

    fn _launch_like(
        &self,
        array: &ArrayView,
        block_size: Dim,
        args: &dyn KernelArgs,
    ) -> Result<Event> {
        let domain = array.domain;
        let dist = array
            .handle
            .distribution
            .as_work_distribution()
            .ok_or_else(|| anyhow!("array has no work distribution specified"))?;

        let transform = array
            .transform
            .to_regular()
            .ok_or_else(|| anyhow!("array transform is not regular"))?;

        let superblocks = dist
            .query_region(transform.apply_bounds(domain))
            .into_iter()
            .map(move |(e, d)| (e, transform.inverse_bounds(d, domain)))
            .collect_vec();

        self._launch_distribution(domain, block_size, &superblocks, args)
    }

    pub fn launch_distribution(
        &self,
        index_domain: impl Into<Rect>,
        block_size: impl Into<Dim>,
        superblocks: impl Into<Vec<(ExecutorId, Rect)>>,
        args: impl KernelArgs,
    ) -> Result<Event> {
        self._launch_distribution(
            index_domain.into(),
            block_size.into(),
            &superblocks.into(),
            &args,
        )
    }

    fn _launch_distribution(
        &self,
        index_domain: Rect,
        block_size: Dim,
        superblocks: &[(ExecutorId, Rect)],
        args: &dyn KernelArgs,
    ) -> Result<Event> {
        let launcher = self.build_launcher(block_size, args)?;

        self.context.submit(|stage| {
            for &(executor, superblock) in superblocks {
                let (lo, hi) = (superblock.low(), superblock.high());

                let mut is_divisible = true;
                for i in 0..MAX_DIMS {
                    if lo[i] % block_size[i] != 0 && lo[i] > index_domain.low()[i] {
                        is_divisible = false;
                    }

                    if hi[i] % block_size[i] != 0 && hi[i] < index_domain.high()[i] {
                        is_divisible = false;
                    }
                }

                if !is_divisible {
                    bail!(
                        "superblock {:?} is not divisible by thread block size {:?}",
                        superblock,
                        block_size
                    );
                }

                let block_offset = lo / block_size;
                let block_count = (hi - lo).to_dim().div_ceil(block_size);

                launcher.submit_launch(stage, executor, block_offset, block_count)?;
            }

            Ok(())
        })
    }

    fn build_launcher<A: KernelArgs>(&self, block_size: Dim, args: A) -> Result<CudaLauncher> {
        let mut builder = CudaLauncherBuilder::new(self, block_size);
        args.apply(&mut builder)?;
        builder.into_launcher()
    }

    fn launch_single<A: KernelArgs>(
        &self,
        executor: ExecutorId,
        block_size: Dim,
        block_offset: Point,
        block_count: Dim,
        args: A,
    ) -> Result<Event> {
        let launcher = self.build_launcher(block_size, args)?;

        self.context.submit(|stage| {
            launcher.submit_launch(stage, executor, block_offset, block_count)?;

            Ok(())
        })
    }
}

/// Tuple of arguments which can be passed to [`Kernel#launch`].
pub trait KernelArgs {
    #[doc(hidden)]
    fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result;
}

impl KernelArgs for &dyn KernelArgs {
    fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result {
        (&**self).apply(launch)
    }
}

// Generate impl KernelArgs for all tuples.
macro_rules! impl_kernel_args {
    ($($k:ident)*) => {
        impl_kernel_args!($($k)* | );
    };
    ($first:ident $($rest:ident)* | $($k:ident)*) => {
        impl_kernel_args!(@impl $($k)*);
        impl_kernel_args!($($rest)* | $($k)* $first);
    };
    ( | $($k:ident)*) => {
        impl_kernel_args!(@impl $($k)*);
    };
    (@impl $($k:ident)*) => {
        impl<$($k),*> KernelArgs for ($($k,)*)
        where
            $($k: KernelArg),*
        {
            #[allow(unused_variables, non_snake_case)]
            fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result {
                let ($($k,)*) = self;
                $(
                    $k.apply(launch)?;
                )*
                Ok(())
            }
        }
    }
}

impl_kernel_args!(A B C D E F G H I J K L M O P Q);

pub trait KernelArg {
    #[doc(hidden)]
    fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result;
}

impl<T: Into<DataValue> + Clone> KernelArg for T {
    fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result {
        launch.push_value(self.clone().into())
    }
}

impl<T> KernelArg for &Array<T> {
    fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result {
        launch.push_array(&self.inner)
    }
}

impl<T> KernelArg for Array<T> {
    fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result {
        launch.push_array(&self.inner)
    }
}

impl KernelArg for &ArrayView {
    fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result {
        launch.push_array(&**self)
    }
}

impl KernelArg for ArrayView {
    fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result {
        launch.push_array(self)
    }
}

pub struct CudaLauncherBuilder<'a> {
    kernel: &'a CudaKernel,
    block_size: Dim3,
    args: Vec<CudaLauncherArg>,
}

impl<'a> CudaLauncherBuilder<'a> {
    pub fn new(kernel: &'a CudaKernel, block_size: Dim3) -> Self {
        Self {
            kernel,
            block_size,
            args: Vec::with_capacity(kernel.params.len()),
        }
    }

    fn invalid_argument(&self, msg: &dyn Display) -> Result {
        let index = self.args.len();
        if let Some(param) = self.kernel.params.get(index) {
            let param_name = match param {
                Param::Value { name, .. } => name,
                Param::Array { name, .. } => name,
            };

            bail!(
                "argument {} (parameter \"{}\") of kernel {:?} is incorrect: {}",
                index + 1,
                param_name,
                self.kernel.name(),
                msg,
            )
        } else {
            bail!("error while calling {:?}: {}", self.kernel.name(), msg,)
        }
    }

    pub fn push_value(&mut self, mut value: DataValue) -> Result {
        let index = self.args.len();
        let dtype = match self.kernel.params.get(index) {
            Some(&Param::Value { dtype, .. }) => dtype,
            Some(_) => {
                return self.invalid_argument(&format_args!(
                    "expecting array, given value of type {}",
                    value.data_type(),
                ));
            }
            None => {
                return self.invalid_argument(&format_args!(
                    "expecting exactly {} arguments",
                    self.kernel.params.len()
                ));
            }
        };

        if value.data_type() != dtype {
            value = match value.cast(dtype) {
                Ok(v) => v,
                Err(_) => {
                    return self.invalid_argument(&format_args!(
                        "expecting value of type {}, given value of type {}",
                        dtype,
                        value.data_type(),
                    ));
                }
            };
        }

        self.args.push(CudaLauncherArg::Value { value });
        Ok(())
    }

    pub fn push_array(&mut self, array: &ArrayView) -> Result {
        let index = self.args.len();
        let (dtype, access_mode, patterns) = match self.kernel.params.get(index) {
            Some(Param::Array {
                dtype,
                access_mode,
                access_patterns: patterns,
                ..
            }) => (*dtype, *access_mode, patterns),
            Some(_) => {
                return self.invalid_argument(&format_args!(
                    "expecting value, given array of type {}",
                    array.data_type(),
                ));
            }
            None => {
                return self.invalid_argument(&format_args!(
                    "expecting exactly {} arguments",
                    self.kernel.params.len()
                ));
            }
        };

        if dtype != array.data_type() {
            return self.invalid_argument(&format_args!(
                "expecting array of type {}, given array of type {}",
                dtype,
                array.data_type(),
            ));
        }

        let ndims = patterns[0].len();
        let extents = array.domain.extents();

        if any(&extents[ndims..], |&v| v != 1) {
            return self.invalid_argument(&format_args!(
                "expecting {}-dimensional array, given array has shape {:?}",
                ndims, extents,
            ));
        }

        let mut access_patterns = SmallVec::with_capacity(1);
        for pattern in &**patterns {
            access_patterns.push(match self.rewrite_slice_exprs(&pattern) {
                Some(v) => v,
                _ => return self.invalid_argument(&"failed to evaluate expression"),
            })
        }

        self.args.push(CudaLauncherArg::Array {
            id: array.id(),
            domain: array.domain,
            transform: array.transform.clone(),
            access_mode,
            access_patterns,
        });

        Ok(())
    }

    fn rewrite_slice_exprs(&self, pattern: &AccessPattern) -> Option<AffineAccessPattern> {
        let vars = |key: &str| {
            if let Some(value) = self.kernel.constants.get(key) {
                return Some(*value);
            }

            for (param, arg) in zip(&self.kernel.params, &self.args) {
                if let (Param::Value { name, .. }, CudaLauncherArg::Value { value }) = (param, arg)
                {
                    if name == key {
                        if let Ok(v) = value.cast(DTYPE_I64) {
                            if let Ok(v) = v.try_into() {
                                return Some(v);
                            }
                        }
                    }
                }
            }

            warn!("undefined variable {:?} in expression", key);
            return None;
        };

        pattern.rewrite(self.block_size, &vars).ok()
    }

    fn into_launcher(self) -> Result<CudaLauncher> {
        if self.args.len() != self.kernel.params.len() {
            bail!("invalid number of arguments given");
        }

        Ok(CudaLauncher {
            kernel_id: self.kernel.id,
            block_size: self.block_size,
            args: self.args,
        })
    }
}
