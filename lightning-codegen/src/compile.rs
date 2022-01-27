use lightning_core::prelude::*;
use lightning_cuda::prelude::*;
use serde::{Deserialize, Serialize};
use std::ffi::{c_void, OsStr, OsString};
use std::io;
use std::io::{Read as _, Write as _};
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;
use std::process::Command;

const DEFAULT_COMMAND: &str = "nvcc";

#[derive(Error, Debug)]
pub enum CompilationError {
    #[error("{0}")]
    IO(#[from] io::Error),

    #[error("{0}")]
    Cuda(#[from] CudaError),

    #[error("compilation failed ({cmd}): {stderr}")]
    CompilationFailed {
        cmd: String,
        stdout: String,
        stderr: String,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct KernelCompiler {
    pub command: Option<OsString>,
    pub options: Vec<OsString>,
    pub working_dir: Option<PathBuf>,
    pub debugging: bool,
}

impl KernelCompiler {
    pub(crate) fn compile(
        &self,
        context: CudaContextHandle,
        source: &[u8],
    ) -> Result<CudaModule, CompilationError> {
        let command: &OsStr = match &self.command {
            Some(c) => c,
            None => DEFAULT_COMMAND.as_ref(),
        };

        let mut cmd = Command::new(command);
        cmd.args(&self.options);

        // Add architecture flags
        let device = context.device()?;
        let (major, minor) = device.compute_capability()?;
        cmd.arg(format!("--gpu-architecture=sm_{}{}", major, minor));

        // Verbose
        // debugging yes/no
        if self.debugging {
            warn!(
                "kernel will be compiled in debugging mode. DO NOT USE FOR PERFORMANCE BENCHMARKS"
            );
            cmd.args(&["--verbose", "--debug", "--device-debug"]);
            cmd.args(&["--define-macro", "DEBUG=1"]);
        } else {
            cmd.args(&["--define-macro", "NDEBUG=1"]);
        }

        // Prepare input file
        let mut input_file = tempfile::Builder::new()
            .prefix("lightning_")
            .suffix(".cu")
            .tempfile()?;

        input_file.write_all(&source)?;
        input_file.flush()?;
        cmd.arg(input_file.path());

        // Prepare output file
        let mut output_file = tempfile::Builder::new()
            .prefix("lightning_")
            .suffix(".cubin")
            .tempfile()?;

        if !any(&self.options, |p| p.as_bytes().starts_with(b"-std")) {
            cmd.arg("-std=c++14");
        }

        cmd.arg("--cubin");
        cmd.arg("--output-file");
        cmd.arg(output_file.path());

        // Set current working directory
        if let Some(path) = &self.working_dir {
            cmd.current_dir(path);
        }

        // Go!
        let result = cmd.output()?;

        let stderr = String::from_utf8_lossy(&result.stderr).into_owned();
        let stdout = String::from_utf8_lossy(&result.stdout).into_owned();

        // Check output
        if !result.status.success() {
            return Err(CompilationError::CompilationFailed {
                cmd: format!("{:?}", cmd),
                stderr,
                stdout,
            });
        }

        if !stderr.is_empty() {
            debug!("{:?}: {}", cmd, stderr);
        }

        // Read compiled image
        let mut image = vec![];
        output_file.read_to_end(&mut image)?;
        image.push(0);

        // Load compiled image
        let module = context
            .try_with(|| unsafe { CudaModule::load_image(image.as_ptr() as *const c_void) })?;

        Ok(module)
    }
}
