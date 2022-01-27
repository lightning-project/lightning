use lightning_codegen::KernelSpecializationPolicy;
use std::env;
use std::path::PathBuf;

use crate::prelude::*;

#[derive(Debug)]
pub struct Config {
    pub driver: DriverConfig,
    pub worker: WorkerConfig,
}

impl Config {
    pub fn new(driver: DriverConfig, worker: WorkerConfig) -> Self {
        Self { driver, worker }
    }

    pub fn from_env() -> Self {
        Self {
            driver: DriverConfig::from_env(),
            worker: WorkerConfig::from_env(),
        }
    }
}

#[derive(Debug)]
pub struct WorkerConfig {
    pub storage_dir: Option<PathBuf>,
    pub storage_capacity: u64,
    pub host_mem_max: usize,
    pub host_mem_block: usize,
    pub device_mem_max: Option<usize>,
    pub scheduling_lookahead_size: usize,
    pub specialization_policy: KernelSpecializationPolicy,
}

impl WorkerConfig {
    pub fn from_env() -> Self {
        let mut specialization_policy = default();

        if let Ok(level) = env::var("LIGHTNING_SPECIALIZATION") {
            use KernelSpecializationPolicy::*;

            specialization_policy = match level.trim() {
                "0" | "none" => None,
                "1" | "mild" => Mild,
                "2" | "standard" | "" => Standard,
                "3" | "aggressive" => Aggressive,
                "4" => VeryAggressive,
                s => {
                    warn!("unknown specialization level {:?}, reverting to level 2", s);
                    Standard
                }
            }
        }

        Self {
            storage_dir: None,
            storage_capacity: u64::MAX,
            host_mem_max: 40_000_000_000,
            host_mem_block: 1_000_000_000,
            scheduling_lookahead_size: 1_000_000_000,
            device_mem_max: None,
            specialization_policy,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DriverConfig {
    pub trace_file: Option<PathBuf>,
}

impl DriverConfig {
    pub fn from_env() -> Self {
        let mut out = Self { trace_file: None };

        if let Ok(filename) = env::var("LIGHTNING_TRACE") {
            let filename = filename.trim();

            if !filename.is_empty() {
                info!("writing trace to {:?}", filename);
                out.trace_file = Some(filename.into());
            }
        }

        out
    }
}
