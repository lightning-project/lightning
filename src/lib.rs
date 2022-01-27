#![allow(dead_code)]
#![allow(clippy::too_many_arguments, clippy::many_single_char_names)]

pub mod driver;
mod network;
#[macro_use]
mod prelude;
pub mod api;
mod planner;
pub mod types;
mod worker;

pub fn hostname() -> &'static str {
    lazy_static::lazy_static! {
        static ref HOSTNAME: String = {
            match ::hostname::get() {
                Ok(s) => s.to_string_lossy().into_owned(),
                Err(_) => "<anonymous>".into(),
            }
        };
    };

    &*HOSTNAME
}

pub fn initialize_logger() {
    use crate::prelude::hostname;
    use std::time::Instant;

    lazy_static::lazy_static! {
        static ref START_TIMING: Instant = Instant::now();
    }

    let _ = *START_TIMING;

    env_logger::Builder::from_default_env()
        .format(|formatter, record| {
            use std::io::Write;
            let duration = START_TIMING.elapsed();

            writeln!(
                formatter,
                //"[{} {} {:.03}] {}: {}",
                "[{} {} {:.15}] {}: {}",
                hostname(),
                record.module_path().unwrap_or("?"),
                duration.as_secs_f64(),
                record.level(),
                record.args(),
            )
        })
        .init();
}
