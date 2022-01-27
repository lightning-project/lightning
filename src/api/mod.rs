//! User-facing functions and types of lightning.
mod array;
mod context;
mod core;
mod domain;
mod event;
mod kernel;

use std::panic::{catch_unwind, AssertUnwindSafe, UnwindSafe};
use std::sync::Arc;

use crate::driver::launch_driver_thread;
use crate::network::execute_endpoints;
use crate::planner::task::register_tasklets;
use crate::planner::Planner;
use crate::prelude::*;
use crate::types::Config;
use crate::worker::execute_worker;

pub use self::array::*;
pub use self::context::*;
pub use self::core::*;
pub use self::domain::*;
pub use self::event::*;
pub use self::kernel::*;
pub use crate::planner::distribution;

pub fn execute<F>(config: Config, fun: F) -> Result
where
    F: FnOnce(Context) -> Result + UnwindSafe,
{
    execute_erased(config, Box::new(fun))
}

fn execute_erased(
    config: Config,
    fun: Box<dyn FnOnce(Context) -> Result + UnwindSafe + '_>,
) -> Result {
    let driver_config = config.driver;
    let worker_config = config.worker;

    register_tasklets(); // Register tasklets

    execute_endpoints(
        |sender, receiver| {
            let driver = launch_driver_thread(driver_config, sender, receiver).unwrap();
            let planner = Arc::new(Mutex::new(Planner::new()));

            let context = Context {
                driver: driver.handle(),
                planner: Arc::downgrade(&planner),
            };

            let result = catch_unwind(AssertUnwindSafe(|| (fun)(context))); // Is it actually true?

            // Ignore errors. There is not much we can do about them now.
            let _ = planner.lock().shutdown(&driver.handle());
            let _ = driver.shutdown_and_wait();

            if let Err(e) = result.unwrap() {
                error!("error occured: {:?}", e);
            }
        },
        |sender, receiver, comm| {
            execute_worker(worker_config, comm, sender, receiver).unwrap();
        },
    )?;

    Ok(())
}
