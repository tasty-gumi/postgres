use pgrx::pg_sys::IsUnderPostmaster;
use pgrx::prelude::*;

pub mod engine;
mod error;
mod hooks;
mod macros;

::pgrx::pg_module_magic!();

#[pg_extern]
fn hello_pg_plansplit() -> &'static str {
    "Hello, pg_plansplit"
}

#[pg_guard]
unsafe extern "C" fn _PG_init() {
    use crate::error::*;

    // 当前的进程必须不是子进程，确保初始化的时候是主进程，确保该插件写在了shared_preload_libraries中
    if unsafe { IsUnderPostmaster } {
        bad_init();
    }
    unsafe {
        hooks::init();
        engine::init();
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_hello_pg_plansplit() {
        assert_eq!("Hello, pg_plansplit", crate::hello_pg_plansplit());
    }
}

/// This module is required by `cargo pgrx test` invocations.
/// It must be visible at the root of your extension crate.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {
        // perform one-off initialization when the pg_test framework starts
    }

    #[must_use]
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        // return any postgresql.conf settings that are required for your tests
        vec![]
    }
}
