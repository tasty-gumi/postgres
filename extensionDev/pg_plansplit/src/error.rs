use pgrx::error;

pub fn bad_init() -> ! {
    error!("plansplit: failed to init plugin");
}

#[derive(Debug, Clone, Copy)]
pub enum ErrorCode {
    DuckDBFailedConfig,
    DuckDBFailedOpen,
}

pub fn bad_engine_init(name: &str) -> ! {
    error!("Engine: {} failed to init", name)
}
