pub mod duck;
use crate::error;

pub unsafe fn init() {
    if duck::DuckDBManager::instance().is_err() {
        error::bad_engine_init("duckdb");
    }
}
