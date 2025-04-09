use crate::cstr_ptr;
use crate::error::ErrorCode;
use duckdb::ffi::{
    duckdb_close, duckdb_config, duckdb_create_config, duckdb_database, duckdb_destroy_config,
    duckdb_destroy_result, duckdb_open_ext, duckdb_result, duckdb_set_config, duckdb_state,
    DuckDBSuccess,
};
use duckdb::{Config, Connection, Result, Statement};
use pgrx::log;
use std::ffi::{c_char, CString};
use std::sync::{Mutex, OnceLock};

// static TPCH_DATABASE_PATH: &str = "/home/windy/postgres/duckdbDatasets/tpch1.db";

pub struct DuckDBEngine {
    database: duckdb_database,
}

impl DuckDBEngine {
    pub unsafe fn new() -> Result<Self, ErrorCode> {
        // 初始化配置
        let mut config: duckdb_config = std::ptr::null_mut();
        //这里的CSting::new()出来的对象的裸指针的生命周期仅仅存在于表达式结束,在进入if的花括号之前，对应的资源就被回收了
        //所以这里需要保证duckdb_create_config不会存储这个指针之后使用，事实上也确实不会
        if duckdb_create_config(&mut config) != DuckDBSuccess
            || duckdb_set_config(
                config,
                cstr_ptr!("custom_user_agent"),
                cstr_ptr!("pg_plansplit"),
            ) != DuckDBSuccess
            || duckdb_set_config(config, cstr_ptr!("max_memory"), cstr_ptr!("1GB")) != DuckDBSuccess
        {
            return Err(ErrorCode::DuckDBFailedConfig);
        }
        log!("DuckDB配置初始化完毕,配置为:{:?}", config);

        // 初始化DuckDB实例
        let mut out_database: duckdb_database = std::ptr::null_mut();
        let path: Option<String> = None; // 或 Some("path/to/db".into())
        let c_path = path.map(|p| CString::new(p).unwrap());
        let mut out_err: *mut c_char = std::ptr::null_mut();
        if duckdb_open_ext(
            c_path.map(|p| p.as_ptr()).unwrap_or(std::ptr::null()),
            &mut out_database,
            config,
            &mut out_err,
        ) != DuckDBSuccess
        {
            if !out_err.is_null() {
                let err_meg = CString::from_raw(out_err);
                log!("DuckDB实例打开失败:{:?}", err_meg);
            }
            return Err(ErrorCode::DuckDBFailedOpen);
        }
        log!("DuckDB实例初始化完毕,实例为:{:?}", out_database);

        duckdb_destroy_config(&mut config);
        Ok(DuckDBEngine {
            database: out_database,
        })
    }

    /// 获取全局唯一的 DuckDBEngine 实例
    pub unsafe fn instance() -> Result<&'static Self, ErrorCode> {
        static INSTANCE: OnceLock<Result<DuckDBEngine, ErrorCode>> = OnceLock::new();
        INSTANCE
            .get_or_init(|| unsafe { DuckDBEngine::new() })
            .as_ref()
            .map_err(|e| *e)
    }

    // pub unsafe fn demo_sql() -> Result<>{

    // }
}

impl Drop for DuckDBEngine {
    fn drop(&mut self) {
        unsafe {
            duckdb_close(&mut self.database);
        }
        log!("DuckDBEngine 已释放");
    }
}
unsafe impl Sync for DuckDBEngine {}
unsafe impl Send for DuckDBEngine {}
