use crate::cstr_ptr;
use crate::error::ErrorCode;

use duckdb::ffi::{
    duckdb_close, duckdb_column_count, duckdb_column_name, duckdb_config, duckdb_connect,
    duckdb_connection, duckdb_create_config, duckdb_database, duckdb_destroy_config,
    duckdb_destroy_prepare, duckdb_destroy_result, duckdb_disconnect, duckdb_execute_prepared,
    duckdb_free, duckdb_open_ext, duckdb_prepare, duckdb_prepared_statement, duckdb_result,
    duckdb_row_count, duckdb_set_config, duckdb_string, duckdb_value_string, DuckDBSuccess,
};
use duckdb::Result;

use pgrx::{error, log};
use std::ffi::{c_char, CStr, CString};
use std::os::raw::c_void;
use std::sync::OnceLock;

// static TPCH_DATABASE_PATH: &str = "/home/windy/postgres/duckdbDatasets/tpch1.db";

pub struct DuckDBEngine {
    database: duckdb_database,
    connection: duckdb_connection,
}

impl DuckDBEngine {
    pub unsafe fn new() -> Result<Self, ErrorCode> {
        // 初始化DuckDB配置
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
        let mut db: duckdb_database = std::ptr::null_mut();
        // 这里不给出数据库的路径，默认开启内存数据库实例
        let db_path: Option<String> = None; // 或 Some("path/to/db".into())
        let c_path = db_path.map(|p| CString::new(p).unwrap());
        let mut out_err: *mut c_char = std::ptr::null_mut();
        if duckdb_open_ext(
            c_path.map(|p| p.as_ptr()).unwrap_or(std::ptr::null()),
            &mut db,
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
        log!("DuckDB实例初始化完毕,实例为:{:?}", db);

        // 初始化对上述DuckDB实例的连接
        let mut conn: duckdb_connection = std::ptr::null_mut();
        if duckdb_connect(db, &mut conn) != DuckDBSuccess {
            return Err(ErrorCode::DuckDBFailedConnect);
        }
        log!("DuckDB连接初始化完毕,连接为:{:?}", conn);

        //配置一次性使用之后丢弃
        duckdb_destroy_config(&mut config);
        Ok(DuckDBEngine {
            database: db,
            connection: conn,
        })
    }

    /// 获取全局唯一的 DuckDBEngine 实例
    pub unsafe fn instance() -> &'static Self {
        static INSTANCE: OnceLock<Result<DuckDBEngine, ErrorCode>> = OnceLock::new();
        let instance = INSTANCE
            .get_or_init(|| unsafe { DuckDBEngine::new() })
            .as_ref();
        if instance.is_err() {
            error!("DuckDB实例初始化或获取失败");
        }
        instance.unwrap()
    }

    // 这里做一个基本SQL功能演示
    pub unsafe fn demo_sql(&self) {
        let sqls = vec![
            r"CREATE TABLE integers(i INTEGER, j INTEGER);",
            r"INSERT INTO integers VALUES (3, 4), (5, 6), (7, NULL);",
            r"SELECT * FROM integers",
        ];
        let mut res: duckdb_result = std::mem::zeroed();
        let mut prepared_stmt: duckdb_prepared_statement = std::ptr::null_mut();
        for i in 0..sqls.len() {
            let sql = sqls[i];
            if i < sqls.len() - 1 {
                if duckdb_prepare(self.connection, cstr_ptr!(sql), &mut prepared_stmt)
                    != DuckDBSuccess
                {
                    error!("无法准备当前DDL/DML语句的执行:{}", sql);
                } else if duckdb_execute_prepared(prepared_stmt, std::ptr::null_mut())
                    != DuckDBSuccess
                {
                    error!("无法执行已经准备好的DDL/DML语句:{}", sql);
                } else {
                    log!("成功执行{}:", sql);
                }
            } else {
                let mut prepared_stmt: duckdb_prepared_statement = std::ptr::null_mut();
                if duckdb_prepare(self.connection, cstr_ptr!(sql), &mut prepared_stmt)
                    != DuckDBSuccess
                {
                    error!("无法准备当前DQL语句的执行:{}", sql);
                } else if duckdb_execute_prepared(prepared_stmt, &mut res) != DuckDBSuccess {
                    error!("无法执行已经准备好的DQL语句:{}", sql);
                } else {
                    log!("成功执行,结果为{:?} ", res);
                }

                let row_count = duckdb_row_count(&mut res);
                let column_count = duckdb_column_count(&mut res);
                let mut rows: Vec<Vec<&str>> = Vec::with_capacity(row_count as usize + 1);
                rows.push(Vec::with_capacity(column_count as usize));
                let mut duck_strs: Vec<*mut c_char> = vec![];
                for idx in 0..column_count {
                    rows[0].push(
                        CStr::from_ptr(duckdb_column_name(&mut res, idx))
                            .to_str()
                            .unwrap(),
                    );
                }
                log!("列名是：{:?}", rows[0]);
                for row_idx in 0..row_count {
                    rows.push(Vec::with_capacity(column_count as usize));
                    for col_idx in 0..column_count {
                        let val = duckdb_value_string(&mut res, col_idx, row_idx);
                        if val.data.is_null() {
                            rows[row_idx as usize + 1].push("<NULL>");
                        } else {
                            rows[row_idx as usize + 1].push(
                                CStr::from_ptr(val.data)
                                    .to_str()
                                    .unwrap_or("<INVALID UTF-8>"),
                            );
                            duck_strs.push(val.data);
                        }
                    }
                }
                log!("{}:dql SQL's stmt is {:?},duckstrs is {:?},collumn count is {},row count is {}", sql, prepared_stmt,duck_strs,column_count,row_count);
                log!("DEMO查询结果:{:?}", rows);
                for ptr in duck_strs {
                    duckdb_free(ptr as *mut c_void);
                }
                duckdb_destroy_result(&mut res);
            }
        }
        duckdb_destroy_prepare(&mut prepared_stmt);
    }

    // 先丢弃连接再丢弃数据库，销毁单例的资源
    pub unsafe fn shutdown(&mut self) {
        duckdb_disconnect(&mut self.connection);
        duckdb_close(&mut self.database);
        log!("DuckDBEngine 已手动释放");
    }
}

// 由于单例模式，静态全局变量的资源退出程序一起回收，没有机会调用drop
// impl Drop for DuckDBEngine {
//     fn drop(&mut self) {}
// }
unsafe impl Sync for DuckDBEngine {}
unsafe impl Send for DuckDBEngine {}
