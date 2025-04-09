use pgrx::log;
use pgrx::pg_sys::{
    object_access_hook_type, planner_hook, planner_hook_type, standard_planner, CommonTableExpr,
    ExecutorStart_hook_type, ParamListInfoData, PlannedStmt, ProcessUtility_hook_type, Query,
};
use pgrx::prelude::*;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::slice::from_raw_parts;

use crate::engine::duck;

static mut PREV_PLANNER_HOOK: planner_hook_type = None;
static mut PREV_EXECUTOR_START: ExecutorStart_hook_type = None;
static mut PREV_PROCESS_UTILITY: ProcessUtility_hook_type = None;
static mut NEXT_OBJECT_ACCESS_HOOK: object_access_hook_type = None;

pub unsafe fn init() {
    unsafe {
        PREV_PLANNER_HOOK = planner_hook;
        planner_hook = Some(plansplit_planner)
    }
}

// 外部函数接口FFI,会查找相同函数签名的C函数
unsafe extern "C" {
    unsafe fn pg_get_querydef(query: *mut Query, pretty_flags: c_char) -> *mut c_char;
}

// 将Query转换成原始查询SQL的接口
pub unsafe fn get_querydef(query: *mut pg_sys::Query, pretty: bool) -> String {
    // 转换 pretty 标志为 PostgreSQL 内部格式
    let pretty_flags = if pretty { 1i8 } else { 0i8 };

    let cstr_ptr = pg_get_querydef(query, pretty_flags);

    let cstr = CStr::from_ptr(cstr_ptr);
    cstr.to_string_lossy().into_owned()
}

// 这里的想法是通过planner hook拦截标准规划器的行为，先将ctelist提取出来，列表中的没一个公共表表达式都存在一个已经由XXXstmt转为Query的结构指针，通过这个Query可以再转换回SQL
#[pg_guard]
unsafe extern "C" fn plansplit_planner(
    parse: *mut Query,
    query_string: *const c_char,
    cursor_options: i32,
    bound_params: *mut ParamListInfoData,
) -> *mut PlannedStmt {
    // 查询带有cte表达式的时候，在规划器规划之前截获，cte列表并且还原出所有cte查询
    if !(*parse).cteList.is_null() {
        // postgres的解析器帮我确保cte长度，以及每个cell指针都不为空，每个listcell的都是指针字段有效，且是C语言可以转换的cte表达式指针
        let mut cte_queries: Vec<String> = vec![];
        let cte_len = (*(*parse).cteList).length as usize;
        let cte_listcells = from_raw_parts((*(*parse).cteList).elements, cte_len);
        for cell in cte_listcells {
            let cte = cell.ptr_value as *mut CommonTableExpr;
            cte_queries.push(get_querydef((*cte).ctequery as *mut Query, false));
        }
        log!("{:?}", cte_queries);
        // duck::DuckDBEngine::instance().unwrap()
        for cte_query in cte_queries {
            // let result = duck::DuckDBManager::instance()
            //     .expect("Failed to get duckdb instance")
            //     .duckdb_prepare_and_query(&cte_query)
            //     .expect("Failed to execute query");
            //                 .expect("Failed to create database connection");
            // let result: Vec<arrow::record_batch::RecordBatch> = conn
            //     .prepare(&cte_query)
            //     .expect("准备查询语句失败")
            //     .query_arrow([])
            //     .expect("执行查询失败")
            //     .collect();
            // log!("{:?}", result);
            // conn.close().expect("关闭连接失败");
        }
    }
    standard_planner(parse, query_string, cursor_options, bound_params)
}
