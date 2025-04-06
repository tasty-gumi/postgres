use once_cell::sync::Lazy;
use pgrx::list::{List, ListCell};
use pgrx::log;
use pgrx::pg_sys::{
    planner_hook, planner_hook_type, standard_planner, CommonTableExpr, IsUnderPostmaster,
    ParamListInfo, ParamListInfoData, PlannedStmt, Query,
};
use pgrx::prelude::*;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::slice::from_raw_parts;

mod egmanager;
mod error;

::pgrx::pg_module_magic!();

#[pg_extern]
fn hello_pg_plansplit() -> &'static str {
    "Hello, pg_plansplit"
}

// 外部函数接口FFI,会查找相同函数签名的C函数
unsafe extern "C" {
    unsafe fn pg_get_querydef(query: *mut Query, pretty_flags: c_char) -> *mut c_char;
}

static ORIGINAL_PLANNER_HOOK: Lazy<planner_hook_type> = Lazy::new(|| Some(plansplit_planner));

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
    }
    standard_planner(parse, query_string, cursor_options, bound_params)
}

#[pg_guard]
unsafe extern "C" fn _PG_init() {
    // 保存原始钩子
    use crate::error::*;

    // 当前的进程必须不是子进程，确保初始化的时候是主进程
    if unsafe { IsUnderPostmaster } {
        bad_init();
    }

    // 注册自定义钩子
    unsafe {
        match ORIGINAL_PLANNER_HOOK.as_ref() {
            Some(hook) => planner_hook = Some(*hook),
            None => planner_hook = None,
        }
    }
}
// 将Query转换成原始查询SQL的接口
pub unsafe fn get_querydef(query: *mut pg_sys::Query, pretty: bool) -> String {
    // 转换 pretty 标志为 PostgreSQL 内部格式
    let pretty_flags = if pretty { 1i8 } else { 0i8 };

    let cstr_ptr = pg_get_querydef(query, pretty_flags);

    let cstr = CStr::from_ptr(cstr_ptr);
    cstr.to_string_lossy().into_owned()
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
