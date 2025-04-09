///使用时只能接受&str类型，新建并返回给定字符串的C风格的*const c_char裸指针
#[macro_export]
macro_rules! cstr_ptr {
    ($s:expr) => {{
        // 类型检查（编译时）
        let s: &str = $s; // 如果不是 &str 会触发类型错误
        std::ffi::CString::new(s).unwrap().as_ptr()
    }};
}
