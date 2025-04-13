pub mod duck;

pub unsafe fn init() {
    pgrx::log!("INIT: engine starts init to create instance");
    duck::DuckDBEngine::instance();
}
