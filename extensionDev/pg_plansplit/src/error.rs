use pgrx::error;

pub fn bad_init() -> ! {
    error!("plansplit: failed to init plugin");
}
