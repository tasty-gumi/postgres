use duckdb::{params, Config, Connection, Result};
use std::path::Path;

static TPCH_DATABASE_PATH: &str = "/home/windy/postgres/duckdbdatasets/tpch1.db";

pub struct DuckDBManager {
    connection: Connection,
}

impl DuckDBManager {
    pub fn init(&mut self) -> Result<()> {
        let mut config = Config::default();
        config = config.custom_user_agent("pg_plansplit").unwrap();

        self.connection = Connection::open_with_flags(TPCH_DATABASE_PATH, config).unwrap();
        Ok(())
    }
}
