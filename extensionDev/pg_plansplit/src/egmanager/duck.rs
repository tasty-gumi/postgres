use duckdb::arrow::array::RecordBatch;
use duckdb::{params, Config, Connection, Result, Statement};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

static TPCH_DATABASE_PATH: &str = "/home/windy/postgres/duckdbDatasets/tpch1.db";

pub struct DuckDBManager {
    connection: &'static Mutex<Connection>,
    database_path: String,
}

impl DuckDBManager {
    pub fn instance() -> Result<&'static Self> {
        static INSTANCE: OnceLock<DuckDBManager> = OnceLock::new();
        Ok(INSTANCE.get_or_init(|| {
            let config = Config::default()
                .custom_user_agent("pg_plansplit")
                .expect("Failed set user agent")
                .max_memory("2GB")
                .expect("Failed to set duckdb max memory")
                .enable_autoload_extension(true)
                .expect("Failed to set duckdb autoload extension")
                .access_mode(duckdb::AccessMode::ReadOnly)
                .expect("Failed to set duckdb access mode")
                .enable_object_cache(true)
                .expect("Failed to set duckdb object cache");
            let conn = Connection::open_with_flags(TPCH_DATABASE_PATH, config)
                .expect("Failed to create database connection");
            let connection_mutex = Box::leak(Box::new(Mutex::new(conn)));
            DuckDBManager {
                connection: connection_mutex,
                database_path: TPCH_DATABASE_PATH.to_string(),
            }
        }))
    }
    pub fn duckdb_prepare_and_query(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        let conn = self.connection.lock().expect("Failed to lock connection");
        let mut prepared_stmt: Statement = conn.prepare(sql).expect("Failed to prepare statement");
        let arrow: Vec<RecordBatch> = prepared_stmt
            .query_arrow([])
            .expect("Failed to execute the prepared stmt")
            .collect();
        Ok(arrow)
    }
}
