#一个数据库计划生成和采集器，负责枚举并采样行列混合计划

from sqlglot import parse_one
from sqlglot.optimizer import optimize,RULES
from sqlglot.expressions import Select
import time 
import json
from pg_interactor import Schema,Postgres
import random

DATASET_PATH = f"/home/windy/postgres/install/dataset/"
QUERY_TOTAL_NUMBER = {"tpcds10":99,"tpch1":22}
DATASET_NAME = "tpcds10"
CREATE_TABLE_SQL= f"""
CREATE TABLE IF NOT EXISTS public.plan_explored (
    id SERIAL PRIMARY KEY,
    name VARCHAR(15) NOT NULL,
    sql TEXT DEFAULT NULL,
    strategy JSONB DEFAULT NULL,
    plan JSONB DEFAULT NULL,
    embedding VECTOR(128) DEFAULT NULL,
    latency_s NUMERIC(10,3) DEFAULT NULL
);
"""

def extract_cte_mapping(root_select: Select):
    cte_mapping = {}
    
    with_node = root_select.args.get("with")
    if not with_node:
        return cte_mapping
    
    for cte in with_node.expressions:        
        cte_mapping[cte.alias] = cte.this.sql(dialect="postgres", pretty=True)  

    return cte_mapping

def execute_sql_to_plan(sql:str):
    plan = pg.plan_latency(sql=sql,cache=False)
    return plan

def cte_to_intermediate_results(cte_name:str,sql:str,use_duckdb=False):
    use =  "USING duckdb" if use_duckdb else " "
    force = "on" if use_duckdb else "off"
    pg.set_settings("duckdb.force_execution",force)
    pg.execute(f"CREATE TEMP TABLE IF NOT EXISTS {cte_name} {use} AS {sql};COMMIT;",retry_limit=1,fetch=False)

def clean_temp_tables(cte_names:list):
    for t in cte_names:
        pg.execute(f"DROP TABLE IF EXISTS {t}",retry_limit=1,fetch=False)

def generate_data_from_sql(sql: str, number: int):
    # if number in range(1,22):
    #     return
    # 解析原始 SQL 得到 AST
    ast = parse_one(sql, dialect='postgres')
    # 进行 AST 优化（包括 CTE 拆分），得到优化后的 AST
    optimized = optimize(ast, schema=table_schemas, dialect='postgres', leave_tables_isolated=False, rules=RULES)
    
    # 提取 CTE 映射，映射为 {CTE名: SQL字符串}
    cte_maps = extract_cte_mapping(optimized)
    optimized_sql = optimized.sql(pretty=True).replace('"','').replace("'", "''")
    
    # 获取主查询
    main_query = optimized.copy()
    main_query.args["with"] = None
    main_query = main_query.sql(dialect='postgres', pretty=True)
    
    # 根据 CTE 数量，计算所有可能组合数量，并限定最多采样 20 个
    cte_names = list(cte_maps.keys())
    if len(cte_names) == 0:
        return
    all_possible = 2 ** len(cte_names)
    max_samples = min(5, all_possible)
    
    unique_strategies = set()
    sample_count = 0
    query_name = f"{DATASET_NAME}-q{number}"
    
    # 先查询数据库中，同名查询已存储的策略{}
    select_sql = f"SELECT strategy FROM public.plan_explored WHERE name = '{query_name}';"
    existing_rows = pg.execute(select_sql,retry_limit=1, fetch=True)

    while sample_count < max_samples:
        sample_count += 1
        # 随机产生布尔型策略，每个CTE随机选 True/False
        strategy = {name: random.choice([True, False]) for name in cte_names}
        # 将布尔策略转换为固定顺序的元组以便比较（True/False 的元组）
        key = tuple(strategy[name] for name in sorted(cte_names))
        if key in unique_strategies:
            continue
        unique_strategies.add(key)
        
        # 将布尔策略转换为所需的字符串映射：True -> "duckdb", False -> "postgres"
        strategy_mapping = {name: ("duckdb" if strategy[name] else "postgres") for name in cte_names}
        strategy_json = json.dumps(strategy_mapping).replace("'", "''")
        duplicate = False

        # 针对已有记录的情况，检查是否已经存在相同策略
        if existing_rows:
            for row in existing_rows:
                stored_strategy = row[0]
                # 对于CTE查询，判断策略是否相同
                if stored_strategy == strategy_mapping:
                    duplicate = True
                    print(f"查询 {query_name} 的策略 {strategy_mapping} 已存在，跳过本次枚举")
                    break
        if duplicate:
            continue
        
        try:
            start_time = time.time()
            # 执行主查询，采集计划
            for name,cte_sql in cte_maps.items():
                cte_to_intermediate_results(cte_name=name, sql=cte_sql, use_duckdb=(strategy_mapping[name]=="duckdb"))
            plan = execute_sql_to_plan(main_query)
            latency = time.time() - start_time
            print(f"Sample ({query_name}) on strategy {strategy_mapping}: {latency:.3f} s")
        except Exception as e:
            print(f"执行查询 {query_name} 时出现异常,记录并跳过本次采样出的计划: {e}")
            clean_temp_tables(cte_names)
            latency = 1000.000 #表示超时或者失败
            insert_sql = f"""
                INSERT INTO public.plan_explored (name, strategy, latency_s)
                VALUES ('{query_name}', '{strategy_json}'::jsonb, {round(latency, 3)});COMMIT;  
            """
            pg.execute(insert_sql, ( ),retry_limit=1, fetch=False)
            continue
            
        clean_temp_tables(cte_names)
        plan_json = json.dumps(plan[0][0][0]).replace("'", "''")
        

        insert_sql = f"""
            INSERT INTO public.plan_explored (name, sql, strategy, plan, latency_s)
            VALUES ('{query_name}', '{optimized_sql}', '{strategy_json}'::jsonb, '{plan_json}'::jsonb, {round(latency, 3)});COMMIT;
        """
        try:
            pg.execute(insert_sql, ( ), retry_limit=1,fetch=False)
        except Exception as e:
            print(f"插入查询 {query_name} 的条目时出错: {e}")
            continue

def generate_baseline_from_sql(sql: str, number: int, duckdb_only=False):
    query_name = f"{DATASET_NAME}-q{number}"
    pg.set_settings("duckdb.force_execution","on" if duckdb_only else "off")
    if duckdb_only:
        strategy = {"all_postgres": "NO" ,"all_duckdb": "YES"}
    else:
        strategy = {"all_postgres": "YES" ,"all_duckdb": "NO"}
    select_sql = f"SELECT count(*) FROM public.plan_explored WHERE name = '{query_name}' and strategy = '{json.dumps(strategy).replace("'", "''")}'::jsonb;"
    existing_rows = pg.execute(select_sql,retry_limit=1, fetch=True)
    if existing_rows and existing_rows[0][0] > 0:
        print(f"查询 {query_name} 的基线计划已存在，跳过本次枚举")
        return
    try:
        start_time = time.time()
        plan = execute_sql_to_plan(sql)
        latency = time.time() - start_time
        print(f"Baseline Sample ({query_name}) on strategy {strategy}: {latency:.3f} s")
        plan_json = json.dumps(plan[0][0][0]).replace("'", "''")
        insert_sql = f"""
            INSERT INTO public.plan_explored (name, sql, strategy, plan, latency_s)
            VALUES ('{query_name}', '{sql.replace("'", "''")}', '{json.dumps(strategy).replace("'", "''")}'::jsonb, '{plan_json}'::jsonb, {round(latency, 3)});COMMIT;  
        """
        pg.execute(insert_sql, ( ),retry_limit=1, fetch=False)
    except Exception as e:
        print(f"执行查询 {query_name} 时出现异常,记录并跳过本次采样出的计划: {e}")
        latency = 60.000 #表示超时或者失败
        insert_sql = f"""
            INSERT INTO public.plan_explored (name, sql,strategy, latency_s)
            VALUES ('{query_name}', '{sql.replace("'", "''")}', '{json.dumps(strategy).replace("'", "''")}'::jsonb, {round(latency, 3)});COMMIT;  
        """
        pg.execute(insert_sql, ( ),retry_limit=1, fetch=False)
        return

if __name__ == "__main__":

    pg = Postgres()
    db_config={
        "dbname":"tpcds10",
        "host":"localhost",
        "user":"windy",
        "password":"",
        "port":5432
    }

    pg.setup(
        dbname=db_config["dbname"],
        host=db_config["host"],
        user=db_config["user"],
        password=db_config["password"],
        port=db_config["port"]
    )

    pg.set_settings("search_path",f"{DATASET_NAME}")
    pg.set_settings("statement_timeout","1min")
    pg.set_settings("duckdb.max_workers_per_postgres_scan","8")
    pg.set_settings("duckdb.threads_for_postgres_scan","8")
    pg.set_settings("duckdb.convert_unsupported_numeric_to_double","1")
    pg.execute(CREATE_TABLE_SQL,retry_limit=1,fetch=False)

    schema = Schema(pg,"tpcds10")
    table_schemas = {}
    for data_table in schema.tables:
        table_name = data_table.name 
        table_schemas[table_name] = data_table.column_types
    # print("Schema:", table_schemas)

    qnumber = QUERY_TOTAL_NUMBER.get(DATASET_NAME)
    for i in range(1,qnumber+1):
        with open(f"{DATASET_PATH}/{DATASET_NAME}/queries/q{str(i).zfill(2)}.sql","r") as f:
            sql = f.read()
            generate_baseline_from_sql(sql=sql,number=i,duckdb_only=True)
            generate_data_from_sql(sql=sql,number=i)

