# 一个数据库计划探索器，负责逻辑计划改写和物理计划搜索和枚举,并进行行列混合计划数据实际运行与采样收集

from typing import Tuple
from sqlglot import parse_one
from sqlglot.optimizer import optimize,RULES
from sqlglot.expressions import Select
import time 
import json
from pg_interactor import Schema,Postgres
import random
import argparse

DATASET_PATH = f"/home/windy/postgres/install/dataset/"
QUERY_TOTAL_NUMBER = {"tpcds10":99,"tpch1":22}
DATASET_NAME = "tpcds10"
DATASET_SIMPLE_NAME = "ds"
# 添加约束
# ALTER TABLE public.plan_explored 
# ADD CONSTRAINT uk_name_sql_strategy 
# UNIQUE (name, strategy);
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
    """
        提取CTE映射，映射为 {CTE名: SQL字符串}
    """
    cte_mapping = {}
    
    with_node = root_select.args.get("with")
    if not with_node:
        return cte_mapping
    
    for cte in with_node.expressions:        
        cte_mapping[cte.alias] = cte.this.sql(dialect="postgres", pretty=True)  

    return cte_mapping

def execute_sql_to_plan(sql:str,use_duckdb=False,exec = True):
    """
        执行SQL并返回其执行计划
        参数:
            sql: SQL字符串
            use_duckdb: 是否使用DuckDB执行SQL
            exec: 是否实际执行SQL以获取执行计划（否则仅生成计划）
        返回:
            SQL的执行计划（JSON结构）
    """
    pg.set_settings("duckdb.force_execution","on" if use_duckdb else "off")
    plan = pg.plan_latency(sql=sql,cache=False) if exec else pg.plan(sql=sql)
    return plan[0][0][0]

def cte_to_intermediate_results(cte_name:str,sql:str,use_duckdb=False):
    """
        将CTE查询作为临时表执行并物化，并返回创建其的执行计划
        参数:
            cte_name: CTE表名
            sql: CTE查询SQL字符串
            use_duckdb: 是否使用DuckDB执行CTE查询
        返回:
            CTE查询的执行计划（JSON结构）
    """
    use =  "USING duckdb" if use_duckdb else " "
    cte_plan = execute_sql_to_plan(f"{sql}",use_duckdb=use_duckdb,exec=False)
    pg.execute(f"CREATE TEMP TABLE IF NOT EXISTS {cte_name} {use} AS {sql};COMMIT;",retry_limit=1,fetch=False)
    return cte_plan

def clean_temp_tables(cte_names:list):
    """
        清理临时表
    """
    for t in cte_names:
        pg.execute(f"DROP TABLE IF EXISTS {t}",retry_limit=1,fetch=False)

def insert_cte_plans(main_plan: dict, cte_maps: dict):
    """
    将CTE计划插入到主查询计划(默认duckdb的计划为主计划)的对应节点中
    
    参数:
        main_plan: 主查询计划（JSON结构，根节点为字典，子节点在根节点的children中）
        cte_maps: CTE名称到CTE计划的字典（键为CTE表名，值为CTE的计划节点）
    返回:
        更新后的主查询计划
    """
    # 提取DuckDB执行计划的根节点（字典类型）
    root_node = main_plan['Plan']['DuckDB Execution Plan']
    
    def traverse(node: dict):
        """递归遍历单个节点（字典），处理当前节点并遍历其子节点"""
        # 1. 处理当前节点：检查是否为CTE表的扫描节点
        extra_info = node.get("extra_info", {})
        table_name = extra_info.get("Table")  # 获取当前节点关联的表名
        
        if table_name in cte_maps:
            # 确保当前节点有children字段（若为空则创建）
            if "children" not in node:
                node["children"] = []
            # 将CTE计划插入到当前节点的children中（与extra_info同级）
            node["children"].append(cte_maps[table_name])
        
        # 2. 递归处理当前节点的子节点（子节点在children列表中，每个子节点也是字典）
        children = node.get("children", [])  # 子节点列表，即使为空也安全处理
        for child in children:
            # 确保子节点是字典（避免非字典类型导致的错误）
            if isinstance(child, dict):
                traverse(child)
    
    # 从根节点开始遍历整个计划树
    traverse(root_node)
    
    # 根节点已被修改（子节点可能被更新），直接返回主计划
    return main_plan

def explore_candidate_plans_for_sql(sql: str, duckdb_probability: float = 0.5) -> Tuple[dict, float]:
    """
    探索SQL的不同执行计划，返回混合执行计划
    
    参数:
        sql: SQL字符串
        duckdb_probability: 使用DuckDB的概率
    
    返回:
        混合执行计划（JSON结构）和对应的执行时延
    """
    try:
        # 解析原始 SQL 得到 AST
        ast = parse_one(sql, dialect='postgres')
        
        # 进行 AST 优化（包括 CTE 拆分），得到优化后的 AST
        optimized = optimize(
            ast, 
            schema=table_schemas, 
            dialect='postgres', 
            leave_tables_isolated=False, 
            rules=RULES
        )
        
        # 提取 CTE 映射，映射为 {CTE名: SQL字符串}
        cte_maps = extract_cte_mapping(optimized)
        
        # 获取主查询
        main_query = optimized.copy()
        main_query.args["with"] = None
        main_query = main_query.sql(dialect='postgres', pretty=True)
        
        cte_plans = {}
        execution_stats = {
            'engines_used': {'duckdb': 0, 'postgres': 0},
            'cte_execution_times': {}
        }
        
        start_total = time.time()
        
        # 随机执行CTE：按概率选择执行引擎
        for name, cte_sql in cte_maps.items():
            cte_start = time.time()
            
            # 随机选择执行引擎
            use_duckdb = random.random() < duckdb_probability
            engine = 'duckdb' if use_duckdb else 'postgres'
            execution_stats['engines_used'][engine] += 1
            
            try:
                cte_plan = cte_to_intermediate_results(
                    cte_name=name, 
                    sql=cte_sql, 
                    use_duckdb=use_duckdb
                )
                cte_plans[name] = cte_plan
                
                # 记录执行时间
                cte_time = time.time() - cte_start
                execution_stats['cte_execution_times'][name] = {
                    'time': cte_time,
                    'engine': engine
                }
                
            except Exception as e:
                print(f"Warning: Failed to execute CTE {name} with {engine}: {str(e)}")
                # 失败时尝试使用另一种引擎
                fallback_engine = 'postgres' if use_duckdb else 'duckdb'
                try:
                    print(f"Trying fallback with {fallback_engine}")
                    cte_plan = cte_to_intermediate_results(
                        cte_name=name, 
                        sql=cte_sql, 
                        use_duckdb=(fallback_engine == 'duckdb')
                    )
                    cte_plans[name] = cte_plan
                    execution_stats['engines_used'][fallback_engine] += 1
                    
                    cte_time = time.time() - cte_start
                    execution_stats['cte_execution_times'][name] = {
                        'time': cte_time,
                        'engine': fallback_engine,
                        'fallback_used': True
                    }
                    
                except Exception as fallback_e:
                    print(f"Error: Both engines failed for CTE {name}: {str(fallback_e)}")
                    raise
        
        # 主计划执行（可选择随机化或固定使用DuckDB）
        main_start = time.time()
        try:
            # 主查询也可以随机选择引擎，这里保持使用DuckDB
            main_plan = execute_sql_to_plan(main_query, use_duckdb=True)
            main_time = time.time() - main_start
            execution_stats['main_execution_time'] = main_time
            execution_stats['main_engine'] = 'duckdb'
            
        except Exception as e:
            print(f"Error executing main query: {str(e)}")
            raise
        
        total_latency = time.time() - start_total
        
        # 构建混合执行计划
        plan = insert_cte_plans(main_plan, cte_plans)
        
        # 添加执行统计信息到计划中
        plan['execution_stats'] = execution_stats
        plan['total_latency'] = total_latency
        plan['engine_mix'] = {
            'duckdb_ratio': execution_stats['engines_used']['duckdb'] / len(cte_maps) if cte_maps else 0,
            'postgres_ratio': execution_stats['engines_used']['postgres'] / len(cte_maps) if cte_maps else 0
        }
        
        # 清理临时表
        clean_temp_tables(cte_maps.keys())
        
        return plan, total_latency
        
    except Exception as e:
        print(f"Error in explore_candidate_plans_for_sql: {str(e)}")
        # 确保在异常时也清理临时表
        if 'cte_maps' in locals():
            clean_temp_tables(cte_maps.keys())
        raise

def explore_plans_for_sql(sql: str, number: int = 0):
    # if number in range(1,22):
    #     return
    # 解析原始 SQL 得到 AST
    ast = parse_one(sql, dialect='postgres')
    # 进行 AST 优化（包括 CTE 拆分），得到优化后的 AST
    optimized = optimize(ast, schema=table_schemas, dialect='postgres', leave_tables_isolated=False, rules=RULES)
    
    # 提取 CTE 映射，映射为 {CTE名: SQL字符串}
    cte_maps = extract_cte_mapping(optimized)
    optimized_sql = optimized.sql(pretty=True).replace('"','').replace("'", "''")
    print(f"{DATASET_SIMPLE_NAME}-q{str(number).zfill(2)} 拥有的CTE数量: {len(cte_maps)}，计划空间约 {2**len(cte_maps)} 种")
    
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
    query_name = f"{DATASET_SIMPLE_NAME}-q{str(number).zfill(2)}"
    
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
        strategy_json_str = json.dumps(strategy_mapping).replace("'", "''")
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
            cte_plans = {}
            start_time = time.time()
            # 执行主查询，采集计划
            for name,cte_sql in cte_maps.items():
                cte_plan = cte_to_intermediate_results(cte_name=name, sql=cte_sql, use_duckdb=(strategy_mapping[name]=="duckdb"))
                # cte_map由名称到sql的映射转换为名称到计划json的映射
                cte_plans[name] = cte_plan
            # 主计划统一使用duckdb执行
            main_plan = execute_sql_to_plan(main_query,use_duckdb=True)
            latency = time.time() - start_time
            plan = insert_cte_plans(main_plan, cte_plans)
            print(f"Sample ({query_name}) on strategy {strategy_mapping}: {latency:.3f} s")
        except Exception as e:
            print(f"执行查询 {query_name} 的{strategy_mapping}时出现异常,记录并跳过本次采样出的计划: {e}")
            continue
            
        clean_temp_tables(cte_names)

        # 保存当前混合计划到数据库系统表中
        plan_json_str = json.dumps(plan).replace("'", "''")
        insert_sql = f"""
            INSERT INTO public.plan_explored (name, sql, strategy, plan, latency_s)
            VALUES ('{query_name}', '{optimized_sql}', '{strategy_json_str}'::jsonb, '{plan_json_str}'::jsonb, {round(latency, 3)});COMMIT;
        """
        try:
            pg.set_settings("duckdb.force_execution","off")
            pg.execute(insert_sql, ( ), retry_limit=1,fetch=False)
        except Exception as e:
            print(f"插入查询 {query_name} 的条目时出错: {e}")
            continue

def generate_baseline_from_sql(sql: str, number: int):
    query_name = f"{DATASET_SIMPLE_NAME}-q{str(number).zfill(2)}"
    strategy1 = {"all_postgres": "NO" ,"all_duckdb": "YES"}
    strategy2 = {"all_postgres": "YES" ,"all_duckdb": "NO"}
    # 初始物理执行计划为空
    plan_duckdb  = {}
    plan_pg = {}
    
    try:
        start_time = time.time()  # 覆盖start_time为实际开始时间
        
        # 执行plan1，成功后更新end1为实际时间
        plan_duckdb = execute_sql_to_plan(sql=sql, use_duckdb=True)
        lat_duckdb = round(time.time() - start_time,3)
        start_time = time.time() 
        
        # 执行plan2，成功后更新end2为实际时间
        plan_pg = execute_sql_to_plan(sql=sql, use_duckdb=False)
        lat_pg = round(time.time() - start_time,3)  # 若此处成功，end2变为真实值

        # 正常流程：计算耗时并插入
        print(f"Baseline Sample ({query_name}) on strategy {strategy1}: {lat_duckdb:.3f} s")
        print(f"Baseline Sample ({query_name}) on strategy {strategy2}: {lat_pg:.3f} s")

        ps_duckdb = json.dumps(plan_duckdb).replace("'", "''")
        ps_pg = json.dumps(plan_pg).replace("'", "''")
        insert_sql = f"""
            INSERT INTO public.plan_explored (name, strategy, plan, latency_s)
            VALUES (
                '{query_name}', 
                '{json.dumps(strategy1).replace("'", "''")}'::jsonb, 
                '{ps_duckdb}'::jsonb, 
                {lat_duckdb} 
            ),(
                '{query_name}', 
                '{json.dumps(strategy2).replace("'", "''")}'::jsonb, 
                '{ps_pg}'::jsonb, 
                {lat_pg}
            )ON CONFLICT (name, strategy) DO UPDATE
            SET 
                latency_s = EXCLUDED.latency_s,
                plan = EXCLUDED.plan;
            COMMIT;
        """
        pg.execute(insert_sql, (), retry_limit=1, fetch=False)
    
    except Exception as e:
        # 无论异常发生在哪个阶段，end1和end2都有值（真实值或默认60.0）
        print(f"执行查询 {query_name} 时出现异常,记录本次计划...")
        
        # 计算耗时：已执行的步骤用真实耗时，未执行的用默认120.0
        lat_duckdb = lat_duckdb if plan_duckdb != {} else 120.0
        lat_pg = lat_pg if plan_pg != {} else 120.0
        # 如果计划执行超时，则使用explain计划代替explain analyze计划
        if plan_duckdb == {}:
            plan_duckdb = execute_sql_to_plan(sql=sql, use_duckdb=True, exec=False)
        if plan_pg == {}:
            plan_pg = execute_sql_to_plan(sql=sql, use_duckdb=False, exec=False)

        # 插入记录（plan字段可留空或设为NULL）
        # 构造批量插入+冲突更新的SQL
        insert_sql = f"""
            INSERT INTO public.plan_explored (name, strategy, plan, latency_s)
            VALUES (
                '{query_name}', 
                '{json.dumps(strategy1).replace("'", "''")}'::jsonb, 
                '{json.dumps(plan_duckdb).replace("'", "''")}'::jsonb,
                {lat_duckdb}
            ),(
                '{query_name}', 
                '{json.dumps(strategy2).replace("'", "''")}'::jsonb, 
                '{json.dumps(plan_pg).replace("'", "''")}'::jsonb, 
                {lat_pg}
            )
            ON CONFLICT (name, strategy) DO UPDATE
            SET 
                latency_s = EXCLUDED.latency_s,
                plan = EXCLUDED.plan;
            COMMIT;
        """
        pg.execute(insert_sql, (), retry_limit=1, fetch=False)
        return

if __name__ == "__main__":

    argparse_parser = argparse.ArgumentParser(description="Plan Explorer for Mixed Execution Strategies")
    argparse_parser.add_argument("--dataset", type=str, default="tpcds10", help="Dataset name (default: tpcds10)")
    argparse_parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline plan generation")
    args = argparse_parser.parse_args()

    skip_baseline = args.skip_baseline

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

    schema = Schema(pg,f"{DATASET_NAME}")
    table_schemas = {}
    for data_table in schema.tables:
        table_name = data_table.name 
        table_schemas[table_name] = data_table.column_types
    # print("Schema:", table_schemas)

    qnumber = QUERY_TOTAL_NUMBER.get(DATASET_NAME)
    for i in range(1,qnumber+1):
        with open(f"{DATASET_PATH}/{DATASET_NAME}/queries/q{str(i).zfill(2)}.sql","r") as f:
            sql = f.read()
            if not skip_baseline:
                generate_baseline_from_sql(sql=sql,number=i)
            explore_plans_for_sql(sql=sql,number=i) 

