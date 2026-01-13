# 一个数据库计划探索器，负责逻辑计划改写和物理计划搜索和枚举,并进行行列混合计划数据实际运行与采样收集
import json
import os
import argparse
import hashlib
from sqlglot import parse_one
from sqlglot.optimizer import optimize,RULES
from sqlglot.expressions import Select,Table
from pg_interactor import Schema,Postgres

DATASET_PATH = f"/home/windy/postgres/install/dataset/"
DATASET_NAMES = ["tpch1","tpcds10"]
DATASET_SIMPLE_NAMES = {"tpcds10": "ds", "tpch1": "h"}

CTREATE_TIME_MARKER = f"""
CREATE TABLE IF NOT EXISTS public.timing_marker (
    id INT PRIMARY KEY DEFAULT 1,
    start_time TIMESTAMPTZ NULL
);
INSERT INTO public.timing_marker (id) 
VALUES (1) 
ON CONFLICT (id) DO NOTHING;
"""

DROP_TIME_MARKER = f"""
DROP TABLE IF EXISTS public.timing_marker;
"""

UPDATE_TIME_MARKER_SQL = f"""
UPDATE public.timing_marker 
            SET start_time = clock_timestamp() 
            WHERE id = 1;
"""

SELECT_TIME_MARKER_SQL = f"""
SELECT ROUND((EXTRACT(EPOCH FROM (clock_timestamp() - start_time)) * 1000)::numeric, 3) AS exec_time_ms
FROM public.timing_marker 
WHERE id = 1;
"""

CREATE_TABLE_SQL= f"""
CREATE TABLE IF NOT EXISTS public.explored_plans (
    id SERIAL PRIMARY KEY,
    signature VARCHAR(64) NOT NULL,
    name VARCHAR(15) NOT NULL,
    sql TEXT DEFAULT NULL,
    plan JSONB DEFAULT NULL,
    latency_ms NUMERIC(10,3) DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_signature ON public.explored_plans(signature);
COMMIT;
"""

def enumerate_all_hybrid_plans(optimized_ast: Select, query_name: str):
    """
    为优化之后的表达式枚举所有可能的混合执行计划，Select表达式已经被处理成长链WITH子句的形式，预期的astCTE数量 >= 基表数量
    参数:
        optimized_ast: 优化后的Select表达式
        number: 查询编号
    """
    def insert_row_plan(col_plan: dict, row_plan: dict, ref_name: str):
        """
        原地修改列主计划(col_plan)，递归查找目标节点，将额外的行计划(row_plan)插入到 children 中。
        
        目标节点需满足：
        1. name 是 'PGDUCKDB_POSTGRES_SCAN' (注意处理可能存在的尾部空格)
        2. extra_info['Table'] 等于 ref_name
        """
        
        def find_and_insert(nodes):
            if not isinstance(nodes, list):
                return

            for node in nodes:
                # 1. 获取节点名称
                node_name = node.get('name', '')
                
                # 2. 获取 extra_info 字典
                extra_info = node.get('extra_info', {})
                
                # 3. 检查是否匹配目标条件
                # 条件A: 算子名称是 PGDUCKDB_POSTGRES_SCAN
                # 条件B: 涉及的表名是 ref_name
                if (node_name == 'PGDUCKDB_POSTGRES_SCAN ' and 
                    extra_info.get('Table') == ref_name):
                    
                    # 确保 children 列表存在
                    if 'children' not in node:
                        node['children'] = []
                    
                    # 4. 执行插入操作
                    # 注意：这里直接插入 row_plan 字典。根据你的示例，row_plan 应该是一个 {'Plan': ...} 结构的字典
                    node['children'].append(row_plan)
                    
                    # 找到后通常该分支是叶子节点（Scan节点），不需要继续向下递归查找该节点的子节点
                    # 但如果同一个表在查询中被多次引用（例如自连接），可能需要继续查找兄弟节点
                    # 这里不做 return，继续遍历同级节点的其他部分
                
                # 5. 递归处理子节点
                if 'children' in node:
                    find_and_insert(node['children'])

        try:
            # 定位 DuckDB 执行计划的入口列表
            # 结构通常是: col_plan -> 'Plan' -> 'DuckDB Execution Plan' -> [List of Nodes]
            duckdb_plan_root = col_plan.get('Plan', {}).get('DuckDB Execution Plan')
            
            if duckdb_plan_root:
                find_and_insert(duckdb_plan_root)
            else:
                print(f"[WARN]: 在 col_plan 中未找到 'DuckDB Execution Plan' 字段，无法插入 {ref_name} 的行计划。")
                
        except Exception as e:
            print(f"[ERROR]: 插入行计划失败: {e}")
            # 这里可以选择 raise 抛出异常，或者仅打印日志

    def get_row_cte_plans(cte_sqls:dict, cte_deps:dict, deps:list, plans:list, visited:set)->str:
        for dep in deps:
            if dep in visited:
                continue
            # 当前的依赖项还依赖其他CTE，递归处理
            if dep in cte_deps and len(cte_deps[dep]) > 0:
                get_row_cte_plans(cte_sqls=cte_sqls, cte_deps=cte_deps, deps=cte_deps[dep], plans=plans, visited=visited)
            plans.append(f"{dep} as ({cte_sqls[dep]})")
            visited.add(dep)

    print(f"[INFO]:正在处理查询{query_name}的行列混合计划枚举...")

    cte_list = optimized_ast.args.get("with")
    # 解析CTE依赖关系并且构造CTE名称到查询的映射
    cte_sqls = {}
    cte_deps= {}
    for cte in cte_list.expressions:
        cte_sqls[cte.alias] = cte.this.sql(dialect="postgres", pretty=True)
        cte_deps[cte.alias] = []
        for ref in cte.this.find_all(Select):
            for table in ref.find_all(Table):
                if table.name in cte_sqls.keys():
                    cte_deps[cte.alias].append(table.name)
    
    # 对每一个CTE单独执行行引擎计划，剩余的部分执行列引擎计划
    # 越靠近后面的CTE，在整体计划树上越靠近根节点，也就是行列转换点会更高
    total  = len(cte_sqls)
    idx = 0
    for name, sql in cte_sqls.items():
        idx += 1
        # 处理某个CTE到行引擎执行,先递归处理它的依赖项,将依赖项都放入cte计划中
        row_cte_plans = []
        visited = set()
        get_row_cte_plans(cte_sqls=cte_sqls, cte_deps=cte_deps, deps=cte_deps[name], plans=row_cte_plans, visited=visited)
        visited.add(name)
        row_cte = f"WITH {', '.join(row_cte_plans)}" if len(row_cte_plans) > 0 else ""
        row_sql = f"{row_cte} {sql}"
        # 对数据库表有写入操作必须增加提交操作，否则可能会让duckdb无法感知数据变化
        row_exec_sql = f"CREATE TEMP TABLE IF NOT EXISTS {name} AS {row_sql};COMMIT;"
        row_clean_sql = f"DROP TABLE IF EXISTS {name};COMMIT;"

        # 剩余的CTE和主查询使用列引擎的计划,单独把当前CTE从全部的CTE列表中剥离出来，即使出现CTE冗余,也不影响列引擎计划的正确性
        ast = optimized_ast.copy()
        ast.args["with"] = None
        for cte in cte_sqls.keys():
            if cte == name:
                continue
            ast = ast.with_(cte,cte_sqls[cte],append=True,dialect="postgres")
        columnar_sql = ast.sql(dialect="postgres", pretty=True)
        col_exec_sql = f"{columnar_sql};"

        # 合成我们最终的执行SQL
        exec_sql = f"""
SET duckdb.force_execution=off;
{UPDATE_TIME_MARKER_SQL}
{row_exec_sql}
SET duckdb.force_execution=on;
{col_exec_sql}
SET duckdb.force_execution=off;
{SELECT_TIME_MARKER_SQL}
"""
        sig = hashlib.sha256(exec_sql.encode('utf-8')).hexdigest()
        # 如果当前数据库中的explored_plans表中已经存在该signature的计划，则跳过执行
        existing_plans = pg.execute(sql=f"SELECT id FROM public.explored_plans WHERE signature = '{sig}';",fetch=True,retry_limit=1)
        if len(existing_plans) > 0:
            print(f"[INFO]:查询{query_name}的混合计划中，{sig}计划已存在，跳过执行")
            continue

        try:
            # 先直接执行这个混合sql的执行计划得到执行时间
            res = pg.execute(exec_sql,retry_limit=1,fetch=True)
            exec_time_ms = res[0][0]
            # 分别拿出各自引擎的计划，拿出列计划时需要保证行结果存在
            row_plan = pg.execute(sql=f"SET duckdb.force_execution = off; EXPLAIN (FORMAT JSON) {row_sql};",cache=False,retry_limit=1,fetch=True)[0][0][0]
            col_plan = pg.execute(sql=f"SET duckdb.force_execution = on; EXPLAIN (FORMAT JSON) {columnar_sql};",cache=False,retry_limit=1,fetch=True)[0][0][0]
            # 按照名称索引插入行计划，得到行列混合计划
            insert_row_plan(col_plan=col_plan,row_plan=row_plan,ref_name=name)
            # 清理临时行表中间结果
            pg.execute(row_clean_sql,retry_limit=1,fetch=False)
            # 保存上面探索的计划到explored_plans表中
            save_sql = f"""
INSERT INTO public.explored_plans (signature, name, sql, plan, latency_ms)
VALUES ('{sig}', '{query_name}', '{exec_sql.replace('"','').replace("'", "''")}'::TEXT, '{json.dumps(col_plan).replace("'", "''")}'::JSONB, {exec_time_ms});
COMMIT;"""
            pg.execute(save_sql,retry_limit=1,fetch=False)
            print(f"[INFO]:已探索{query_name} {idx}/{total} : 执行时间{exec_time_ms}ms")
        except Exception as e:
            print(f"[ERROR]:在枚举查询{query_name} {idx}/{total}的行列混合计划时出错")
            pg.execute(row_clean_sql,retry_limit=1,fetch=False)
            continue

if __name__ == "__main__":

    argparse_parser = argparse.ArgumentParser(description="Plan Explorer for Mixed Execution Strategies")
    argparse_parser.add_argument("--dataset", type=str, default="tpcds10", help="Dataset name (default: tpcds10)")
    args = argparse_parser.parse_args()

    dataset_name = args.dataset
    if dataset_name not in DATASET_NAMES:
        raise ValueError(f"[ERROR]:Unsupported dataset: {dataset_name}. Supported datasets are: {DATASET_NAMES}")

    pg = Postgres()
    db_config={
        "dbname":f"{dataset_name}",
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
    
    pg.set_settings("search_path",f"{dataset_name}")
    pg.set_settings("statement_timeout","1min")
    pg.set_settings("duckdb.max_workers_per_postgres_scan","8")
    pg.set_settings("duckdb.threads_for_postgres_scan","8")
    pg.set_settings("duckdb.convert_unsupported_numeric_to_double","1")
    pg.execute(CREATE_TABLE_SQL,retry_limit=1,fetch=False)
    pg.execute(CTREATE_TIME_MARKER,retry_limit=1,fetch=False)

    schema = Schema(pg,f"{dataset_name}")
    table_schemas = {}
    for data_table in schema.tables:
        table_schemas[data_table.name] = data_table.column_types

    if os.path.exists(f"{DATASET_PATH}/{dataset_name}/queries/{dataset_name}_train.txt") is False:
        raise FileNotFoundError(f"[ERROR]:你需要将探索SQL放置在该目录的此文件下:{DATASET_PATH}/{dataset_name}/queries/{dataset_name}_train.txt")
    
    with open(f"{DATASET_PATH}/{dataset_name}/queries/{dataset_name}_train.txt","r") as f:
        for i, line in enumerate(f, start=1):
            sql = line.strip()
            if not sql:
                print(f"Skipping empty query at line {i}.")
                continue
            try:  
                ast = parse_one(sql, dialect='postgres')
                optimized_ast = optimize(ast, schema=table_schemas, dialect='postgres', leave_tables_isolated=True, rules=RULES)
                enumerate_all_hybrid_plans(optimized_ast=optimized_ast,query_name=f"{DATASET_SIMPLE_NAMES[dataset_name]}-q{str(i).zfill(3)}")
            except Exception as e:
                print(f"Error processing query {i}: {e}")


        



