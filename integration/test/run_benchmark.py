import psycopg2
import argparse
import os
import json
import multiprocessing.pool

SUPPORTED_BENCHMARKS = ['tpcds10', 'tpch1']
DATA_DIR = '/home/windy/postgres/install/dataset/'

parser = argparse.ArgumentParser(description='test benchmark on postgres with pg_duckdb extension database')
parser.add_argument(
    '--bench',
    dest='bench',
    action='store',
    help='The benchmark(also database name and schema name) to run',
    default='tpch1',
)
parser.add_argument(
    '--duckdbexec',
    dest='duckdbexec',
    action='store',
    type=bool,
    help='whether to use duckdb execute or not',
    default=False,
)
parser.add_argument(
    '--hybrid',
    dest='hybrid',
    action='store',
    type=bool,
    help='whether to use hybrid storage(parquet) and hybrid plan or not',
    default=False,
)
parser.add_argument(
    '--query-list', dest='query_list', action='store', help='The list of queries to run (default = all)', default=''
)
parser.add_argument('--nthreads', dest='nthreads', action='store', type=int, help='The number of threads', default=0)
args = parser.parse_args()

bench =args.bench
if bench not in SUPPORTED_BENCHMARKS:
    raise ValueError(f"Unsupported benchmark: {bench}. Supported benchmarks are {SUPPORTED_BENCHMARKS}.")
query_dir = f'{DATA_DIR}{bench}/queries'
save_dir = f'{DATA_DIR}{bench}/plans'

duckdb_force_execution = args.duckdbexec
hybrid = args.hybrid

if hybrid and not duckdb_force_execution:
    raise ValueError("Hybrid storage requires duckdb execution to be enabled.")

queries = os.listdir(query_dir)
queries.sort()
if len(args.query_list) > 0:
    passing_queries = [x + '.sql' for x in args.query_list.split(',')]
    queries = [x for x in queries if x in passing_queries]
    queries.sort()


con = psycopg2.connect(database=f'{bench}',user='windy',host='localhost',port=5432)
c = con.cursor()

def run_query(q):
    print(q)
    with open(os.path.join(query_dir, q), 'r') as f:
        sql = f.read()
    try:
        c.execute("BEGIN;")
        c.execute(f'SET search_path to {bench};')
        c.execute(f'SET statement_timeout = \'1min\';')
        c.execute(f'SET duckdb.force_execution = {"1" if duckdb_force_execution else "0"};')
        c.execute(f'SET duckdb.hybrid = {"1" if duckdb_force_execution and hybrid else "0"}')
        c.execute(f'SET duckdb.max_workers_per_postgres_scan=8;')
        c.execute(f"EXPLAIN (ANALYZE,FORMAT JSON) {sql}")
        result = c.fetchone()[0][0]
        c.execute("COMMIT;")
    except psycopg2.Error as e:
        c.execute("ROLLBACK;")
        if isinstance(e, psycopg2.errors.QueryCanceled):
            print(f"Query {q} was canceled due to timeout.")
            return
        else:
            raise e
    with open(os.path.join(save_dir, q[:-4] + f'_{"hybrid" if hybrid else "duckdb" if duckdb_force_execution else "pg"}.json'), 'w') as f:
        json.dump(result, f, indent=4)

if args.nthreads == 0:
    for q in queries:
        run_query(q)
else:
    pool = multiprocessing.pool.ThreadPool(processes=args.nthreads)

    pool.map(run_query, queries)

