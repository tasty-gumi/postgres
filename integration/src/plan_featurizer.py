#一个计划特征化器，将数据库执行计划转换为向量embedding表示

import math
import numpy as np
import torch
import typing
import torch.nn as nn
from pg_interactor import Postgres
from lib.torch.sequential_data import Sequence
from collections.abc import Iterable
import json

class MultiInputLSTM(torch.nn.Module):
    def __init__(self, hidden_size, in_feature_size=None, input_branches=2, output_branches=1):
        super().__init__()
        self.out_feature_size = hidden_size
        self.input_branches = input_branches
        self.output_branches = output_branches

        if in_feature_size is None:
            in_feature_size = 0
        self.fc = torch.nn.Linear(hidden_size * input_branches + in_feature_size, (input_branches + 3) * hidden_size * output_branches)

    def forward(self, branches, input=None):
        hs, cs = zip(*branches)
        if input is not None:
            fc_input = torch.cat([*hs, input], dim=-1)
        else:
            fc_input = torch.cat(hs, dim=-1)

        lstm_in = self.fc(fc_input)
        a, i, o, *fs = lstm_in.chunk(self.input_branches + 3, -1)
        c = a.tanh() * i.sigmoid()
        for f, _c in zip(fs, cs):
            _c = _c.repeat(*(1 for i in range(_c.ndim - 1)), self.output_branches)
            c = c + f.sigmoid() * _c
        h = o.sigmoid() * c.tanh()
        return h, c

FEATURE_LIST = ['Node Type', 'Startup Cost',
                'Total Cost', 'Plan Rows', 'Plan Width']
LABEL_LIST = ['Actual Startup Time', 'Actual Total Time', 'Actual Self Time']

UNKNOWN_OP_TYPE = "Unknown"
PG_OPERATOR_LIST = ['Bitmap Heap Scan', 'Merge Join', 'Materialize', 'Function Scan', 'Limit', 'Gather', 
                    'Merge Append', 'Seq Scan', 'Nested Loop', 'Append', 'Incremental Sort', 'Hash', 'Sort',
                      'Hash Join', 'Custom Scan', 'CTE Scan', 'Memoize', 'SetOp', 'WindowAgg', 'Gather Merge',
                        'Index Only Scan', 'Group', 'Aggregate', 'BitmapAnd', 'Result', 'Subquery Scan', 
                        'Bitmap Index Scan', 'Unique', 'Index Scan']
DUCKDB_OPERATOR_LIST = ['PROJECTION', 'HASH_JOIN', 'COLUMN_DATA_SCAN', 'UNNEST', 'EXPLAIN_ANALYZE', 
                        'HASH_GROUP_BY', 'LEFT_DELIM_JOIN', 'TOP_N', 'PGDUCKDB_POSTGRES_SCAN ',
                          'STREAMING_LIMIT', 'PERFECT_HASH_GROUP_BY', 'DUMMY_SCAN', 'SEQ_SCAN ', 
                          'CROSS_PRODUCT', 'DELIM_SCAN', 'ORDER_BY', 'FILTER', 'STREAMING_WINDOW', 
                          'WINDOW', 'INOUT_FUNCTION', 'UNGROUPED_AGGREGATE']
# 默认的行列算子类型列表，用于独热编码，统一的算子类型列表，包含 PG 和 DuckDB 的算子
UNIFIED_OPERATOR_LIST = [UNKNOWN_OP_TYPE] + PG_OPERATOR_LIST + DUCKDB_OPERATOR_LIST
# 生成算子名称到索引的映射
OPERATOR_MAP = {op: i for i, op in enumerate(UNIFIED_OPERATOR_LIST)}

DATASET_PATH = f"/home/windy/postgres/install/dataset/"
QUERY_TOTAL_NUMBER = {"tpcds10":99,"tpch1":22}
DATASET_NAME = "tpcds10"

class UnifiedPlanNode:
    """
    统一计划节点，统一处理 PG 和 DuckDB 风格的计划字典；
    如果 node_dict 中含有 "Node Type"，则认为是 PG 计划，
    如果有 "operator_name"，则认为是 DuckDB 计划。
    """
    def __init__(self, node_dict: dict):
        if "Node Type" in node_dict:
            # PostgreSQL 风格
            self.node_type = node_dict.get("Node Type", "Unknown")
            self.startup_cost = float(node_dict.get("Startup Cost", 0))
            self.total_cost = float(node_dict.get("Total Cost", 0))
            self.plan_rows = float(node_dict.get("Plan Rows", 0))
            self.plan_width = float(node_dict.get("Plan Width", 0))
            self.actual_startup_time = float(node_dict.get("Actual Startup Time", 0))
            self.actual_total_time = float(node_dict.get("Actual Total Time", 0))
            self.actual_rows = float(node_dict.get("Actual Rows", 0))
            self.actual_loops = float(node_dict.get("Actual Loops", 0))
        elif "operator_name" in node_dict:
            # DuckDB 风格——注意这里用 operator_timing、operator_cardinality 等字段作为近似值
            self.node_type = node_dict.get("operator_name", "Unknown")
            timing = float(node_dict.get("operator_timing", 0))
            self.startup_cost = timing
            self.total_cost = timing
            self.plan_rows = float(node_dict.get("operator_cardinality", 0))
            self.plan_width = 0  # duckdb计划中没有宽度信息
            self.actual_startup_time = timing
            self.actual_total_time = timing
            self.actual_rows = float(node_dict.get("operator_cardinality", 0))
            self.actual_loops = 1
        else:
            # 不识别的节点类型
            self.node_type = "Unknown"
            self.startup_cost = self.total_cost = self.plan_rows = self.plan_width = 0
            self.actual_startup_time = self.actual_total_time = self.actual_rows = self.actual_loops = 0

        # 递归构造子节点：支持 PG 格式（"Plans" 或 "children"）以及 DuckDB 格式（可能嵌套在 "DuckDB Execution Plan" 中）
        self.children = []
        for key in ["Plans", "children"]:
            if key in node_dict and isinstance(node_dict[key], list):
                for child in node_dict[key]:
                    self.children.append(UnifiedPlanNode(child))
                break
        if not self.children and "DuckDB Execution Plan" in node_dict:
            dde = node_dict["DuckDB Execution Plan"]
            if "children" in dde and isinstance(dde["children"], list):
                for child in dde["children"]:
                    self.children.append(UnifiedPlanNode(child))

    def unified_op_to_one_hot(self, op_name):
        """对算子名称进行 one-hot 编码；未匹配项归为 Unknown"""
        one_hot = np.zeros(len(UNIFIED_OPERATOR_LIST), dtype=np.float32)
        idx = OPERATOR_MAP.get(op_name, OPERATOR_MAP["Unknown"])
        one_hot[idx] = 1.0
        return one_hot
    def safe_log(self,x):
        """保证数值至少为1后取对数"""
        return math.log(max(x, 1))

    def to_vector(self):
        """
        获取节点向量：由
          1. 节点类型的 one-hot 编码（使用统一的算子列表）
          2. 数值特征经过 log 转换后(Startup Cost, Total Cost, Plan Rows, Plan Width,
             Actual Startup Time, Actual Total Time, Actual Rows, Actual Loops)
        组成
        """
        op_vector = self.unified_op_to_one_hot(self.node_type)
        numeric_features = np.array([
            self.startup_cost,
            self.total_cost,
            self.safe_log(self.plan_rows),
            self.plan_width,
            self.actual_startup_time,
            self.actual_total_time,
            self.safe_log(self.actual_rows),
            self.actual_loops
        ], dtype=np.float32)
        return np.concatenate([op_vector, numeric_features])

    def __repr__(self):
        return f"UnifiedPlanNode({self.node_type}, rows={self.plan_rows}, total_cost={self.total_cost}, children={len(self.children)})"
class Plan:
    def __init__(self, json_dict: dict):
        """
        使用 JSON 字典初始化 Plan 对象。
        如果字典中包含 "Plan" 字段，则把它作为根节点，
        否则整个字典作为根节点。
        同时保存规划时间与执行时间（若存在）。
        """
        if "Plan" in json_dict:
            self.root = UnifiedPlanNode(json_dict["Plan"])
        else:
            self.root = UnifiedPlanNode(json_dict)
        self.planning_time = json_dict.get("Planning Time", None)
        self.execution_time = json_dict.get("Execution Time", None)

    def featurize(self):
        """
        返回整个计划树的向量树表示。
        每个节点用一个字典表示，包含 'features' 和 'children' 字段。
        """
        return self._featurize_node(self.root)

    def _featurize_node(self, node: UnifiedPlanNode):
        vec = node.to_vector()
        return {"features": vec, "children": [self._featurize_node(child)
                                                for child in node.children]}

    def __repr__(self):
        return (f"Plan(root={self.root}, planning_time={self.planning_time}, "
                f"execution_time={self.execution_time})")

class TreeLSTM(nn.Module):
    def __init__(self, feature_size, input_size):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Linear(input_size, feature_size, bias=True),
            nn.LayerNorm(feature_size),
            nn.LeakyReLU(),
            nn.Linear(feature_size, feature_size),
        )
        self.lstm = MultiInputLSTM(
            feature_size,
            feature_size,
            input_branches=2,
            output_branches=1,
        )
        self.tail = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.LeakyReLU(),
            nn.Linear(feature_size, 1),
        )

    def forward(self, seq : typing.Union[Sequence, typing.Iterable[Sequence]]):
        if not isinstance(seq, Sequence):
            if isinstance(seq, Iterable):
                seq = Sequence.concat(seq)
            else:
                raise TypeError(f"'{seq.__class__.__name__}' object is not a sequence")

        branches = [(seq['left_hidden'], seq['left_cell']), (seq['right_hidden'], seq['right_cell'])]
        input = seq['extra_input']
        normalized_input = self.preprocess(input)
        res_hidden, res_cell = self.lstm(branches, normalized_input)
        seq['node_hidden'], seq['node_cell'] = res_hidden, res_cell
        return seq

    def predict(self, embeddings):
        if not isinstance(embeddings, torch.Tensor):
            if isinstance(embeddings, Iterable):
                embeddings = torch.stack(tuple(embeddings), dim=0)
        return self.tail(embeddings).squeeze(-1)

# 在 PlanFeaturizer.get_flat_features 方法中使用 TreeCNN：
# ---------------------- 新增：向量树转Sequence + 全局embedding提取 ----------------------
class PlanEmbeddingGenerator:
    def __init__(self, feature_size: int = 128):
        """
        初始化计划embedding生成器
        :param feature_size: 最终embedding的维度（需与TreeLSTM的feature_size一致）
        """
        self.feature_size = feature_size
        # 计算节点输入特征维度（One-Hot长度 + 数值特征长度）
        self.input_size = len(UNIFIED_OPERATOR_LIST) + 8  # 8个数值特征
        # 初始化TreeLSTM模型
        self.treelstm = TreeLSTM(
            feature_size=self.feature_size,
            input_size=self.input_size
        )
        self.treelstm.eval()

    def _vector_tree_to_sequences(self, vector_tree: dict) -> tuple[Sequence, torch.Tensor, torch.Tensor]:
        """
        后序遍历向量树，转换为Sequence列表，并返回当前节点的hidden/cell状态
        """
        # 1. 递归处理所有子节点
        child_hiddens = []
        child_cells = []
        for child_tree in vector_tree["children"]:
            _, child_h, child_c = self._vector_tree_to_sequences(child_tree)
            child_hiddens.append(child_h)
            child_cells.append(child_c)

        # 2. 初始化当前节点的Sequence（关键修复：补充sequence_length参数）
        # 每个节点作为单个序列元素处理，因此sequence_length=1
        current_seq = Sequence(sequence_length=1)  
        
        # 2.1 处理子节点状态（适配MultiInputLSTM的2个输入分支）
        left_h = child_hiddens[0] if len(child_hiddens) >= 1 else torch.zeros(self.feature_size, dtype=torch.float32)
        left_c = child_cells[0] if len(child_cells) >= 1 else torch.zeros(self.feature_size, dtype=torch.float32)
        right_h = child_hiddens[1] if len(child_hiddens) >= 2 else torch.zeros(self.feature_size, dtype=torch.float32)
        right_c = child_cells[1] if len(child_cells) >= 2 else torch.zeros(self.feature_size, dtype=torch.float32)
        
        # 存入Sequence（增加batch维度）
        current_seq['left_hidden'] = left_h.unsqueeze(0)
        current_seq['left_cell'] = left_c.unsqueeze(0)
        current_seq['right_hidden'] = right_h.unsqueeze(0)
        current_seq['right_cell'] = right_c.unsqueeze(0)

        # 2.2 处理当前节点自身特征
        extra_input = torch.tensor(vector_tree["features"], dtype=torch.float32).unsqueeze(0)  # 增加batch维度
        current_seq['extra_input'] = extra_input

        # 3. 用TreeLSTM处理当前节点
        processed_seq = self.treelstm(current_seq)
        current_h = processed_seq['node_hidden'].squeeze(0)  # 去掉batch维度
        current_c = processed_seq['node_cell'].squeeze(0)

        return processed_seq, current_h, current_c

    def generate_global_embedding(self, plan: Plan) -> np.ndarray:
        """生成计划的全局embedding（根节点的node_hidden）"""
        vector_tree = plan.featurize()
        
        with torch.no_grad():  # 禁用梯度计算
            _, root_hidden, _ = self._vector_tree_to_sequences(vector_tree)
        
        global_embedding = root_hidden.cpu().numpy()
        assert global_embedding.shape == (self.feature_size,), f"embedding维度错误：预期{self.feature_size}维，实际{global_embedding.shape}"
        return global_embedding
    
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

    pg.set_settings("search_path","public")
    pg.set_settings("statement_timeout","1min")

    query = f"SELECT id,plan from plan_explored;"

    # 执行查询获取计划数据
    plans = pg.execute(query, retry_limit=1, fetch=True)
    
    # 初始化集合用于存储去重后的字段值
    node_types = set()
    operator_names = set()
    
    # 遍历所有计划并提取字段
    total_plans = len(plans)
    pe = PlanEmbeddingGenerator()
    for i, (plan_id, plan_json) in enumerate(plans):
        print(f"Processing plan {i+1}/{total_plans}")
        try:
            plan = Plan(plan_json)
            plan_embedding = pe.generate_global_embedding(plan)
            print(f"提取结果:{plan_embedding},正在修改数据库{plan_id}号计划embedding字段...")
            update_query = f"UPDATE plan_explored SET embedding = '{json.dumps(plan_embedding.tolist())}' WHERE id = {plan_id};"
            pg.execute(update_query,fetch=False)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in plan {i+1}: {str(e)}")
        except Exception as e:
            print(f"Error processing plan {i+1}: {str(e)}")
    
