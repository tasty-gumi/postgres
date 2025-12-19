#一个计划特征化器，将数据库执行计划转换为向量embedding表示,是一个计划特征化的预处理模型,和训练阶段需要训练的计划选择模型(排序模型)解耦

import math
import numpy as np
import torch
import typing
import torch.nn as nn
from pg_interactor import Postgres
from lib.torch.sequential_data import Sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim
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
WORKLOAD_BASE_TABLE_LIST  = [0.0] * 32
WORKLOAD_MERTIRAZED_TABLE_LIST = [0.0] * 32
# 生成算子名称到索引的映射
OPERATOR_MAP = {op: i for i, op in enumerate(UNIFIED_OPERATOR_LIST)}

DATASET_PATH = f"/home/windy/postgres/install/dataset/"
QUERY_TOTAL_NUMBER = {"tpcds10":99,"tpch1":22}
DATASET_NAME = "tpcds10"

class UnifiedPlanNode:
    """
    统一计划节点，统一处理 PG 和 DuckDB 风格的计划字典；
    如果 node 中含有 "Node Type"，则认为是 PG 计划，
    如果有 "operator_name"，则认为是 DuckDB 计划。
    """
    def __init__(self, node: dict):
        # duckdb执行计划和pg执行计划根节点不可能同时出现
        if 'Plan' in node:
            node = node['Plan']
        if 'DuckDB Execution Plan' in node:
            node = node['DuckDB Execution Plan']
        self.node_type = node.get("Node Type", "Unknown") if "Node Type" in node else node.get("operator_name", "Unknown")
        if(self.node_type == "Unknown"):
            print(f"发现未知算子类型:{node}")
        self.is_duckdb_plan_type = self.node_type in DUCKDB_OPERATOR_LIST 
        self.row_startup_cost = float(node.get("Startup Cost", 0))
        self.row_total_cost = float(node.get("Total Cost", 0))
        self.row_plan_rows = float(node.get("Plan Rows", 0))
        self.row_plan_width = float(node.get("Plan Width", 0))
        self.row_worker_planned = float(node.get("Workers Planned", 1))
        self.col_operator_timing = float(node.get("operator_timing", 0))
        self.col_result_set_size = float(node.get("result_set_size", 0))
        self.col_operator_cardinality = float(node.get("operator_cardinality", 0))
        self.col_operator_rows_scanned = float(node.get("operator_rows_scanned", 0))

        # 递归构造子节点：支持 PG 格式（"Plans" 或 "children"）以及 DuckDB 格式（可能嵌套在 "DuckDB Execution Plan" 中）
        self.children = []
        for key in ["Plans", "children"]:
            if key in node and isinstance(node[key], list):
                for child in node[key]:
                    self.children.append(UnifiedPlanNode(child))
                break
        # if not self.children and "DuckDB Execution Plan" in node:
        #     dde = node["DuckDB Execution Plan"]
        #     if "children" in dde and isinstance(dde["children"], list):
        #         for child in dde["children"]:
        #             self.children.append(UnifiedPlanNode(child))

    def unified_op_to_one_hot(self, op_name):
        """对算子名称进行 one-hot 编码；未匹配项归为 Unknown"""
        one_hot = np.zeros(len(UNIFIED_OPERATOR_LIST), dtype=np.float32)
        idx = OPERATOR_MAP.get(op_name, OPERATOR_MAP["Unknown"])
        one_hot[idx] = 1.0
        return one_hot
    def safe_log(self,x):
        """保证数值至少为1后取对数"""
        return math.log(max(x, 1))

    def plan_node_to_vector(self):
        """
        获取节点向量：由
          1. 节点类型的 one-hot 编码（使用统一的算子列表）
          2. 数值特征经过 log 转换后(Startup Cost, Total Cost, Plan Rows, Plan Width,
             Actual Startup Time, Actual Total Time, Actual Rows, Actual Loops)
        组成
        """
        op_vector = self.unified_op_to_one_hot(self.node_type)
        numeric_features = np.array([
            self.row_startup_cost,
            self.row_total_cost,
            self.row_plan_rows,
            self.row_plan_width,
            self.row_worker_planned,
            self.col_operator_timing,
            self.col_result_set_size,
            self.col_operator_cardinality,
            self.col_operator_rows_scanned,
        ], dtype=np.float32)
        return np.concatenate([op_vector, numeric_features])

    def __repr__(self):
        return f"UnifiedPlanNode({self.node_type}, rows={self.row_plan_rows}, total_cost={self.row_total_cost}, children={len(self.children)})"
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
        vec = node.plan_node_to_vector()
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
        self.input_size = len(UNIFIED_OPERATOR_LIST) + 9  # 算子特征+9个数值特征
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
        feat_len = len(vector_tree["features"])
        if feat_len != self.input_size:
             raise RuntimeError(f"节点特征长度不匹配: 实际 {feat_len}，期望 {self.input_size}；请检查 UnifiedPlanNode.to_vector 和 PlanEmbeddingGenerator.input_size 设置。 节点数据：{vector_tree['features']}")
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
    
class PlanDataset(Dataset):
    """计划数据集，返回向量树（字典）和张量标签"""
    def __init__(self, plan_list):
        self.valid_data = []
        for plan_obj, exec_time in plan_list:
            try:
                exec_time = float(exec_time)
                if exec_time > 0:
                    vector_tree = plan_obj.featurize()
                    # 关键修复：将标签转换为PyTorch张量
                    label_tensor = torch.tensor(math.log(exec_time), dtype=torch.float32)
                    self.valid_data.append((vector_tree, label_tensor))
            except (ValueError, TypeError, Exception) as e:
                print(f"过滤无效样本: {e}")
                continue

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        return self.valid_data[idx]  # 返回 (vector_tree_dict, label_tensor)

# 训练函数无需修改（保持之前的 custom_collate）
def train_treelstm(generator, train_dataset, val_dataset, epochs=30, batch_size=16, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.treelstm.to(device)
    generator.treelstm.train()

    optimizer = optim.Adam(generator.treelstm.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 自定义collate_fn：保持样本结构
    def custom_collate(batch):
        return batch[0]  # 直接返回单个样本

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate
    )

    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(epochs):
        train_loss = 0.0
        total_train = 0
        optimizer.zero_grad()

        # 训练阶段
        for vector_tree, label in train_loader:
            # 现在label是张量，可以调用.to(device)
            label = label.to(device)

            _, root_hidden, _ = generator._vector_tree_to_sequences(vector_tree)
            embedding = root_hidden.to(device).unsqueeze(0)

            pred = generator.treelstm.predict(embedding)
            loss = criterion(pred, label.unsqueeze(0))  # 匹配batch维度
            loss.backward()

            train_loss += loss.item()
            total_train += 1

            if total_train % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

        if total_train % batch_size != 0:
            optimizer.step()
            optimizer.zero_grad()

        # 验证阶段
        val_loss = 0.0
        total_val = 0
        generator.treelstm.eval()
        with torch.no_grad():
            for vector_tree, label in val_loader:
                label = label.to(device)

                _, root_hidden, _ = generator._vector_tree_to_sequences(vector_tree)
                embedding = root_hidden.to(device).unsqueeze(0)

                pred = generator.treelstm.predict(embedding)
                val_loss += criterion(pred, label.unsqueeze(0)).item()
                total_val += 1
        generator.treelstm.train()

        avg_train_loss = train_loss / total_train if total_train > 0 else 0
        avg_val_loss = val_loss / total_val if total_val > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss and total_val > 0:
            best_val_loss = avg_val_loss
            best_weights = generator.treelstm.state_dict()

    if best_weights is not None:
        generator.treelstm.load_state_dict(best_weights)
        torch.save(best_weights, "best_treelstm_weights.pth")
        print(f"训练完成，最佳验证损失: {best_val_loss:.4f}")
    else:
        print("警告：未保存有效权重")
    
    generator.treelstm.eval()
    return generator

if __name__ == "__main__":
    pg = Postgres()
    db_config = {
        "dbname": "tpcds10",
        "host": "localhost",
        "user": "windy",
        "password": "",
        "port": 5432
    }

    pg.setup(
        dbname=db_config["dbname"],
        host=db_config["host"],
        user=db_config["user"],
        password=db_config["password"],
        port=db_config["port"]
    )

    pg.set_settings("search_path", "public")
    pg.set_settings("statement_timeout", "1min")

    # 1. 读取计划数据（包含执行时间）
    print("加载计划数据及执行时间...")
    query = "SELECT  id, plan, latency_s FROM plan_explored;"
    plans_with_labels = pg.execute(query, retry_limit=1, fetch=True)
    total_plans = len(plans_with_labels)
    print(f"共加载 {total_plans} 条计划数据")

    # 2. 解析计划并准备数据
    train_data = []  # (Plan对象, execution_time)
    all_plans = []   # (plan_id, Plan对象)
    for i, (plan_id, plan_json, exec_time) in enumerate(plans_with_labels):
        try:
            plan = Plan(plan_json)
            all_plans.append((plan_id, plan))
            train_data.append((plan, exec_time))
        except Exception as e:
            print(f"解析计划 {plan_id} 失败: {str(e)}")
            continue

    # 3. 划分训练集和验证集
    print("划分训练集和验证集...")
    train_samples, val_samples = train_test_split(train_data, test_size=0.2, random_state=42)
    train_dataset = PlanDataset(train_samples)
    val_dataset = PlanDataset(val_samples)
    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")

    # 4. 初始化并训练模型
    print("开始训练TreeLSTM模型...")
    pe = PlanEmbeddingGenerator(feature_size=128)
    if len(train_dataset) > 0 and len(val_dataset) > 0:
        pe = train_treelstm(pe, train_dataset, val_dataset, epochs=30, batch_size=16)
    else:
        print("警告：训练数据不足，使用随机权重（不推荐）")

    # 5. 生成嵌入并更新数据库
    print("生成嵌入并更新数据库...")
    for i, (plan_id, plan) in enumerate(all_plans):
        try:
            print(f"处理计划 {i+1}/{len(all_plans)} (ID: {plan_id})")
            plan_embedding = pe.generate_global_embedding(plan)
            embedding_str = json.dumps(plan_embedding.tolist())
            # 参数化查询避免SQL注入
            update_query = f"""
                UPDATE plan_explored 
                SET embedding = '{embedding_str}' 
                WHERE id = {plan_id};
                COMMIT;
            """
            pg.execute(update_query, fetch=False)
        except Exception as e:
            print(f"更新计划 {plan_id} 失败: {str(e)}")

    print("所有计划处理完成")
