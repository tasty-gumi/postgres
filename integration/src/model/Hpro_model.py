import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import Iterable, List, Dict, Tuple, Optional,cast  # 类型提示

DEFAULT_MODEL_ID = 'demo'
FEATURE_LIST = ['Node Type', 'Startup Cost',
                'Total Cost', 'Plan Rows', 'Plan Width']
LABEL_LIST = ['Actual Startup Time', 'Actual Total Time', 'Actual Self Time']

CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if CUDA else "cpu")
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

NODE_FEATURE_DIM = len(UNIFIED_OPERATOR_LIST) + 7  # one-hot 编码长度 + 数值特征数量

MODEL_PATH = f"/home/windy/postgres/integration/src/model"

class UnifiedPlanNode:
    """
    统一计划节点：能够同时解析 PG 的标准字段和 DuckDB 的 extra_info
    """
    def __init__(self, node: dict):
        # 1. 拆包处理：处理 {"Plan": {...}} 或 {"DuckDB Execution Plan": [...]} 这种嵌套外壳
        if 'Plan' in node and isinstance(node['Plan'], dict):
            node = node['Plan']
        # 注意：DuckDB 的 JSON 根经常是一个 List，但在递归过程中通常是 Dict
        # 如果是列表，通常取第一个元素作为根（视具体 JSON 结构而定）
        if 'DuckDB Execution Plan' in node and isinstance(node['DuckDB Execution Plan'], list):
             node = node['DuckDB Execution Plan'][0]

        # 2. 识别节点类型
        # PG 使用 "Node Type", DuckDB 使用 "name"
        if "Node Type" in node:
            self.node_type = node["Node Type"]
            self.is_duckdb = 0.0
        elif "name" in node:
            self.node_type = node["name"]
            self.is_duckdb = 1.0
        else:
            self.node_type = "Unknown"
            self.is_duckdb = 0.0 # Default

        # 3. 提取特征 (Feature Extraction)
        self._extract_features(node)

        # 4. 递归构造子节点
        self.children = []
        
        # PG 的子节点通常在 "Plans" 中
        if "Plans" in node and isinstance(node["Plans"], list):
            for child in node["Plans"]:
                self.children.append(UnifiedPlanNode(child))
        
        # DuckDB 的子节点通常在 "children" 中
        if "children" in node and isinstance(node["children"], list):
            for child in node["children"]:
                # DuckDB 的 children 有时直接是 Dict，有时包裹在 List 里
                self.children.append(UnifiedPlanNode(child))
        
        # 特殊情况：DuckDB 节点内部包裹了 PG 节点 (如 PGDUCKDB_POSTGRES_SCAN)
        # 这种情况下，PG 的 Plan 可能藏在 children 里，已经被上面的逻辑处理了
        # 但有时 PGDUCKDB 会把 PG Plan 放在 extra_info 或其他字段，需根据实际 JSON 调整
        # 根据你提供的 JSON，PG Plan 是作为 children 列表的一个元素存在的，
        # 且该元素是一个 {"Plan": ...} 的字典，上面的递归逻辑应该能覆盖。

    def _extract_features(self, node: dict):
        """核心：特征对齐逻辑"""
        
        # --- A. 提取 Cardinality (行数) ---
        if not self.is_duckdb:
            # PG
            self.est_rows = float(node.get("Plan Rows", 0))
        else:
            # DuckDB: 也就是 extra_info -> Estimated Cardinality
            extra = node.get("extra_info", {})
            # 注意：JSON 中可能是字符串 "133111200"，需要强转
            card_str = str(extra.get("Estimated Cardinality", "0"))
            try:
                self.est_rows = float(card_str)
            except ValueError:
                self.est_rows = 0.0

        # --- B. 提取 Cost (代价) ---
        if not self.is_duckdb:
            # PG
            self.total_cost = float(node.get("Total Cost", 0))
            self.startup_cost = float(node.get("Startup Cost", 0))
        else:
            # DuckDB 无 Cost 概念，置 0
            self.total_cost = 0.0
            self.startup_cost = 0.0

        # --- C. 提取 Width (宽度/列数) ---
        if not self.is_duckdb:
            # PG: 直接有字节宽度
            self.width = float(node.get("Plan Width", 0))
        else:
            # DuckDB: 使用 Projections 列表长度作为宽度的代理
            extra = node.get("extra_info", {})
            projections = extra.get("Projections", [])
            if isinstance(projections, list):
                self.width = float(len(projections))
            elif isinstance(projections, str):
                # 有时是 "col1, col2" 字符串
                self.width = float(len(projections.split(',')))
            else:
                self.width = 1.0 # 默认值

        # --- D. 提取 Condition Complexity (过滤条件数量) ---
        # 这是一个新特征，用于弥补 DuckDB 没有 Cost 的信息缺失
        self.num_conditions = 0.0
        if not self.is_duckdb:
            # PG: 简单的统计 Filter 字符串长度，或者简单的 0/1
            if "Filter" in node:
                self.num_conditions = 1.0 + node["Filter"].count("AND")
        else:
            # DuckDB
            extra = node.get("extra_info", {})
            # 可能是 "Conditions" (List or Str) 或 "Filters"
            conds = extra.get("Conditions", extra.get("Filters", []))
            if isinstance(conds, list):
                self.num_conditions = float(len(conds))
            elif isinstance(conds, str):
                # 比如 "d_year=2002"
                self.num_conditions = 1.0
                if len(conds) > 0:
                     # 粗略估计复杂度
                     self.num_conditions += conds.count("AND") + conds.count("OR")

    def unified_op_to_one_hot(self, op_name):
        one_hot = np.zeros(len(UNIFIED_OPERATOR_LIST), dtype=np.float32)
        idx = OPERATOR_MAP.get(op_name, OPERATOR_MAP[UNKNOWN_OP_TYPE])
        one_hot[idx] = 1.0
        return one_hot

    def plan_node_to_vector(self):
        """
        生成数值向量。
        关键：必须使用 log1p 处理 Rows 和 Cost，因为它们即使在同一棵树里，
        PG 的 Cost 和 DuckDB 的 Cardinality 也可能不在一个数量级。
        """
        op_vector = self.unified_op_to_one_hot(self.node_type)
        
        numeric_features = np.array([
            # 1. 引擎标识 (非常重要)
            self.is_duckdb, 
            
            # 2. 基础数值 (Log处理)
            np.log1p(self.est_rows),      # 统一后的行数
            np.log1p(self.width),         # 统一后的宽度 (PG是字节, DuckDB是列数, 网络会自己学习区别)
            
            # 3. 代价特征 (DuckDB 为 0)
            np.log1p(self.total_cost),
            np.log1p(self.startup_cost),
            
            # 4. 复杂度特征
            np.log1p(self.num_conditions), # 过滤/连接条件数量
            
            # 5. 结构特征
            float(len(self.children))      # 子节点数量
            
        ], dtype=np.float32)
        
        return np.concatenate([op_vector, numeric_features])

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

BatchData = Tuple[List[List[Plan]], torch.Tensor]

class PlanDatasetListwise(Dataset):
    """Listwise 训练数据集：每组包含 k 个候选计划树（JSON）和对应的实际延迟"""
    def __init__(self, plan_groups: List[List[Dict]], latency_groups: List[List[float]],k: int =5):
        """
        Args:
            plan_groups: 计划组列表，每个元素是 5 个计划树的 JSON 字典（根节点）
            latency_groups: 延迟组列表，每个元素是 5 个计划的实际执行延迟（与计划组一一对应）
        """
        assert len(plan_groups) == len(latency_groups), "计划组与延迟组数量必须一致"
        assert all(len(group) == k for group in plan_groups), f"每组必须包含 {k} 个候选计划"
        assert all(len(latency) == k for latency in latency_groups), f"每组延迟必须包含 {k} 个值"

        self.latency_groups =  torch.tensor(latency_groups, dtype=torch.float32)
        print("正在预处理 Plan 对象，请稍候...")
        self.plan_objects_groups = []
        # 这里只做一次，之后训练直接取对象
        for group in plan_groups:
            obj_group = [Plan(p) for p in group]
            self.plan_objects_groups.append(obj_group)
        print("预处理完成！")

    def __len__(self) -> int:
        return len(self.plan_objects_groups)
    
    def __getitem__(self, idx: int) -> Tuple[List[Plan], torch.Tensor]:
        """返回单组数据：(k个计划树对象, k个实际延迟)"""
        return self.plan_objects_groups[idx], self.latency_groups[idx]
    
class MultiInputLSTM(nn.Module):
    def __init__(self, hidden_size, in_feature_size=None, input_branches=2, output_branches=1):
        """
        Multi-input LSTM-like aggregation unit for tree nodes.

        设计意图：针对树形结构（每个节点有多个子节点），将子节点的隐藏态(h, c)
        与当前节点的特征拼接后，使用一个全连接层计算类似 LSTM 门控的向量，
        再通过门控机制合成当前节点的 cell/state 和 hidden 输出。

        参数说明：
        - hidden_size: 单个分支（子节点）的隐藏向量维度（h 的维度）。
        - in_feature_size: 额外输入特征的维度（例如节点自身的投影特征）；
                           如果为 None，则视为 0。
        - input_branches: 输入分支数量（期望的最大子节点数），不足时用零态填充，
                          超过时只保留前面几个子节点（truncate）。
        - output_branches: 输出分支倍数，默认 1（保留与 hidden_size 相同的输出维度）。

        计算细节：
        - 全连接层 self.fc 的输入维度为 `hidden_size * input_branches + in_feature_size`。
        - self.fc 的输出维度为 `(input_branches + 3) * hidden_size * output_branches`，
          其中分成若干片段：a, i, o（分别对应 LSTM 的候选、输入门、输出门），
          以及每个子分支对应的 forget-like gate f（数量等于 input_branches），
          通过 `chunk(self.input_branches + 3, -1)` 划分。

        返回：
        - (h, c)：当前节点的 hidden 与 cell，shape 均为 [1, hidden_size * output_branches]
        """
        super().__init__()
        self.out_feature_size = hidden_size
        self.input_branches = input_branches
        self.output_branches = output_branches

        if in_feature_size is None:
            in_feature_size = 0
        # 全连接层负责一次性计算所有门和候选向量。
        # 注意：显式指定 device/dtype 保持与其它模块一致性（但 Module 中通常不建议这样硬编码）。
        self.fc = torch.nn.Linear(
            hidden_size * input_branches + in_feature_size,
            (input_branches + 3) * hidden_size * output_branches,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=torch.float32,
        )

    def forward(self, branches, input=None):
        """
        branches: list/tuple of (h, c) 对，应长度为 self.input_branches（或更少，会补零）
                  其中 h/c 的形状为 [1, hidden_size * output_branches]（本实现中通常 output_branches==1）。
        input: 可选的额外输入张量，形状为 [1, in_feature_size]，通常来自节点自身的投影特征。

        处理步骤：
        1. 将所有子节点的 h 与可选的 input 在最后一维拼接，作为全连接层输入；
        2. 通过 self.fc 计算一个长向量，再按 (input_branches + 3) 分割为 a, i, o, f1, f2, ...；
        3. 用 a 和 i 计算候选 cell（c = tanh(a) * sigmoid(i)）；
        4. 对每个子节点，用对应的 f gate 与子节点的 cell (_c) 做加权累加（类似 forget gate），
           注意这里会根据 output_branches 扩展 _c 的最后一维以匹配维度；
        5. 用 o gate 与 tanh(c) 计算最终的 hidden h；
        6. 返回 (h, c)，用于上层聚合或最终输出。

        说明：此模块并不是标准的 PyTorch LSTMCell，而是针对树聚合（tree-LSTM 风格）
        的简化/定制化实现，主要用于把多个子节点信息以门控的方式融合。
        """
        # branches 中每个元素为 (h, c)，先拆分为两组元组
        hs, cs = zip(*branches)

        # 拼接所有子节点的 h；如果有额外 input，一并拼接
        # hs: tuple -> list of tensors, each tensor shape: [1, hidden_size * output_branches]
        if input is not None:
            # input 的 shape 假定为 [1, in_feature_size]
            fc_input = torch.cat([*hs, input], dim=-1)
        else:
            fc_input = torch.cat(hs, dim=-1)

        # 通过全连接层得到用于门控的长向量
        lstm_in = self.fc(fc_input)

        # 将长向量按最后一维均等切分为 a, i, o, f1, f2, ... （数量为 input_branches + 3）
        a, i, o, *fs = lstm_in.chunk(self.input_branches + 3, -1)

        # 计算候选 cell（使用 tanh 和 sigmoid 模拟 LSTM 的候选与输入门）
        c = a.tanh() * i.sigmoid()

        # 对每个子节点的 cell 进行 forget-like 加权累加
        # fs 数量等于实际 input_branches；cs 对应子节点的 cell
        for f, _c in zip(fs, cs):
            # 如果 output_branches > 1，需要把子节点 _c 在最后一维重复以匹配维度
            _c = _c.repeat(*(1 for i in range(_c.ndim - 1)), self.output_branches)
            c = c + f.sigmoid() * _c

        # 最终 hidden 与标准 LSTM 类似：h = o * tanh(c)
        h = o.sigmoid() * c.tanh()
        return h, c

class PlanFeaturizer(nn.Module):
    def __init__(self, 
                 hidden_size: int,          # LSTM隐状态维度
                 node_feature_dim: int,     # 计划节点的特征向量维度（op_one_hot + 数值特征）
                 input_branches: int = 3,    # 输入分支数（子节点数量，默认二叉树，多叉树的时候可以选择最重要的两个孩子）
                 output_branches: int = 1): # 输出分支数（默认1，单编码输出）
        super().__init__()
        self.hidden_size = hidden_size
        self.node_feature_dim = node_feature_dim
        self.input_branches = input_branches
        self.output_branches = output_branches
        self.device = device
        
        # 线性层：将节点原始特征映射到hidden_size维度（适配MultiInputLSTM的输入）
        self.node_feature_proj = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_size,device=self.device,dtype=torch.float32),
            nn.LayerNorm(hidden_size,device=self.device,dtype=torch.float32),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size,device=self.device,dtype=torch.float32)
        )
        
        # 多输入LSTM单元（核心：子节点作为输入分支）
        self.multi_input_lstm = MultiInputLSTM(
            hidden_size=hidden_size,
            in_feature_size=hidden_size,  # 输入：映射后的节点特征（hidden_size维度）
            input_branches=input_branches,             # 根据子节点数量动态调整，初始化无意义
            output_branches=output_branches
        )
        # 使用 register_buffer，这样调用 model.to(device) 时它们会自动移动，
        # 且保存模型 state_dict 时也会包含它们
        self.register_buffer('zero_h', torch.zeros(1, self.hidden_size, device=self.device, dtype=torch.float32))
        self.register_buffer('zero_c', torch.zeros(1, self.hidden_size, device=self.device, dtype=torch.float32))

    def forward(self, plan: Plan) -> torch.Tensor:
        """
        对整个计划树编码，返回根节点的隐状态（全局编码）
        :param plan: 计划树对象（Plan类实例）
        :return: 根节点h，shape: [1, hidden_size * output_branches]（batch维度为1）
        """
        root_node = plan.root
        # 后序递归编码根节点
        root_h, _ = self._encode_node(root_node)
        return root_h

    def _encode_node(self, node: UnifiedPlanNode) -> tuple[torch.Tensor, torch.Tensor]:
        """
        递归编码单个节点（后序遍历）
        :param node: 计划节点（UnifiedPlanNode实例）
        :return: (当前节点的h, 当前节点的c)，shape均为 [1, hidden_size * output_branches]
        """
        # -------------------------- 步骤1：编码所有子节点 --------------------------
        child_hc = []  # 存储子节点的(h, c)，作为当前节点的输入分支
        for child in node.children:
            child_h, child_c = self._encode_node(child)
            child_hc.append((child_h, child_c))
        # 1. 截取前2个（多余子节点丢弃）
        child_hc = child_hc[:self.input_branches]
        # 3. 生成零张量（维度：[1, hidden_size]，匹配LSTM隐状态维度）

        # 4. 补零到指定长度（input_branches=2）
        pad_num = self.input_branches - len(child_hc)
        child_hc += [(self.zero_h, self.zero_c)] * pad_num
        
        # -------------------------- 步骤2：处理当前节点特征 --------------------------
        # 1. 获取节点原始特征向量（numpy -> torch）
        node_feat_np = node.plan_node_to_vector()
        node_feat = torch.from_numpy(node_feat_np).float().unsqueeze(0)  # [1, node_feature_dim]
        # 2. 映射到hidden_size维度（作为MultiInputLSTM的额外输入）
        proj_node_feat = self.node_feature_proj(node_feat)  # [1, hidden_size]

        # -------------------------- 步骤3：调用MultiInputLSTM编码当前节点 --------------------------
        # 调用MultiInputLSTM：子节点hc作为branches，映射后的节点特征作为input
        current_h, current_c = self.multi_input_lstm(
            branches=child_hc,
            input=proj_node_feat
        )

        return current_h, current_c

class ListwiseComparator(nn.Module):
    """
    Listwise 比较层：将 k 个 plan embedding 映射为 k 个归一化概率。

    接口说明：
    - 支持输入形状 `(k, hidden)` 或 `(batch, k, hidden)`。
    - 默认实现为 MLP 打分：可选地把全局上下文（k 个 embedding 的均值）拼接到每个 embedding 上再打分。
    - 也支持基于自注意力的变体（mode='attn'），先做 self-attention，再对每个位置打分。

    返回值：`probs, scores`，其中 `probs` 是归一化概率 (batch, k)，`scores` 是未归一化分数 (batch, k)
    """
    def __init__(self, hidden_size: int = 8, k: int = 5, mode: str = "mlp", use_context: bool = True, attn_heads: int = 4, plan_featurizer: PlanFeaturizer = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.k = k
        self.mode = mode
        self.use_context = use_context
        self.device = device

        # 内置 PlanFeaturizer：若外部未提供，则构造一个默认的实例（使用全局 NODE_FEATURE_DIM）
        self.plan_featurizer = PlanFeaturizer(hidden_size=self.hidden_size, node_feature_dim=NODE_FEATURE_DIM) if plan_featurizer is None else plan_featurizer
        if mode == "mlp":
            in_size = hidden_size * (2 if use_context else 1)
            self.scorer = nn.Sequential(
                nn.Linear(in_size, hidden_size, device=self.device, dtype=torch.float32),
                nn.ReLU(),
                nn.Linear(hidden_size, 1, device=self.device, dtype=torch.float32)
            )
        elif mode == "attn":
            # 使用 MultiheadAttention 进行跨候选间的信息交互
            self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=attn_heads, batch_first=True)
            self.scorer = nn.Linear(hidden_size, 1, device=self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, plan_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        embeddings: Tensor, shape will be (batch, k, hidden)
        返回 (probs, scores) -> both shape (batch, k)
        """
        if self.mode == "mlp":
            if self.use_context:
                context = plan_embeddings.mean(dim=1, keepdim=True)  # (batch, 1, hidden)
                context = context.expand(-1, self.k, -1)  # (batch, k, hidden)
                inp = torch.cat([plan_embeddings, context], dim=-1)  # (batch, k, hidden*2)
            else:
                inp = plan_embeddings
            scores = self.scorer(inp).squeeze(-1)  # (batch, k)

        else:  # attn
            # self-attention: queries=keys=values=embeddings
            attn_out, _ = self.attn(plan_embeddings, plan_embeddings, plan_embeddings)
            scores = self.scorer(attn_out).squeeze(-1)

        probs = torch.softmax(scores, dim=-1)
        return probs, scores

    def fit(self,
            plan_groups: List[List[Dict]],
            latency_groups: List[List[float]],
            epochs: int = 300,
            lr: float = 1e-3,
            batch_size: int = 8,
            train_featurizer: bool = True,
            print_every: int = 10,
            lambda_latency = 1.0,   # alpha 的缩放系数：控制对长尾延迟的关注度
            omega_topk = 3.0,        # beta 的加权系数：Top-k 样本的权重增加倍数
            tolerance_ratio = 0.01 ,# 默认允许模型选择计划和最优计划之间有5%的时延误差，视作不影响实践的物理计划实际性能
            patience = 7,       # 容忍多少个 epoch 验证集 loss 不下降
            min_delta = 0.0,  # 只有 loss 下降幅度超过这个值才算有效
            ):
        """
        在给定若干查询组（每组固定大小 k，例如 process_query_groups 的输出）的基础上训练 ListwiseComparator。

        训练逻辑（listwise）：
        - 对每个候选计划使用传入的 `plan_featurizer` 编码为 embedding（shape: [1, hidden])，
          组装成形状 (batch, k, hidden) 的张量并输入 comparator 得到预测概率 `probs`。
        - 目标分布使用 softmax(-latency)：延迟越小，目标概率越大（即越靠前）。
        - 损失使用交叉熵形式的 listwise 损失：loss = - sum(target * log(pred))，对 batch 求均值。

        参数：
        - plan_groups, latency_groups: 与 `process_query_groups` 返回格式一致的 list。
        - plan_featurizer: `PlanFeaturizer` 实例，用于将 plan JSON 编码为 embedding（Module）。
        - train_featurizer: 若为 True，则同时更新 featurizer 的参数（较慢）；默认仅训练 comparator。

        返回：最后一个 epoch 的平均训练损失。
        """
        total_samples = len(plan_groups)
        indices = list(range(total_samples))
        random.shuffle(indices) # 随机打乱索引
        
        split_idx = int(total_samples * 0.9)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # 构建训练集和验证集列表
        train_plans = [plan_groups[i] for i in train_indices]
        train_latencies = [latency_groups[i] for i in train_indices]
        
        val_plans = [plan_groups[i] for i in val_indices]
        val_latencies = [latency_groups[i] for i in val_indices]
        
        print(f"Dataset Split: Train={len(train_plans)}, Val={len(val_plans)}")

        self.to(self.device)
        self.plan_featurizer.to(self.device)

        # 数据集和 DataLoader
        # 训练集 Loader
        train_dataset = PlanDatasetListwise(train_plans, train_latencies)
        train_loader: Iterable[BatchData] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_listwise_fn)

        # 验证集 Loader (不需要 shuffle)
        val_dataset = PlanDatasetListwise(val_plans, val_latencies)
        val_loader: Iterable[BatchData] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_listwise_fn)

        # 优化器：默认只优化 comparator 的参数，除非 train_featurizer=True
        params = list(self.parameters())
        if train_featurizer:
            params += list(self.plan_featurizer.parameters())

        self.optimizer = optim.Adam(params, lr=lr)
        self.train_loss_history = []# 训练集历史记录
        self.val_loss_history = [] # 验证集历史记录

        eps = 1e-12
        # 计算LAL-loss和准确率
        def compute_loss_and_acc(embeddings:torch.Tensor, batch_latencies:torch.Tensor) -> tuple[torch.Tensor, float]:
            # batch_latencies: (B, K)
            
            # 1. 找到每组的最优延迟
            min_l = batch_latencies.min(dim=1, keepdim=True).values # (B, 1)
            
            # 2. 定义“有效最优集合” (Effectively Best Set)
            # 只要延迟不超过 min * (1 + tolerance)，就认为是同等优秀的计划
            # 例如：min=200ms, tolerance=0.05 -> 210ms 以内的都算最优
            threshold = min_l * (1 + tolerance_ratio)
            is_effectively_best = (batch_latencies <= threshold) # Mask (B, K)

            # 修正 Target 分布 (防止模型强行区分和TOP-1时延非常接近的计划)
            target_latencies = batch_latencies.clone()
            
            # 将所有“有效最优”计划的延迟强制设为 min_l这样 Softmax 之后，它们的概率就是完全一样的
            # Expand min_l to match shape (B, K)
            min_l_expanded = min_l.expand_as(batch_latencies)
            target_latencies[is_effectively_best] = min_l_expanded[is_effectively_best]
            
            # 接着做常规的归一化和 Softmax
            max_l = batch_latencies.max(dim=1, keepdim=True).values
            denominator = max_l - min_l
            denominator[denominator < 1e-6] = 1.0
            
            # 注意：这里用修正过的 target_latencies 来做归一化
            norm_latencies = (target_latencies - min_l) / denominator
            temperature = 0.1
            target = torch.softmax(-norm_latencies / temperature, dim=-1)

            beta = 1.0 + is_effectively_best.float() * omega_topk

            probs, scores = self(embeddings)
            
            relative_gap = (max_l - min_l) / (min_l + 1e-6)
            alpha = 1.0 + lambda_latency * torch.log1p(relative_gap)

            element_loss = - (target * torch.log(probs + eps))
            weighted_item_loss = beta * element_loss
            group_loss = weighted_item_loss.sum(dim=1, keepdim=True) * alpha
            loss_tensor = group_loss.mean()
            
            pred_idx = scores.argmax(dim=1) # (B,)
            
            pred_latency = batch_latencies.gather(1, pred_idx.unsqueeze(1)).squeeze(1) # (B,)
            
            # 判断：预测的延迟是否在容忍范围内？
            is_correct = (pred_latency <= threshold.squeeze(1)).float()
            acc = is_correct.mean()
            
            return loss_tensor, acc
        
                # -------------------------------------------------------------
        
        # [初始化] Early Stopping 状态变量
        # -------------------------------------------------------------
        best_val_loss = float('inf')  # 记录史上最低的验证 Loss
        patience_counter = 0          # 记录连续多少次没变好
        best_epoch = 0                # 记录最好的是第几个 epoch
        
        # -------------------------------------------------------------
        # 3. 训练主循环
        # -------------------------------------------------------------
        for epoch in range(1, epochs + 1):
            # === Training Phase ===
            self.train() # 启用 Dropout, BatchNorm 等
            if train_featurizer:
                self.plan_featurizer.train()
            else:
                self.plan_featurizer.eval()

            train_epoch_losses = []
            train_epoch_accs = []
            
            for step, batch_data in enumerate(train_loader, start=1):
                batch_plans_objs, batch_latencies = cast(BatchData, batch_data)
                batch_latencies = batch_latencies.to(self.device)

                # 构建 Embeddings
                embeddings_list = []
                # 训练时根据 train_featurizer 决定是否计算梯度
                ctx = torch.no_grad if not train_featurizer else torch.enable_grad
                with ctx():
                    for group in batch_plans_objs:
                        emb_k = []
                        for plan_obj in group:
                            emb = self.plan_featurizer(plan_obj)
                            emb_k.append(emb.squeeze(0).to(self.device))
                        embeddings_list.append(torch.stack(emb_k, dim=0))
                embeddings = torch.stack(embeddings_list, dim=0)

                # 计算 Loss 和 Acc
                loss_tensor, acc = compute_loss_and_acc(embeddings=embeddings, batch_latencies=batch_latencies)

                # 反向传播
                self.optimizer.zero_grad()
                loss_tensor.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0) # 加上梯度裁剪
                self.optimizer.step()

                train_epoch_losses.append(loss_tensor.item())
                train_epoch_accs.append(acc.item())

                if step % print_every == 0:
                    print(f"[Train] Epoch {epoch} step {step}: loss={loss_tensor.item():.4f}, acc={acc.item():.2%}")

            # === Validation Phase ===
            self.eval() # 切换到评估模式 (关闭 Dropout 等)
            self.plan_featurizer.eval() # Featurizer 必定是 Eval 模式
            
            val_epoch_losses = []
            val_epoch_accs = []
            
            with torch.no_grad(): # 验证阶段严禁计算梯度
                for batch_data in val_loader:
                    batch_plans_objs, batch_latencies = cast(BatchData, batch_data)
                    batch_latencies = batch_latencies.to(self.device)

                    # 构建 Embeddings (无梯度)
                    embeddings_list = []
                    for group in batch_plans_objs:
                        emb_k = []
                        for plan_obj in group:
                            emb = self.plan_featurizer(plan_obj)
                            emb_k.append(emb.squeeze(0).to(self.device))
                        embeddings_list.append(torch.stack(emb_k, dim=0))
                    embeddings = torch.stack(embeddings_list, dim=0)

                    # 计算 Loss 和 Acc
                    loss_tensor, acc = compute_loss_and_acc(embeddings, batch_latencies)
                    
                    val_epoch_losses.append(loss_tensor.item())
                    val_epoch_accs.append(acc.item())

            # === Epoch Summary ===
            avg_train_loss = float(np.mean(train_epoch_losses))
            avg_val_loss = float(np.mean(val_epoch_losses)) if val_epoch_losses else 0.0
            avg_val_acc = float(np.mean(val_epoch_accs)) if val_epoch_accs else 0.0
            
            self.train_loss_history.append(avg_train_loss)
            self.val_loss_history.append(avg_val_loss)
            
            print(f"Epoch {epoch} Done | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2%}")
            # -------------------------------------------------------------
            # Early Stopping 核心逻辑
            # -------------------------------------------------------------
            # 判断当前验证集 loss 是否比历史最好还小 (考虑 min_delta 阈值)
            if avg_val_loss < best_val_loss - min_delta:
                # === 情况 1: 创下新低 (Model Improved) ===
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0  # 重置计数器
                
                # 保存当前最好的模型
                # 注意：这里调用你之前写好的 save 函数
                save(self, DEFAULT_MODEL_ID, self.optimizer, extra_info={"epoch": epoch, "val_loss": avg_val_loss})
            
            else:
                # === 情况 2: 没有改善 (No Improvement) ===
                patience_counter += 1
                print(f"   >>> No improvement for {patience_counter}/{patience} epochs.")
                
                if patience_counter >= patience:
                    print(f"\n[Early Stopping] Training stopped manually.")
                    print(f"Best Val Loss was {best_val_loss:.6f} at Epoch {best_epoch}.")
                    print(f"Restoring best model weights...")
                    
                    # 关键一步：停止前，把内存中的模型加载回最好的状态
                    # 这样函数返回后，外部拿到的 model 就是最好的那个，而不是最后过拟合的那个
                    load(DEFAULT_MODEL_ID, self, self.optimizer)
                    break # 跳出 epoch 循环


        return self.train_loss_history, self.val_loss_history
    
def save(comparator: 'ListwiseComparator', model_id: str, optimizer: Optional[optim.Optimizer] = None, extra_info: Dict = None):
    """
    保存模型、优化器状态以及训练/验证 Loss 记录。
    """
    if comparator is None:
        raise ValueError("comparator must be provided")

    dirpath = os.path.dirname(MODEL_PATH)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # 1. 尝试获取 Optimizer
    if optimizer is None and hasattr(comparator, 'optimizer'):
        optimizer = comparator.optimizer

    # 2. 获取 Loss History (适配新的 fit 函数)
    # 优先获取拆分后的 history，如果不存在则尝试获取旧版的 loss_history
    train_loss = getattr(comparator, 'train_loss_history', getattr(comparator, 'loss_history', []))
    val_loss = getattr(comparator, 'val_loss_history', [])

    state = {
        # 保存主模型参数 (包含内部的 plan_featurizer 参数)
        "comparator_state_dict": comparator.state_dict(),
        
        # 单独保存 featurizer 参数 (可选，方便仅加载特征提取器用于其他任务)
        "featurizer_state_dict": comparator.plan_featurizer.state_dict() if hasattr(comparator, "plan_featurizer") else None,
        
        # 保存 Loss 曲线
        "train_loss_history": train_loss,
        "val_loss_history": val_loss,
        
        # 元数据
        "meta": {
            "hidden_size": getattr(comparator, "hidden_size", None),
            "k": getattr(comparator, "k", None),
            "mode": getattr(comparator, "mode", None),
            "extra_info": extra_info 
        }
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(state, f"{MODEL_PATH}/Hpro_{model_id}.pth")
    print(f"Model saved to {MODEL_PATH}/Hpro_{model_id}.pth (Success)")

def load(model_id:str, comparator: Optional[ListwiseComparator] = None, optimizer:Optional[optim.Optimizer] = None, map_location:  Optional[torch.device] = None):
    """Load saved state from `path`.

    If `comparator` is provided, its `state_dict` and its `plan_featurizer` (if present)
    will be loaded in-place. If `optimizer` is provided and saved optimizer state
    exists, it will be restored as well.

    If `comparator` is None, the raw state dict will be returned for manual handling.
    """
    if map_location is None:
        map_location = device

    state = torch.load(f"{MODEL_PATH}/Hpro_{model_id}.pth", map_location=map_location)

    # 如果不提供 comparator，返回原始 state 以便用户自行处理
    if comparator is None:
        return state

    # 加载 comparator 的参数
    comp_sd = state.get("comparator_state_dict")
    if comp_sd is not None:
        comparator.load_state_dict(comp_sd)

    # 加载内部 featurizer（如果存在并且保存了）
    feat_sd = state.get("featurizer_state_dict")
    if feat_sd is not None and hasattr(comparator, "plan_featurizer"):
        try:
            comparator.plan_featurizer.load_state_dict(feat_sd)
        except Exception:
            # 若 featurizer 架构不匹配，给出提示但不抛出
            print("Warning: failed to load plan_featurizer state_dict (architecture mismatch?)")

    # 恢复 optimizer（如果提供且保存了）
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        try:
            optimizer.load_state_dict(state.get("optimizer_state_dict"))
        except Exception:
            print("Warning: failed to load optimizer state_dict")

    # 将 comparator 移到当前 device
    try:
        comparator.to(device)
    except Exception:
        pass

    print(f"Loaded comparator and featurizer from {MODEL_PATH}/Hpro_{model_id}.pth")
    return state

def collate_listwise_fn(batch: List[Tuple[List[Dict], torch.Tensor]]) -> Tuple[List[List[Dict]], torch.Tensor]:
    """Listwise 数据加载_collate函数：批量处理组数据"""
    plan_groups = [item[0] for item in batch]  # 形状：(batch_size, k)
    latency_groups = torch.stack([item[1] for item in batch], dim=0)  # 形状：(batch_size, k)
    return plan_groups, latency_groups
