import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import List, Dict, Tuple,Any, Optional  # 类型提示

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

NODE_FEATURE_DIM = len(UNIFIED_OPERATOR_LIST) + 9  # one-hot 编码长度 + 数值特征数量

def process_query_groups(query_groups, group_size=20, pad_latency=600.0):
    """
    处理query_groups，按每个查询的计划构建训练组（每组20个）
    不足20个的用空字典{}和pad_latency填充
    
    Args:
        query_groups: 原始查询组列表，每个元素含'query_name'、'plans'、'latencies'等字段
        group_size: 每组计划数量（固定为20）
        pad_latency: 填充计划的时延标签
    
    Returns:
        plan_groups: 处理后的计划组列表，形状为 (总组数, 20)，每个元素是计划JSON（或空字典）
        latency_groups: 对应的时延组列表，形状为 (总组数, 20)
    """
    plan_groups = []
    latency_groups = []
    
    for group in query_groups:
        query_name = group['query_name']
        plans = group['plans']  # 该查询的所有计划（JSON列表）
        latencies = group['latencies']  # 对应计划的实际时延（列表）
        
        # 校验plans和latencies长度一致
        assert len(plans) == len(latencies), \
            f"查询 {query_name} 的plans与latencies长度不匹配"
        
        total_plans = len(plans)
        # 计算需要拆分的组数（向上取整）
        num_groups = (total_plans + group_size - 1) // group_size
        
        for i in range(num_groups):
            # 提取当前组的计划和时延（左闭右开区间）
            start_idx = i * group_size
            end_idx = start_idx + group_size
            current_plans = plans[start_idx:end_idx]
            current_latencies = latencies[start_idx:end_idx]
            
            # 计算需要填充的数量
            pad_count = group_size - len(current_plans)
            if pad_count > 0:
                # 填充空字典作为计划，填充pad_latency作为时延
                current_plans += [{} for _ in range(pad_count)]
                current_latencies += [pad_latency for _ in range(pad_count)]
            
            # 确认每组正好20个
            assert len(current_plans) == group_size and len(current_latencies) == group_size, \
                f"查询 {query_name} 的第{i}组计划数量异常"
            
            # 添加到结果列表
            plan_groups.append(current_plans)
            latency_groups.append(current_latencies)
    
    print(f"处理完成：共生成 {len(plan_groups)} 个训练组，每组 {group_size} 个计划")
    return plan_groups, latency_groups

class PlanDatasetListwise(Dataset):
    """Listwise 训练数据集：每组包含 20 个候选计划树（JSON）和对应的实际延迟"""
    def __init__(self, plan_groups: List[List[Dict]], latency_groups: List[List[float]]):
        """
        Args:
            plan_groups: 计划组列表，每个元素是 20 个计划树的 JSON 字典（根节点）
            latency_groups: 延迟组列表，每个元素是 20 个计划的实际执行延迟（与计划组一一对应）
        """
        assert len(plan_groups) == len(latency_groups), "计划组与延迟组数量必须一致"
        assert all(len(group) == 20 for group in plan_groups), "每组必须包含 20 个候选计划"
        assert all(len(latency) == 20 for latency in latency_groups), "每组延迟必须包含 20 个值"
        
        self.plan_groups = plan_groups
        self.latency_groups = latency_groups

    def __len__(self) -> int:
        return len(self.plan_groups)

    def __getitem__(self, idx: int) -> Tuple[List[Dict], torch.Tensor]:
        """返回单组数据：(20个计划树JSON, 20个实际延迟)"""
        plans = self.plan_groups[idx]
        latencies = torch.tensor(self.latency_groups[idx], dtype=torch.float32)
        return plans, latencies

class UnifiedPlanNode:
    """
    统一计划节点，统一处理 PG 和 DuckDB 风格的计划字典；
    如果 node 中含有 "Node Type"，则认为是 PG 计划，
    如果有 "operator_name"，则认为是 DuckDB 计划。
    """
    def __init__(self, node: dict):
        # duckdb执行计划和pg执行计划根节点不可能同时出现
        if 'Plan' in node and isinstance(node['Plan'], dict):
            node = node['Plan']
        if 'DuckDB Execution Plan' in node and isinstance(node['DuckDB Execution Plan'], dict):
            node = node['DuckDB Execution Plan']
        self.node_type = node.get("Node Type", "Unknown") if "Node Type" in node else node.get("operator_name", "Unknown")
        self.row_startup_cost = np.float32(node.get("Startup Cost", 0))
        self.row_total_cost = np.float32(node.get("Total Cost", 0))
        self.row_plan_rows = np.float32(node.get("Plan Rows", 0))
        self.row_plan_width = np.float32(node.get("Plan Width", 0))
        self.row_worker_planned = np.float32(node.get("Workers Planned", 1))
        self.col_operator_timing = np.float32(node.get("operator_timing", 0))
        self.col_result_set_size = np.float32(node.get("result_set_size", 0))
        self.col_operator_cardinality = np.float32(node.get("operator_cardinality", 0))
        self.col_operator_rows_scanned = np.float32(node.get("operator_rows_scanned", 0))

        # 递归构造子节点：支持 PG 格式（"Plans" 或 "children"）以及 DuckDB 格式（可能嵌套在 "DuckDB Execution Plan" 中）
        self.children = []
        for key in ["Plans", "children"]:
            if key in node and isinstance(node[key], list):
                for child in node[key]:
                    if isinstance(child, dict):
                        self.children.append(UnifiedPlanNode(child))
                break

    def unified_op_to_one_hot(self, op_name):
        """对算子进行 one-hot 编码；未匹配项归为 Unknown"""
        one_hot = np.zeros(len(UNIFIED_OPERATOR_LIST), dtype=np.float32)
        idx = OPERATOR_MAP.get(op_name, OPERATOR_MAP["Unknown"])
        one_hot[idx] = 1.0
        return one_hot

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
                 input_branches: int = 2,    # 输入分支数（子节点数量，默认二叉树，多叉树的时候可以选择最重要的两个孩子）
                 output_branches: int = 1): # 输出分支数（默认1，单编码输出）
        super().__init__()
        self.hidden_size = hidden_size
        self.node_feature_dim = node_feature_dim
        self.input_branches = input_branches
        self.output_branches = output_branches
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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

        self.zero_h = torch.zeros(1, self.hidden_size, device=self.device, dtype=torch.float32)
        self.zero_c = torch.zeros(1, self.hidden_size, device=self.device, dtype=torch.float32)

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
    def __init__(self, hidden_size: int = 8, k: int = 20, mode: str = "mlp", use_context: bool = True, attn_heads: int = 4, plan_featurizer: PlanFeaturizer = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.k = k
        self.mode = mode
        self.use_context = use_context
        self.device = device

        # 内置 PlanFeaturizer：若外部未提供，则构造一个默认的实例（使用全局 NODE_FEATURE_DIM）
        self.plan_featurizer = PlanFeaturizer(hidden_size=hidden_size, node_feature_dim=NODE_FEATURE_DIM) if plan_featurizer is None else plan_featurizer
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
            epochs: int = 100,
            lr: float = 1e-4,
            batch_size: int = 8,
            fine_tune_featurizer: bool = True,
            print_every: int = 10):
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
        - fine_tune_featurizer: 若为 True，则同时更新 featurizer 的参数（较慢）；默认仅训练 comparator。

        返回：最后一个 epoch 的平均训练损失。
        """
        self.to(self.device)
        self.plan_featurizer.to(self.device)

        # 数据集和 DataLoader
        dataset = PlanDatasetListwise(plan_groups, latency_groups)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_listwise_fn)

        # 优化器：默认只优化 comparator 的参数，除非 fine_tune_featurizer=True
        params = list(self.parameters())
        if fine_tune_featurizer:
            params += list(self.plan_featurizer.parameters())
            self.plan_featurizer.train()
        else:
            self.plan_featurizer.eval()

        optimizer = optim.Adam(params, lr=lr)

        eps = 1e-12
        last_epoch_loss = 0.0
        for epoch in range(1, epochs + 1):
            self.train()
            epoch_losses = []
            for step, (batch_plans, batch_latencies) in enumerate(dataloader, start=1):
                # batch_plans: list length=batch_size, 每项为 k 个 plan dict
                # batch_latencies: tensor (batch, k)
                batch_latencies = batch_latencies.to(self.device)

                # 构建 embeddings: (batch, k, hidden)
                embeddings_list = []
                # 如果不微调 featurizer，则在 no_grad 下计算 embedding，节省内存
                ctx = torch.no_grad if not fine_tune_featurizer else torch.enable_grad
                with ctx():
                    for group in batch_plans:
                        emb_k = []
                        for plan_json in group:
                            # Plan 接受 dict，构造 Plan 对象并编码
                            plan_obj = Plan(plan_json)
                            emb = self.plan_featurizer(plan_obj)  # 返回 [1, hidden]
                            emb_k.append(emb.squeeze(0).to(self.device))
                        embeddings_list.append(torch.stack(emb_k, dim=0))

                embeddings = torch.stack(embeddings_list, dim=0)  # (batch, k, hidden)

                # 前向：得到预测概率
                probs, scores = self(embeddings)

                # 目标分布：延迟越小概率越大，使用 softmax(-latency)
                target = torch.softmax(-batch_latencies, dim=-1)

                # 损失：交叉熵形式的 listwise 损失
                loss_tensor = - (target * torch.log(probs + eps)).sum(dim=1).mean()

                optimizer.zero_grad()
                loss_tensor.backward()
                optimizer.step()

                epoch_losses.append(loss_tensor.item())
                if step % print_every == 0:
                    print(f"Epoch {epoch} step {step}: loss={loss_tensor.item():.6f}")

            last_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            print(f"Epoch {epoch} completed, avg loss={last_epoch_loss:.6f}")

        return last_epoch_loss
    
def save(comparator: 'ListwiseComparator', path: str = "listwise_comparator.pth", optimizer:Optional[optim.Optimizer] = None):
    """将 comparator 和其 plan_featurizer 的 state_dict 一并保存到给定路径。

    会把两个 state_dict 包装到一个 dict 中，便于后续 load。
    """
    if comparator is None:
        raise ValueError("comparator must be provided")

    # 确保目录存在
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    state = {
        "comparator_state_dict": comparator.state_dict(),
        "featurizer_state_dict": comparator.plan_featurizer.state_dict() if hasattr(comparator, "plan_featurizer") else None,
        "meta": {
            "hidden_size": getattr(comparator, "hidden_size", None),
            "k": getattr(comparator, "k", None),
            "mode": getattr(comparator, "mode", None),
        }
    }
    if optimizer is not None:
        try:
            state["optimizer_state_dict"] = optimizer.state_dict()
        except Exception:
            # 忽略无法序列化的 optimizer
            state["optimizer_state_dict"] = None

    torch.save(state, path)
    print(f"Saved comparator and featurizer to {path}")

def load(path: str, comparator: Optional[ListwiseComparator] = None, optimizer:Optional[optim.Optimizer] = None, map_location:  Optional[torch.device] = None):
    """Load saved state from `path`.

    If `comparator` is provided, its `state_dict` and its `plan_featurizer` (if present)
    will be loaded in-place. If `optimizer` is provided and saved optimizer state
    exists, it will be restored as well.

    If `comparator` is None, the raw state dict will be returned for manual handling.
    """
    if map_location is None:
        map_location = device

    state = torch.load(path, map_location=map_location)

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

    print(f"Loaded comparator and featurizer from {path}")
    return state

def collate_listwise_fn(batch: List[Tuple[List[Dict], torch.Tensor]]) -> Tuple[List[List[Dict]], torch.Tensor]:
    """Listwise 数据加载_collate函数：批量处理组数据"""
    plan_groups = [item[0] for item in batch]  # 形状：(batch_size, 20)
    latency_groups = torch.stack([item[1] for item in batch], dim=0)  # 形状：(batch_size, 20)
    return plan_groups, latency_groups
