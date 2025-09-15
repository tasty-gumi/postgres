#一个计划特征化器，将数据库执行计划转换为向量embedding表示

import math
import numpy as np
import torch

# 默认算子列表，用于独热编码
DEFAULT_OPERATOR_LIST = [
    "Custom Scan", "Seq Scan", "Hash Join",
    "Projection", "Top_N", "Table Scan", "Index Scan"
]

class PlanNode:
    def __init__(self, node_dict: dict):
        """
        根据传入的节点字典构造一个计划节点，
        保存节点属性及其子节点列表。
        主要属性包括：
          - 节点类型 (Node Type)
          - 估计代价和实际代价 (Startup Cost, Total Cost)
          - 估计基数与实际基数 (Plan Rows, Actual Rows)
          - 其它数值属性包括 Plan Width, Actual Startup/Total Time, Actual Loops 等
        """
        self.node_type = node_dict.get("Node Type", "")
        self.startup_cost = float(node_dict.get("Startup Cost", 0))
        self.total_cost = float(node_dict.get("Total Cost", 0))
        self.plan_rows = float(node_dict.get("Plan Rows", 0))
        self.plan_width = float(node_dict.get("Plan Width", 0))
        self.actual_startup_time = float(node_dict.get("Actual Startup Time", 0))
        self.actual_total_time = float(node_dict.get("Actual Total Time", 0))
        self.actual_rows = float(node_dict.get("Actual Rows", 0))
        self.actual_loops = float(node_dict.get("Actual Loops", 0))
        # 保存其它附加属性
        self.attributes = { k: v for k, v in node_dict.items() if k not in (
            "Node Type", "Startup Cost", "Total Cost", "Plan Rows", "Plan Width",
            "Actual Startup Time", "Actual Total Time", "Actual Rows", "Actual Loops",
            "children", "DuckDB Execution Plan"
        )}
        # 递归构造子节点列表；
        # 考虑到 DuckDB 计划可能在 "DuckDB Execution Plan" 内嵌 "children"
        self.children = []
        if "children" in node_dict and isinstance(node_dict["children"], list):
            for child in node_dict["children"]:
                self.children.append(PlanNode(child))
        elif "DuckDB Execution Plan" in node_dict:
            dde = node_dict["DuckDB Execution Plan"]
            if "children" in dde and isinstance(dde["children"], list):
                for child in dde["children"]:
                    self.children.append(PlanNode(child))

    def to_vector(self, operator_to_index=None):
        """
        将当前节点编码为一个固定长度的向量表示，
        向量由两部分组成：
          1. 节点类型的独热编码（根据 operator_to_index，若不在映射中则全0）
          2. 数值特征数组：
             [log(Startup Cost), log(Total Cost), log(Plan Rows),
              log(Plan Width), log(Actual Startup Time + 1e-6),
              log(Actual Total Time + 1e-6), log(Actual Rows), log(Actual Loops)]
        """
        if operator_to_index is None:
            operator_to_index = {op: i for i, op in enumerate(DEFAULT_OPERATOR_LIST)}
        one_hot = [0] * len(operator_to_index)
        if self.node_type in operator_to_index:
            one_hot[operator_to_index[self.node_type]] = 1

        def safe_log(x):
            return math.log(max(x, 1))
        numeric_feats = [
            safe_log(self.startup_cost),
            safe_log(self.total_cost),
            safe_log(self.plan_rows),
            safe_log(self.plan_width),
            safe_log(self.actual_startup_time + 1e-6),
            safe_log(self.actual_total_time + 1e-6),
            safe_log(self.actual_rows),
            safe_log(self.actual_loops)
        ]
        return np.array(one_hot + numeric_feats, dtype=np.float32)

    def __repr__(self):
        return (f"PlanNode({self.node_type}, plan_rows={self.plan_rows}, "
                f"total_cost={self.total_cost}, children={len(self.children)})")

class Plan:
    def __init__(self, json_dict: dict):
        """
        使用 JSON 字典初始化 Plan 对象。
        如果字典中包含 "Plan" 字段，则把它作为根节点，
        否则整个字典作为根节点。
        同时保存规划时间与执行时间（若存在）。
        """
        if "Plan" in json_dict:
            self.root = PlanNode(json_dict["Plan"])
        else:
            self.root = PlanNode(json_dict)
        self.planning_time = json_dict.get("Planning Time", None)
        self.execution_time = json_dict.get("Execution Time", None)

    def featurize(self):
        """
        返回整个计划树的向量树表示。
        每个节点用一个字典表示，包含 'features' 和 'children' 字段。
        """
        operator_to_index = {op: i for i, op in enumerate(DEFAULT_OPERATOR_LIST)}
        return self._featurize_node(self.root, operator_to_index)

    def _featurize_node(self, node: PlanNode, operator_to_index: dict):
        vec = node.to_vector(operator_to_index)
        return {"features": vec, "children": [self._featurize_node(child, operator_to_index)
                                                for child in node.children]}

    def __repr__(self):
        return (f"Plan(root={self.root}, planning_time={self.planning_time}, "
                f"execution_time={self.execution_time})")


class PlanFeaturizer:
    def __init__(self, plan: Plan):
        self.plan = plan

    def get_feature_tree(self):
        """
        返回计划树的向量树结构表示
        """
        return self.plan.featurize()

    def get_flat_features(self):
        """
        将计划树展平，返回所有结点的特征向量列表
        """
        flat_feats = []
        def traverse(node):
            flat_feats.append(node["features"])
            for child in node["children"]:
                traverse(child)
        traverse(self.get_feature_tree())
        return flat_feats

# 示例用法：
if __name__ == "__main__":
    # 假设 sample_plan 是一个通过 EXPLAIN (FORMAT JSON) 得到的计划字典
    sample_plan = {
        "Plan": {
            "Node Type": "Custom Scan",
            "Startup Cost": 0.0,
            "Total Cost": 0.0,
            "Plan Rows": 0,
            "Plan Width": 0,
            "Actual Startup Time": 0.001,
            "Actual Total Time": 0.001,
            "Actual Rows": 0,
            "Actual Loops": 1,
            "DuckDB Execution Plan": {
                "children": [
                    {
                        "Node Type": "Top_N",
                        "Startup Cost": 0.0,
                        "Total Cost": 0.0,
                        "Plan Rows": 100,
                        "Plan Width": 0,
                        "Actual Startup Time": 0.0,
                        "Actual Total Time": 0.0,
                        "Actual Rows": 1600,
                        "Actual Loops": 1,
                        "children": []
                    }
                ]
            }
        },
        "Planning Time": 1.284,
        "Execution Time": 0.65
    }

    plan_obj = Plan(sample_plan)
    print("Plan Object:", plan_obj)
    
    featurizer = PlanFeaturizer(plan_obj)
    feature_tree = featurizer.get_feature_tree()
    print("Feature Tree:", feature_tree)
    
    flat_features = featurizer.get_flat_features()
    print("Flat Features:")
    for vec in flat_features:
        print(vec)