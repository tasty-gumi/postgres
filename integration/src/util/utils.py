import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
from typing import List, Dict, Tuple
from pg_interactor import Postgres
import numpy as np


def analyze_feature_importance(embeddings, latencies):
    """分析特征重要性"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    
    # 使用随机森林评估特征重要性
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(embeddings, latencies)
    
    # 特征重要性
    feature_importance = rf.feature_importances_
    
    # 排列重要性
    perm_importance = permutation_importance(
        rf, embeddings, latencies, n_repeats=10, random_state=42
    )
    
    return {
        'feature_importance': feature_importance,
        'permutation_importance': perm_importance.importances_mean,
        'permutation_std': perm_importance.importances_std
    }

def evaluate_feature_discrimination(embeddings, query_names):
    """评估特征区分度"""
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder
    
    # 将查询名转换为数字标签
    le = LabelEncoder()
    labels = le.fit_transform(query_names)
    
    # 计算轮廓系数（衡量同一查询的计划特征相似度）
    silhouette = silhouette_score(embeddings, labels)
    
    # 计算类内和类间距离
    unique_labels = np.unique(labels)
    intra_distances = []
    inter_distances = []
    
    for label in unique_labels:
        mask = labels == label
        group_embeddings = embeddings[mask]
        
        if len(group_embeddings) > 1:
            # 类内距离
            from scipy.spatial.distance import pdist
            intra_dist = np.mean(pdist(group_embeddings))
            intra_distances.append(intra_dist)
        
        # 类间距离
        other_embeddings = embeddings[~mask]
        if len(other_embeddings) > 0:
            from scipy.spatial.distance import cdist
            inter_dist = np.mean(cdist(group_embeddings, other_embeddings))
            inter_distances.append(inter_dist)
    
    return {
        'silhouette_score': silhouette,
        'avg_intra_distance': np.mean(intra_distances),
        'avg_inter_distance': np.mean(inter_distances),
        'discrimination_ratio': np.mean(inter_distances) / np.mean(intra_distances)
    }

def process_query_groups(query_groups, group_size=20, pad_latency=600.0):
    """
    处理query_groups，按每个查询的计划构建训练组（每组20个）
    不足20个的用空字典{}和pad_latency填充
    
    Args:
        query_groups: 原始查询组列表，每个元素含'query_name'、'plans'、'latencies'等字段
        group_size: 每组计划数量（默认为20）
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

def load_data_from_db(db_config):
    """从数据库加载数据并按查询名分组"""
    pg = Postgres()

    pg.setup(
        dbname=db_config["dbname"],
        host=db_config["host"],
        user=db_config["user"],
        password=db_config["password"],
        port=db_config["port"]
    )
    
    # 查询所有数据
    rows = pg.execute("""
        SELECT name, latency_ms, plan
        FROM public.explored_plans 
        WHERE latency_ms IS NOT NULL
        ORDER BY name, latency_ms;
    """,fetch=True)
    
    # 按查询名分组
    query_groups = {}
    for name, latency, plan_json in rows:        
        if name not in query_groups:
            query_groups[name] = {
                'plans': [],
                'latencies': [],
                'strategy_jsons': []
            }
        
        # 存储解析后的数值列表（而非原始字符串）
        query_groups[name]['plans'].append(plan_json)
        # query_groups[name]['embeddings'].append(embedding)  # 数值列表
        query_groups[name]['latencies'].append(float(latency))
    
    # 转换为列表格式
    grouped_data = []
    for name, group in query_groups.items():
        if len(group['plans']) >= 2:  # 只保留有至少2个计划的查询
            grouped_data.append({
                'query_name': name,
                'plans': group['plans'],
                # 'embeddings': group['embeddings'],  # 此时是数值列表的列表
                'latencies': group['latencies'],
            })
    
    return grouped_data