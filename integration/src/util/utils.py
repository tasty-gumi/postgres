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
        SELECT name, embedding, latency_s, id, plan, strategy 
        FROM public.plan_explored 
        WHERE embedding IS NOT NULL AND latency_s IS NOT NULL
        ORDER BY name, latency_s
    """,fetch=True)
    
    # 按查询名分组
    query_groups = {}
    for name, embedding_str, latency, plan_id, plan_json, strategy_json in rows:
        # 关键改进：解析字符串为float列表
        try:
            # 用json.loads解析字符串（兼容"[x1,x2,...]"格式）
            embedding = json.loads(embedding_str)
            # 验证是否为数值列表（避免解析后不是数组的情况）
            if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
                raise ValueError(f"嵌入向量解析后不是有效的数值列表: {embedding_str}")
        except Exception as e:
            # 处理解析失败的情况（如格式错误）
            print(f"解析嵌入向量失败（{name}）:{e}")
            continue  # 跳过无效数据
        
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
        query_groups[name]['strategy_jsons'].append(strategy_json)
    
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