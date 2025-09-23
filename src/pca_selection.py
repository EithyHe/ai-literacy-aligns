# src/pca_selection.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Tuple, Dict, List

def _ensure_parent(path: str):
    """确保父目录存在"""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def run_pca_selection(
    emb_file: str,
    out_metrics_csv: str,
    out_scree_png: str,
    max_components: int = None,
    variance_threshold: float = 0.95,
    min_components: int = 2,
    random_state: int = 42
) -> Dict:
    """
    智能选择主成分数量，基于多种标准：
    1. 累积方差解释率
    2. 肘部法则（特征值下降）
    3. 聚类质量（轮廓系数）
    4. 特征值 > 1 规则
    
    参数:
    - emb_file: 嵌入向量文件
    - out_metrics_csv: 输出指标CSV
    - out_scree_png: 碎石图PNG
    - max_components: 最大主成分数（默认为min(样本数-1, 特征数)）
    - variance_threshold: 累积方差阈值（默认0.95）
    - min_components: 最小主成分数（默认2）
    - random_state: 随机种子
    
    返回:
    - dict: 包含最佳主成分数和相关指标
    """
    
    # 加载数据
    X = np.load(emb_file)
    n_samples, n_features = X.shape
    
    # 设置最大主成分数
    if max_components is None:
        max_components = min(n_samples - 1, n_features, 50)  # 保守限制
    
    max_components = min(max_components, n_samples - 1, n_features)
    
    print(f"数据维度: {n_samples} 样本 × {n_features} 特征")
    print(f"测试主成分数范围: {min_components} - {max_components}")
    
    # 计算所有可能的主成分
    pca_full = PCA(n_components=max_components, svd_solver="auto", random_state=random_state)
    pca_full.fit(X)
    
    # 提取指标
    explained_variance_ratio = pca_full.explained_variance_ratio_
    explained_variance = pca_full.explained_variance_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # 计算不同主成分数的聚类质量
    silhouette_scores = []
    inertias = []
    
    for n_comp in range(min_components, max_components + 1):
        # 降维
        pca = PCA(n_components=n_comp, svd_solver="auto", random_state=random_state)
        X_reduced = pca.fit_transform(X)
        
        # 计算聚类质量（使用KMeans）
        from sklearn.cluster import KMeans
        if n_comp >= 2:  # 至少需要2维才能聚类
            # 尝试不同的聚类数
            best_silhouette = -1
            best_inertia = float('inf')
            
            for k in range(2, min(6, n_samples // 2 + 1)):  # 尝试2-5个聚类
                try:
                    # scikit-learn >=1.3 允许字符串，但为兼容性统一用整数
                    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
                    labels = kmeans.fit_predict(X_reduced)
                    
                    if len(set(labels)) > 1:  # 确保有多个聚类
                        sil_score = silhouette_score(X_reduced, labels)
                        best_silhouette = max(best_silhouette, sil_score)
                        best_inertia = min(best_inertia, kmeans.inertia_)
                except:
                    continue
            
            silhouette_scores.append(best_silhouette if best_silhouette > -1 else 0)
            inertias.append(best_inertia if best_inertia < float('inf') else 0)
        else:
            silhouette_scores.append(0)
            inertias.append(0)
    
    # 选择最佳主成分数
    n_components_range = list(range(min_components, max_components + 1))
    
    # 1. 累积方差阈值
    variance_cutoff = next((i for i, cumvar in enumerate(cumulative_variance) 
                           if cumvar >= variance_threshold), len(cumulative_variance) - 1) + 1
    variance_cutoff = min(variance_cutoff, max_components)
    
    # 2. 特征值 > 1 规则（Kaiser准则）
    kaiser_cutoff = sum(explained_variance > 1)
    kaiser_cutoff = min(kaiser_cutoff, max_components)
    
    # 3. 肘部法则（特征值下降）
    # 计算特征值的二阶导数，找到"肘部"
    if len(explained_variance) > 3:
        second_derivative = np.diff(explained_variance, 2)
        elbow_idx = np.argmax(second_derivative) + 2  # +2 因为二阶导数
        elbow_cutoff = min(elbow_idx + 1, max_components)
    else:
        elbow_cutoff = min_components
    
    # 4. 最佳聚类质量
    if silhouette_scores:
        best_cluster_idx = np.argmax(silhouette_scores)
        cluster_cutoff = n_components_range[best_cluster_idx]
    else:
        cluster_cutoff = min_components
    
    # 综合选择：取各种方法的平均值，但不超过合理范围
    candidates = [variance_cutoff, kaiser_cutoff, elbow_cutoff, cluster_cutoff]
    candidates = [c for c in candidates if min_components <= c <= max_components]
    
    if candidates:
        best_n_components = int(np.median(candidates))  # 使用中位数更稳健
    else:
        best_n_components = min_components
    
    # 确保在合理范围内
    best_n_components = max(min_components, min(best_n_components, max_components))
    
    # 保存指标
    metrics_df = pd.DataFrame({
        'n_components': n_components_range,
        'explained_variance_ratio': explained_variance_ratio[:len(n_components_range)],
        'cumulative_variance': cumulative_variance[:len(n_components_range)],
        'silhouette_score': silhouette_scores,
        'inertia': inertias
    })
    
    _ensure_parent(out_metrics_csv)
    metrics_df.to_csv(out_metrics_csv, index=False)
    
    # 绘制碎石图
    _ensure_parent(out_scree_png)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # 碎石图
    ax1.plot(n_components_range, explained_variance_ratio[:len(n_components_range)], 'bo-')
    ax1.axvline(x=best_n_components, color='r', linestyle='--', label=f'Selected: {best_n_components}')
    ax1.set_xlabel('主成分数')
    ax1.set_ylabel('解释方差比')
    ax1.set_title('碎石图 - 解释方差比')
    ax1.legend()
    ax1.grid(True)
    
    # 累积方差
    ax2.plot(n_components_range, cumulative_variance[:len(n_components_range)], 'go-')
    ax2.axhline(y=variance_threshold, color='r', linestyle='--', label=f'阈值: {variance_threshold}')
    ax2.axvline(x=best_n_components, color='r', linestyle='--', label=f'Selected: {best_n_components}')
    ax2.set_xlabel('主成分数')
    ax2.set_ylabel('累积解释方差')
    ax2.set_title('累积解释方差')
    ax2.legend()
    ax2.grid(True)
    
    # 聚类质量
    if silhouette_scores:
        ax3.plot(n_components_range, silhouette_scores, 'mo-', label='轮廓系数')
        ax3.axvline(x=best_n_components, color='r', linestyle='--', label=f'Selected: {best_n_components}')
        ax3.set_xlabel('主成分数')
        ax3.set_ylabel('轮廓系数')
        ax3.set_title('聚类质量 (轮廓系数)')
        ax3.legend()
        ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(out_scree_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 返回结果
    result = {
        'best_n_components': best_n_components,
        'variance_cutoff': variance_cutoff,
        'kaiser_cutoff': kaiser_cutoff,
        'elbow_cutoff': elbow_cutoff,
        'cluster_cutoff': cluster_cutoff,
        'explained_variance_ratio': float(cumulative_variance[best_n_components - 1]),
        'metrics_csv': out_metrics_csv,
        'scree_png': out_scree_png
    }
    
    print(f"主成分选择结果:")
    print(f"  最佳主成分数: {best_n_components}")
    print(f"  累积方差解释: {result['explained_variance_ratio']:.3f}")
    print(f"  方差阈值法: {variance_cutoff}")
    print(f"  Kaiser准则: {kaiser_cutoff}")
    print(f"  肘部法则: {elbow_cutoff}")
    print(f"  聚类质量法: {cluster_cutoff}")
    
    return result
