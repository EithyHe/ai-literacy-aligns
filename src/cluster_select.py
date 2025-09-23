# src/cluster_select.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt

PC_PREFIX = "PC"   # 你的pca_scores.csv里列名形如 PC1, PC2, ...

def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def _load_pc_matrix(pca_scores_csv: str) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(pca_scores_csv)
    # 取出ID和PC列
    id_col = "id" if "id" in df.columns else df.columns[0]
    pc_cols = [c for c in df.columns if str(c).startswith(PC_PREFIX)]
    if not pc_cols:
        raise ValueError(f"No PC columns found in {pca_scores_csv}. Expect columns like 'PC1','PC2',...")
    X = df[pc_cols].to_numpy()
    ids = df[id_col].astype(str).tolist()
    return X, ids

def run_k_selection(
    pca_scores_csv: str,
    out_metrics_csv: str,
    out_elbow_png: str,
    k_min: int = 2,
    k_max: int = 15,
    random_state: int = 42,
) -> dict:
    """
    在 PCA scores 上用 KMeans 评估 k，输出指标表和肘部图。
    返回 {'best_k': int, 'metrics_csv': ..., 'elbow_png': ...}
    """
    X, _ = _load_pc_matrix(pca_scores_csv)
    ks, inertia, sils, chs = [], [], [], []

    for k in range(k_min, k_max + 1):
        # Use integer n_init for broad scikit-learn compatibility
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        ks.append(k)
        inertia.append(km.inertia_)  # SSE/惯性，用于肘部法
        # Silhouette和CH在k=1时无定义，这里k>=2已满足
        sils.append(silhouette_score(X, labels))
        chs.append(calinski_harabasz_score(X, labels))

    # 保存指标
    _ensure_parent(out_metrics_csv)
    metrics_df = pd.DataFrame({
        "k": ks,
        "inertia": inertia,
        "silhouette": sils,
        "calinski_harabasz": chs,
    })
    metrics_df.to_csv(out_metrics_csv, index=False)

    # 画肘部 + 轮廓/CH 对比
    _ensure_parent(out_elbow_png)
    plt.figure(figsize=(9,6))
    ax1 = plt.gca()
    ax1.plot(ks, inertia, marker="o", label="Inertia (SSE)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia (lower is better)")
    ax2 = ax1.twinx()
    ax2.plot(ks, sils, marker="s", linestyle="--", label="Silhouette (higher better)")
    ax2.plot(ks, chs, marker="^", linestyle=":", label="Calinski-Harabasz (higher better)")
    ax2.set_ylabel("Silhouette / CH")

    # 合并图例
    lines, labels = [], []
    for ax in (ax1, ax2):
        L = ax.get_lines()
        lines += L
        labels += [l.get_label() for l in L]
    plt.title("K selection on PCA scores")
    plt.legend(lines, labels, loc="best")
    plt.tight_layout()
    plt.savefig(out_elbow_png, dpi=160)
    plt.close()

    # 选择最佳k：先用 silhouette 最大；如并列，用 CH 最大；再并列取较小k
    max_sil = max(sils)
    cand = [i for i, s in enumerate(sils) if s == max_sil]
    if len(cand) > 1:
        best_idx = max(cand, key=lambda i: chs[i])
        # 若还想更保守，可再：best_idx = min([i for i in cand if chs[i]==chs[best_idx]], key=lambda i: ks[i])
    else:
        best_idx = cand[0]
    best_k = ks[best_idx]

    return {"best_k": int(best_k), "metrics_csv": out_metrics_csv, "elbow_png": out_elbow_png}

def run_cluster_apply(
    pca_scores_csv: str,
    out_labels_csv: str,
    k: int,
    random_state: int = 42,
) -> str:
    """
    用选择好的 k 在 PCA scores 上做 KMeans，输出 item→cluster 的标签表。
    """
    X, ids = _load_pc_matrix(pca_scores_csv)
    # Use integer n_init for broad scikit-learn compatibility
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    df_out = pd.DataFrame({"id": ids, "cluster": labels})
    _ensure_parent(out_labels_csv)
    df_out.to_csv(out_labels_csv, index=False)
    return out_labels_csv
