# umap_plot.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

def run_umap(
    emb_file: str,
    labels_file: str | None,
    out_png: str,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 42,
    point_size: int = 6,
    alpha: float = 0.75,
    title: str | None = None,
):
    """
    在嵌入空间上执行 UMAP 并输出为 PNG 图片。
    - 若提供 labels_file（parquet/csv），将尝试按 `label` 列进行着色。
    - 若未提供或读取失败，则使用单色散点。

    参数
    ----
    emb_file : str
        N×D 的向量文件（.npy）。
    labels_file : str | None
        包含聚类或分组标签的文件（parquet 或 csv），需包含列 `label`。可为 None。
    out_png : str
        输出图片路径。
    n_neighbors : int
        UMAP 的近邻数，控制局部/全局结构权衡。
    min_dist : float
        UMAP 的最小距离，控制点簇的紧凑程度。
    random_state : int
        随机种子，保证可复现。
    point_size : int
        散点大小。
    alpha : float
        散点透明度。
    title : str | None
        自定义图标题；若为 None，则自动设置。
    """
    # 载入嵌入
    X = np.load(emb_file)  # [N, D]

    # UMAP 降维
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    Y = reducer.fit_transform(X)  # [N, 2]

    # 读取标签（可选）
    colors = None
    legend_note = ""
    if labels_file:
        try:
            if labels_file.endswith(".parquet"):
                lab = pd.read_parquet(labels_file)
            else:
                lab = pd.read_csv(labels_file)
            if "label" in lab.columns and len(lab) >= len(Y):
                colors = lab["label"].values[: len(Y)]
                legend_note = " (colored by labels)"
        except Exception:
            colors = None
            legend_note = " (labels not available)"

    # 绘图
    plt.figure(figsize=(8, 6), dpi=140)
    plt.scatter(Y[:, 0], Y[:, 1], s=point_size, alpha=alpha, c=colors)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    if title is None:
        title = f"UMAP on embeddings{legend_note}"
    plt.title(title)
    plt.tight_layout()

    # 保存
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    plt.close()

    print(f"[UMAP] Saved to {out_png}")
    return out_png
