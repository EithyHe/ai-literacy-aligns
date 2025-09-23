# construct_clustering.py
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def run_construct_clustering(
    items_file: str,
    out_dir: str,
    emb_file: str = "data/processed/emb_base.npy",
    n_clusters: int = 20,
):
    """
    在嵌入空间执行 KMeans，导出粗粒度的“构念簇”。

    参数
    ----
    items_file : str
        题项表（parquet 或 csv），最好包含 item_id（若无会自动生成）。
    out_dir : str
        输出目录，文件名固定为 construct_clusters.parquet。
    emb_file : str, default="data/processed/emb_base.npy"
        与 items_file 对应顺序的一组向量 (N × D)。
    n_clusters : int, default=20
        期望聚类数；函数会基于样本量做下限保护。

    输出
    ----
    out_path : str
        生成的 parquet 文件路径（两列：item_id, cluster）。
    """
    # 载入向量
    X = np.load(emb_file)  # [N, D]

    # 载入题项表（容错 parquet/csv）
    try:
        items = pd.read_parquet(items_file)
    except Exception:
        items = pd.read_csv(items_file)

    # 若无 item_id，则按行号生成
    if "item_id" not in items.columns:
        items["item_id"] = [f"it_{i:06d}" for i in range(len(items))]

    # 基于样本量的聚类数保护：至少 2 类，最多不超过样本量/20
    k_max_data = max(2, len(items) // 20)
    k = min(max(2, n_clusters), k_max_data)

    # 运行 KMeans
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X)

    # 输出结果
    out = pd.DataFrame({
        "item_id": items["item_id"].astype(str),
        "cluster": labels
    })

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "construct_clusters.parquet")
    out.to_parquet(out_path, index=False)

    return out_path
