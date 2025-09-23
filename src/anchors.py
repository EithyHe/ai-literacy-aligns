# anchors.py
import os
import json
import numpy as np
import pandas as pd
from typing import Optional

def run_anchors(
    items_file: str,
    ids_file: str,
    emb_file: str,
    config_file: Optional[str],
    anchors_file: str
):
    """
    计算题项的“锚点向量”(anchors/prototypes)。

    逻辑：
    - 如果 items_file 中有 construct 列，则按构念分组取平均，得到每个构念的原型向量。
    - 如果没有 construct 列，则只输出全局质心。
    - 输出 anchors.npy (矩阵) + anchors.npy.json (索引到构念的映射)。

    参数
    ----
    items_file : str
        题项文件 (parquet/csv)，需包含 item_id。
    ids_file : str
        item_ids.txt，用于确保顺序一致。
    emb_file : str
        向量文件 (npy)，与 ids_file 顺序对应。
    config_file : str | None
        可选的配置文件（目前未使用，保留接口）。
    anchors_file : str
        输出 .npy 文件路径（会同时写 .json）。
    """
    # 加载向量
    X = np.load(emb_file)  # [N, D]

    # 加载 ids
    with open(ids_file, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]

    # 加载题项表
    try:
        items = pd.read_parquet(items_file)
    except Exception:
        items = pd.read_csv(items_file)

    if "item_id" not in items.columns:
        items["item_id"] = ids

    # 确保 item_id 对齐
    items = items.merge(pd.DataFrame({"item_id": ids}), on="item_id", how="right")

    # 是否有构念列
    if "construct" in items.columns and items["construct"].notna().any():
        groups = items["construct"].fillna("NA").astype(str)
        labels = sorted(groups.unique().tolist())

        A = []
        for g in labels:
            idx = groups[groups == g].index.values
            idx = [i for i in idx if i < X.shape[0]]
            if len(idx) == 0:
                A.append(np.zeros((X.shape[1],), dtype=float))
            else:
                A.append(X[idx].mean(axis=0))
        A = np.stack(A, axis=0)
        mapping = {i: lbl for i, lbl in enumerate(labels)}
    else:
        # 全局质心
        A = X.mean(axis=0, keepdims=True)
        mapping = {0: "global_centroid"}

    # 保存
    os.makedirs(os.path.dirname(anchors_file), exist_ok=True)
    np.save(anchors_file, A)
    with open(anchors_file + ".json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"[Anchors] Saved vectors to {anchors_file}, mapping to {anchors_file}.json")
    return anchors_file
