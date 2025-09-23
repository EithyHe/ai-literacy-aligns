import os
import numpy as np
import pandas as pd

def run_sanity_rev(ids_file: str, emb_file: str, items_file: str, out_md: str):
    """
    反向题 / 异常项 Sanity 检查。
    思路：
      1. 计算所有题项向量的全局质心。
      2. 按余弦相似度排序，找出最不像质心的题项（可能是反向题/异常项）。
      3. 同时列出最像质心的题项（核对主题一致性）。
      4. 输出为 Markdown 文件，方便人工检查。

    参数:
      ids_file   : data/processed/item_ids.txt
      emb_file   : data/processed/emb_base.npy
      items_file : data/interim/items_master.parquet/csv
      out_md     : 输出报告路径 (Markdown)
    """
    # 读 ID 列表
    with open(ids_file, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]

    # 读向量
    X = np.load(emb_file)  # [N, D]

    # 读题项文本
    try:
        items = pd.read_parquet(items_file)
    except Exception:
        items = pd.read_csv(items_file)
    if "item_id" not in items.columns:
        items["item_id"] = ids

    # 选择文本列
    text_col = None
    for c in ["text_norm", "text", "item_text", "stem"]:
        if c in items.columns:
            text_col = c
            break
    if text_col is None:
        items[text_col := "text"] = ""

    # === 计算到质心的余弦相似度 ===
    mu = X.mean(axis=0, keepdims=True)  # [1, D]

    def cosine(a, b):
        an = np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
        bn = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
        return (a @ b.T) / (an * bn.T)

    cs = cosine(X, mu).ravel()

    # 最不像的前 20
    idx_low = np.argsort(cs)[:20]
    # 最像的前 20
    idx_high = np.argsort(-cs)[:20]

    # === 生成 Markdown 报告 ===
    def rows_to_md(indices, title):
        lines = [
            f"### {title}",
            "",
            "| rank | item_id | cos_to_centroid | text |",
            "|---:|---|---:|---|",
        ]
        for r, i in enumerate(indices, 1):
            iid = ids[i] if i < len(ids) else str(i)
            row = items[items["item_id"] == iid]
            txt = str(row[text_col].values[0]) if not row.empty else ""
            lines.append(f"| {r} | {iid} | {cs[i]:.4f} | {txt[:120]} |")
        return "\n".join(lines)

    md = [
        "# Reverse-key / Outlier Sanity Report",
        "",
        rows_to_md(idx_low, "Lowest cosine (potential reverse/odd items)"),
        "",
        rows_to_md(idx_high, "Highest cosine (on-theme items)"),
    ]

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n\n".join(md))

    print(f"[Sanity] Report written to {out_md}")
    return out_md
