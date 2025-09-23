import os
import pandas as pd
import re

def _read_table(path: str) -> pd.DataFrame:
    try:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _write_table(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8")

def run_add_item_ids(items_file: str, items_csv: str, ids_file: str, ids_csv: str):
    """
    确保 items_file 里有唯一的 item_id。
    - 写回 parquet
    - 生成 CSV snapshot
    - 输出 item_ids.txt 和 item_ids.csv
    """
    df = _read_table(items_file)
    if "item_id" not in df.columns:
        df["item_id"] = [f"it_{i:06d}" for i in range(len(df))]
    df["item_id"] = df["item_id"].astype(str)

    # 覆盖保存
    _write_table(df, items_file)
    _write_table(df, items_csv)

    # 写 ID 文件
    os.makedirs(os.path.dirname(ids_file), exist_ok=True)
    with open(ids_file, "w", encoding="utf-8") as f:
        for x in df["item_id"].tolist():
            f.write(str(x) + "\n")
    _write_table(pd.DataFrame({"item_id": df["item_id"]}), ids_csv)

def _normalize_text(s: str) -> str:
    if not isinstance(s, str): 
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def run_normalize(items_in: str, items_out: str):
    """
    标准化文本，生成 text_norm 列。
    优先选择 text_norm，其次 text/item_text/stem。
    """
    df = _read_table(items_in)
    # 确定源列
    for col in ["text_norm", "text", "item_text", "stem"]:
        if col in df.columns:
            src_col = col
            break
    else:
        src_col = None
        df["text"] = ""

    if src_col != "text_norm":
        src = df[src_col] if src_col else df["text"]
        df["text_norm"] = [ _normalize_text(x) for x in src ]
    else:
        df["text_norm"] = [ _normalize_text(x) for x in df["text_norm"] ]

    _write_table(df, items_out)

def run_qc(items_in: str, items_out: str):
    """
    简单质量检查：标记空文本。
    """
    df = _read_table(items_in)
    df["is_empty"] = df["text_norm"].fillna("").eq("")
    _write_table(df, items_out)
