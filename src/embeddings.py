import os
import numpy as np
import pandas as pd
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


def _read_table(path: str) -> pd.DataFrame:
    try:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def run_embed(
    items_file: str,
    text_col: str,
    emb_file: str,
    ids_file: Optional[str] = None,
    model: str = "text-embedding-3-large",
    batch_size: int = 100
):
    """
    Use OpenAI API to generate embeddings.
    Saves: emb_file (npy) and ids_file (txt) if provided.
    """
    df = _read_table(items_file)
    assert text_col in df.columns, f"Missing text column: {text_col}"
    if "item_id" not in df.columns:
        df["item_id"] = [f"it_{i:06d}" for i in range(len(df))]

    texts = df[text_col].fillna("").astype(str).tolist()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_API_BASE"))

    embeddings = []
    # batching to avoid rate limit
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        embs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        embeddings.extend(embs)
    X = np.vstack(embeddings)

    os.makedirs(os.path.dirname(emb_file), exist_ok=True)
    np.save(emb_file, X)

    if ids_file:
        os.makedirs(os.path.dirname(ids_file), exist_ok=True)
        with open(ids_file, "w", encoding="utf-8") as f:
            for x in df["item_id"].astype(str).tolist():
                f.write(x + "\n")