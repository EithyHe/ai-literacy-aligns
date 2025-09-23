import os
import pandas as pd

def _read_table(path: str) -> pd.DataFrame:
    try:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def run_hypergraph(
    items_file: str,
    cluster_labels_csv: str,
    out_nodes_csv: str,
    out_edges_csv: str,
    construct_col: str = "construct_reported",
):
    """
    Build a tripartite hypergraph linking:
      - items (item_id)
      - constructs (from items_file[construct_col])
      - clusters (from cluster_labels_csv['cluster'])

    Outputs two CSVs:
      - nodes: id, label, type
      - edges: src, dst, rel, weight
        rel in {item_construct, item_cluster, construct_cluster}
        construct_cluster weight is the number of items connecting the pair
    """
    items = _read_table(items_file)
    if "item_id" not in items.columns:
        items["item_id"] = [f"it_{i:06d}" for i in range(len(items))]
    items["item_id"] = items["item_id"].astype(str)

    if construct_col not in items.columns:
        items[construct_col] = ""

    labs = pd.read_csv(cluster_labels_csv)
    # cluster_select.run_cluster_apply writes columns: id, cluster
    id_col = "id" if "id" in labs.columns else ("item_id" if "item_id" in labs.columns else labs.columns[0])
    labs[id_col] = labs[id_col].astype(str)

    df = items[["item_id", construct_col]].merge(
        labs[[id_col, "cluster"]], left_on="item_id", right_on=id_col, how="inner"
    )
    df.drop(columns=[id_col], inplace=True, errors="ignore")

    # Nodes
    item_nodes = pd.DataFrame({
        "id": df["item_id"].astype(str),
        "label": df["item_id"].astype(str),
        "type": "item",
    }).drop_duplicates("id")

    construct_nodes = pd.DataFrame({
        "id": df[construct_col].fillna("").astype(str),
        "label": df[construct_col].fillna("").astype(str),
        "type": "construct",
    }).drop_duplicates("id")

    cluster_nodes = pd.DataFrame({
        "id": df["cluster"].astype(str).map(lambda x: f"cluster_{x}"),
        "label": df["cluster"].astype(str).map(lambda x: f"Cluster {x}"),
        "type": "cluster",
    }).drop_duplicates("id")

    nodes = pd.concat([item_nodes, construct_nodes, cluster_nodes], ignore_index=True)

    # Edges (simple graph view of hypergraph incidence)
    e_ic = pd.DataFrame({
        "src": df["item_id"],
        "dst": df[construct_col].fillna("").astype(str),
        "rel": "item_construct",
        "weight": 1.0,
    })

    e_il = pd.DataFrame({
        "src": df["item_id"],
        "dst": df["cluster"].astype(str).map(lambda x: f"cluster_{x}"),
        "rel": "item_cluster",
        "weight": 1.0,
    })

    # Aggregate construct -> cluster edges with counts
    cc = (
        df.assign(cluster_id=df["cluster"].astype(str).map(lambda x: f"cluster_{x}"))
          .groupby([construct_col, "cluster_id"], dropna=False)
          .size()
          .reset_index(name="weight")
    )
    e_cl = pd.DataFrame({
        "src": cc[construct_col].fillna("").astype(str),
        "dst": cc["cluster_id"].astype(str),
        "rel": "construct_cluster",
        "weight": cc["weight"].astype(float),
    })

    edges = pd.concat([e_ic, e_il, e_cl], ignore_index=True)

    _ensure_parent(out_nodes_csv)
    _ensure_parent(out_edges_csv)
    nodes.to_csv(out_nodes_csv, index=False, encoding="utf-8")
    edges.to_csv(out_edges_csv, index=False, encoding="utf-8")

    return {"nodes_csv": out_nodes_csv, "edges_csv": out_edges_csv}


