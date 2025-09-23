import subprocess

def run_contrastive(emb_file, ids_file, items_file, out_emb, out_model=None):
    """
    emb_file: emb_base.npy
    ids_file: item_ids.txt
    items_file: parquet/csv 包含 construct_reported 列
    out_emb: 输出 emb_contrastive.npy
    out_model: 保存投影头参数
    """
    cmd = [
        "python", "src/contrastive_supcon.py",
        "--emb-npy", emb_file,
        "--ids-file", ids_file,
        "--table", items_file,
        "--item-id-col", "item_id",
        "--label-col", "construct_reported",
        "--out-npy", out_emb,
        "--epochs", "10",
        "--batch-size", "256",
        "--lr", "1e-3",
        "--out-dim", "384"
    ]
    if out_model:
        cmd.extend(["--out-model", out_model])
    subprocess.run(cmd, check=True)
