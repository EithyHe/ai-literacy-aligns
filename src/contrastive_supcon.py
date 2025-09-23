#!/usr/bin/env python3
"""
Contrastive (SupCon) training on TOP of precomputed embeddings (e.g., OpenAI).
- Uses 'construct_reported' as class label.
- Trains a small projection head to reshape the space.
- Exports new embeddings aligned 1:1 with input IDs.

Requirements: torch, numpy, pandas, pyarrow (if reading parquet), scikit-learn (optional).
"""
import argparse
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def read_table_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    elif ext in [".csv", ".tsv", ".txt"]:
        sep = "," if ext == ".csv" else ("\t" if ext == ".tsv" else ",")
        return pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported table extension: {ext}")

class EmbWithLabelDataset(Dataset):
    def __init__(self, emb: np.ndarray, ids: list[str], labels: list[str] | None):
        assert emb.shape[0] == len(ids), "emb and ids length mismatch"
        self.emb = emb.astype(np.float32)
        self.ids = ids
        self.labels_raw = labels
        # Map string labels to ints; allow NaN/None -> -1
        uniq = sorted({str(x) for x in labels if (x is not None and str(x) != 'nan')})
        self.label2id = {lbl: i for i, lbl in enumerate(uniq)}
        y = []
        for x in labels:
            if (x is None) or (str(x) == 'nan'):
                y.append(-1)
            else:
                y.append(self.label2id[str(x)])
        self.y = np.array(y, dtype=np.int64)
    def __len__(self): return self.emb.shape[0]
    def __getitem__(self, idx):
        return self.emb[idx], self.y[idx]

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 384, hidden: int | None = None):
        super().__init__()
        h = hidden or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, out_dim),
        )
    def forward(self, x):
        z = self.net(x)
        z = F.normalize(z, dim=-1)
        return z

def supcon_loss(z: torch.Tensor, y: torch.Tensor, t: float = 0.07):
    """
    Supervised Contrastive loss on batch.
    - z: [B, d] L2-normalized
    - y: [B] int labels; -1 rows are treated as having no positives
    """
    B = z.size(0)
    sim = (z @ z.t()) / t                                    # [B, B]
    mask_eye = torch.eye(B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask_eye, float('-inf'))
    # positive pairs mask
    y1 = y.view(-1,1)
    pos_mask = (y1 == y1.t()) & (~mask_eye) & (y.view(-1,1) != -1)  # -1 has no positives

    # log-softmax along rows
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)       # [B, B]

    pos_counts = pos_mask.sum(dim=1)                                 # [B]
    # avoid div by zero
    denom = torch.clamp(pos_counts, min=1)
    pos_log_prob = (log_prob * pos_mask).sum(dim=1) / denom          # [B]
    # Only keep rows with at least one positive
    valid = pos_counts > 0
    if valid.any():
        loss = -(pos_log_prob[valid]).mean()
    else:
        loss = torch.tensor(0.0, device=z.device, requires_grad=True)
    return loss

@torch.no_grad()
def evaluate_cohesion(z: torch.Tensor, y: torch.Tensor):
    """Simple diagnostics: mean intra-class cosine vs inter-class cosine."""
    z = F.normalize(z, dim=-1)
    sim = z @ z.t()
    B = z.size(0)
    mask_eye = torch.eye(B, device=z.device, dtype=torch.bool)
    inter_mask = (~mask_eye).clone()
    intra_sims = []
    inter_sims = []
    y_np = y.detach().cpu().numpy()
    for i in range(B):
        same = (y_np == y_np[i]) & (y_np != -1)
        diff = (y_np != y_np[i]) & (y_np != -1) & (y_np[i] != -1)
        if same.sum() > 1:
            intra_sims.append(sim[i, torch.tensor(same, device=z.device)].mean().item())
        if diff.sum() > 0:
            inter_sims.append(sim[i, torch.tensor(diff, device=z.device)].mean().item())
    mean_intra = float(np.mean(intra_sims)) if len(intra_sims) else float('nan')
    mean_inter = float(np.mean(inter_sims)) if len(inter_sims) else float('nan')
    return mean_intra, mean_inter

def make_loader(emb: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    tens = torch.tensor(emb, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(tens, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-npy", required=True, help="Precomputed base embeddings .npy (N,d)")
    ap.add_argument("--ids-file", required=True, help="Text file with N lines, each item_id matching emb rows")
    ap.add_argument("--table", required=True, help="CSV/TSV/Parquet with columns: item_id, construct_reported")
    ap.add_argument("--item-id-col", default="item_id", help="Column name for item IDs (default: item_id)")
    ap.add_argument("--label-col", default="construct_reported", help="Column name for labels (default: construct_reported)")
    ap.add_argument("--out-npy", required=True, help="Output path for contrastive embeddings .npy")
    ap.add_argument("--out-model", default=None, help="Optional: save projection head .pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3, help="LR for projection head")
    ap.add_argument("--out-dim", type=int, default=384)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda_if_available", choices=["cpu", "cuda", "cuda_if_available"])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda_if_available":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load inputs
    emb = np.load(args.emb_npy)  # (N, d)
    with open(args.ids_file, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    assert emb.shape[0] == len(ids), "Rows of emb_npy must match lines of ids_file"

    tbl = read_table_any(args.table)
    if args.item_id_col not in tbl.columns:
        raise ValueError(f"Missing column {args.item_id_col} in table")
    if args.label_col not in tbl.columns:
        raise ValueError(f"Missing column {args.label_col} in table")

    # Align labels to ids order
    sub = tbl[[args.item_id_col, args.label_col]].copy()
    sub = sub.drop_duplicates(subset=[args.item_id_col])
    mapping = dict(zip(sub[args.item_id_col].astype(str), sub[args.label_col]))
    labels = [mapping.get(str(i), None) for i in ids]

    # Dataset build
    ds = EmbWithLabelDataset(emb, ids, labels)
    y = ds.y
    n_classes = len(ds.label2id)
    print(f"[Info] Items: {len(ids)} | Dim: {emb.shape[1]} | Classes (non-NaN): {n_classes}")
    if n_classes == 0:
        print("[Warn] No labels found in construct_reported. Training will likely do nothing.")

    # Train / val split (stratified where possible)
    idx_all = np.arange(len(ids))
    labeled_idx = np.where(y != -1)[0]
    unlabeled_idx = np.where(y == -1)[0]
    rng.shuffle(labeled_idx)
    n_val = max(1, int(0.1 * len(labeled_idx))) if len(labeled_idx) > 10 else 0
    val_idx = labeled_idx[:n_val] if n_val > 0 else np.array([], dtype=int)
    train_idx = np.concatenate([labeled_idx[n_val:], unlabeled_idx])  # allow unlabeled in train (ignored by loss)

    train_loader = make_loader(emb[train_idx], y[train_idx], args.batch_size, shuffle=True)
    val_loader = make_loader(emb[val_idx], y[val_idx], batch_size=min(args.batch_size, len(val_idx) or 1), shuffle=False) if len(val_idx) > 0 else None

    in_dim = emb.shape[1]
    head = ProjectionHead(in_dim, out_dim=args.out_dim, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)

    best_val = float("-inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        n_steps = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            zb = head(xb)
            loss = supcon_loss(zb, yb, t=args.temp)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_steps += 1
        avg_loss = total_loss / max(1, n_steps)

        # Validation: cohesion score (intra - inter)
        if val_loader is not None:
            head.eval()
            with torch.no_grad():
                all_z, all_y = [], []
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    zb = head(xb)
                    all_z.append(zb)
                    all_y.append(yb.to(device))
                Z = torch.cat(all_z, dim=0)
                Y = torch.cat(all_y, dim=0)
                intra, inter = evaluate_cohesion(Z, Y)
                score = (intra - inter) if (not math.isnan(intra) and not math.isnan(inter)) else float("-inf")
        else:
            intra = inter = float("nan")
            score = -avg_loss  # fallback

        print(f"[Epoch {epoch:02d}] loss={avg_loss:.4f} | intra={intra:.4f} | inter={inter:.4f} | score={score:.4f}")

        if score > best_val:
            best_val = score
            best_state = head.state_dict()

    # Load best
    if best_state is not None:
        head.load_state_dict(best_state)

    # Export all transformed embeddings aligned with input order
    head.eval()
    with torch.no_grad():
        Z = []
        bs = 4096
        for i in range(0, emb.shape[0], bs):
            xb = torch.tensor(emb[i:i+bs], dtype=torch.float32, device=device)
            zb = head(xb).cpu().numpy()
            Z.append(zb)
        Z = np.vstack(Z)
    out_dir = os.path.dirname(args.out_npy)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.out_npy, Z)
    print(f"[Done] Saved contrastive embeddings -> {args.out_npy}  shape={Z.shape}")
    if args.out_model:
        mdir = os.path.dirname(args.out_model)
        if mdir:
            os.makedirs(mdir, exist_ok=True)
        torch.save({"state_dict": head.state_dict(), "in_dim": in_dim, "out_dim": args.out_dim, "hidden": args.hidden}, args.out_model)
        print(f"[Done] Saved projection head -> {args.out_model}")

if __name__ == "__main__":
    main()
