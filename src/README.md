# Path B: Supervised Contrastive on top of Precomputed Embeddings

本脚本在**固定的预训练向量**（如 OpenAI `text-embedding-3-large`）之上训练一个**小型投影头（Projection Head）**，用 **Supervised Contrastive (SupCon)** 损失把空间“整形”。标签字段使用你的数据表中的 **`construct_reported`**。

## 文件
- `contrastive_supcon.py`：训练与导出主脚本
- `requirements.txt`：依赖

## 安装
```bash
pip install -r requirements.txt
```

## 输入约定
- `--emb-npy`: 形如 `(N, d)` 的 `.npy`，为每个题项/文本的基向量（与 `ids` 一一对应）
- `--ids-file`: 文本文件，N 行，每行一个 `item_id`，顺序与 `emb-npy` 完全一致
- `--table`: 含 `item_id` 与 `construct_reported` 的表，可为 `.parquet` / `.csv` / `.tsv`
  - 可通过 `--item-id-col` 与 `--label-col` 自定义列名（默认 `item_id`, `construct_reported`）

## 训练 + 导出
示例命令：
```bash
python contrastive_supcon.py \
  --emb-npy data/processed/emb_base.npy \
  --ids-file data/processed/item_ids.txt \
  --table data/interim/items_master.parquet \
  --item-id-col item_id \
  --label-col construct_reported \
  --out-npy data/processed/emb_contrastive.npy \
  --out-model data/models/proj_head.pt \
  --epochs 10 \
  --batch-size 256 \
  --lr 1e-3 \
  --out-dim 384 \
  --temp 0.07
```

训练过程中会输出：
- `loss`：SupCon 训练损失（批内有正样本时才有效）
- `intra / inter`：验证集上“类内平均相似度 / 类间平均相似度”
- `score`：`intra - inter`，越大越好（没有验证集时退化为 `-loss`）

导出：
- `emb_contrastive.npy`：与原 `ids` 顺序一致的新向量，可直接替换到你的 PCA / KNN / 聚类 / 超图流程
- `proj_head.pt`：投影头权重（可复用）

## 训练细节
- 未标注或**批内无正样本**的样本在 SupCon 损失中会被自动忽略，不会破坏训练
- 你可以先仅用**高置信度**的 `construct_reported` 训练，随后用新向量做 KNN 伪标签再迭代一轮
- 建议：`batch-size ≥ 128`；`temp` 取 `0.05~0.2`；`out-dim` 可与下游需要匹配（如 256/384/512）

## 常见问题
1) **报错 “Missing column item_id/construct_reported”**  
   用 `--item-id-col` 或 `--label-col` 指定真实列名。

2) **emb_npy 与 ids 对不上**  
   确保 `item_ids.txt` 的顺序即为 `emb_base.npy` 的行顺序；脚本按 `ids` 与表做对齐。

3) **没有验证集**  
   若可用标签很少，脚本会跳过验证指标，此时 `score` 使用 `-loss` 进行早停选择。

4) **如何只推理（不训练）？**  
   直接加载 `proj_head.pt` 写个小脚本对任意新 `emb.npy` 做一次前向、保存即可。

祝你在构念聚合、跨量表对齐和超图建模中取得更紧致、更可分的表示空间！
