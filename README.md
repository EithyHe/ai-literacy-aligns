# AI Literacy Aligns

## Overview

Analyze AI-literacy survey instruments: extract latent constructs via vector embeddings and PCA, and use an LLM for automatic interpretation and naming.

## Project Structure

```
ai-literacy-aligns/
├── main_pipeline.py                    # Main pipeline
├── hypergraph_visualization.py         # Hypergraph visualization utilities
├── src/                                # Source code
│   ├── preprocessing.py                # Data preprocessing
│   ├── embeddings.py                   # Vector embeddings
│   ├── sanity_rev.py                   # Reverse-keyed item check
│   ├── neighbors_pca.py                # Similarity computation
│   ├── pca_analysis.py                 # PCA analysis
│   ├── pca_selection.py                # PCA component selection
│   ├── llm_interpret.py                # LLM-based interpretation
│   ├── umap_plot.py                    # UMAP visualization
│   ├── umap_from_scores.py             # UMAP from PCA scores
│   ├── cluster.py                      # Clustering analysis
│   ├── cluster_select.py               # Cluster selection
│   ├── construct_clustering.py         # Construct-level clustering
│   ├── anchors.py                      # Anchor analysis
│   ├── contrastive.py                  # Contrastive learning
│   ├── contrastive_supcon.py           # Supervised contrastive learning
│   ├── hypergraph.py                   # Hypergraph construction
│   ├── README.md                       # Source code notes
│   └── requirements.txt                # Source dependencies
├── data/                               # Data directory
│   ├── interim/                        # Intermediate data
│   │   ├── items_master.csv            # Master items table (CSV)
│   │   └── items_master.parquet        # Master items table (Parquet)
│   ├── processed/                      # Processed artifacts
│   │   ├── emb_base.npy                # Base embeddings
│   │   ├── emb_contrastive.npy         # Contrastive embeddings
│   │   ├── item_ids.txt                # Item ID list
│   │   ├── item_ids.csv                # Item ID table
│   │   ├── pca_scores.npy              # PCA score matrix
│   │   ├── pca_scores.csv              # PCA scores (table)
│   │   ├── pca_loadings.csv            # PCA loadings
│   │   ├── pca_top_items.csv           # Representative items per PC
│   │   ├── pca_component_labels.csv    # PCA component labels
│   │   ├── pca_metrics.csv             # PCA metrics
│   │   ├── cluster_labels.csv          # Cluster labels
│   │   ├── cluster_metrics.csv         # Cluster metrics
│   │   ├── anchors.npy                 # Anchor data
│   │   ├── anchors.npy.json            # Anchor metadata
│   │   ├── hypergraph_data.json        # Hypergraph (JSON)
│   │   ├── hypergraph_nodes.csv        # Hypergraph nodes
│   │   ├── hypergraph_edges.csv        # Hypergraph edges
│   │   ├── hyper_nodes.csv             # Hypernodes
│   │   ├── hyper_edges.csv             # Hyperedges
│   │   ├── neighbors_pca.parquet       # PCA-based similarity data
│   │   ├── S_pca_topk20.npz            # Sparse similarity matrix
│   │   └── labels.parquet              # Labels
│   ├── models/                         # Model files
│   │   └── proj_head.pt                # Projection head
│   └── docs/                           # Docs & visualizations
│       ├── umap.png                    # UMAP visualization
│       ├── umap_pca.png                # UMAP based on PCA
│       ├── pca_scree.png               # PCA scree plot
│       ├── cluster_elbow.png           # Clustering elbow plot
│       ├── hypergraph_static.png       # Static hypergraph
│       ├── hypergraph_interactive.html # Interactive hypergraph
│       ├── hypergraph_explorer.html    # Hypergraph explorer
│       ├── ai_literacy_hypergraph_real.html # Real-world hypergraph
│       ├── hypergraph_visualization_summary.md # Hypergraph viz summary
│       └── revkey_sanity.md            # Reverse-key check report
├── .gitignore                          # Git ignore
├── requirements.txt                    # Project dependencies
├── push_to_github.bat                  # GitHub push script (Batch)
├── push.ps1                            # GitHub push script (PowerShell)
└── README.md                           # Project README

```

## Usage

### 1. Environmental Settings

```bash
pip install -r requirements.txt

```

### 2. Environment Variables

```bash
# Set OpenAI API key (for LLM-based interpretation)
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run the pipeline

```bash
python main_pipeline.py
```

## Pipeline

┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 1 │ 数据预处理 (Preprocessing)                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 输入   │ items_master.csv (原始题项数据)                                     │
│ 处理   │ ① 添加题项ID  ② 文本标准化  ③ 质量检查                              │
│ 输出   │ items_master.csv (清洗后), item_ids.txt, item_ids.csv             │
└─────────────────────────────────────────────────────────────────────────────┘


<img width="1501" height="455" alt="image" src="https://github.com/user-attachments/assets/a79cadde-cbea-4d9b-b486-b57815578575" />


┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 2 │ 嵌入向量生成 (Embeddings)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ 输入   │ items_master.csv (text_norm列)                                     │
│ 方法   │ OpenAI text-embedding-3-large (3072维)                             │
│ 输出   │ emb_base.npy (原始嵌入, n×3072)                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 3│ 对比学习增强 (Contrastive Learning)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 输入   │ emb_base.npy + construct信息                                       │
│ 方法   │ Supervised Contrastive Learning                                    │
│ 目的   │ 增强同construct题项的相似性，提高嵌入质量                           │
│ 输出   │ emb_contrastive.npy (增强嵌入), proj_head.pt (投影头模型)          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 4 │ 相似度网络构建 (k-NN Graph)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 输入   │ emb_contrastive.npy                                                │
│ 方法   │ k-NN算法 (k=20, 余弦相似度)                                        │
│ 输出   │ • neighbors_graph.parquet (题项相似度表)                           │
│        │ • S_graph_topk20.npz (稀疏相似度矩阵)                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 5 │ 图聚类分析 (Graph-based Clustering) ★核心步骤★                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 输入   │ emb_contrastive.npy (增强嵌入)                                     │
│ 方法   │ 测试4种图聚类方法：                                                │
│        │   ① knn_kmeans: k-NN图 + 谱嵌入 + KMeans (谱聚类变体)             │
│        │   ② similarity_spectral: 相似度图 + 谱聚类 (社区发现)             │
│        │   ③ graph_embedding: 图嵌入 + KMeans                              │
│        │   ④ dbscan: 密度聚类                                              │
│ 参数   │ • k范围: [2, 15]                                                   │
│        │ • 相似度阈值: 0.7                                                  │
│        │ • 评价指标: Silhouette Score (轮廓系数)                           │
│ 选择   │ 自动选择轮廓系数最高的方法和聚类数k                                │
│ 输出   │ • graph_metrics.csv (所有方法的评估指标)                           │
│        │ • graph_clusters.csv (最终聚类结果: item_id, cluster)             │
│        │ • graph_analysis.png (可视化：t-SNE降维+聚类结果)                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 6 │ 聚类解释 (Cluster Interpretation)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 6.1    │ 代表题项选择 (基于图中心性)                                        │
│ 输入   │ emb_contrastive.npy + graph_clusters.csv                           │
│ 方法   │ 对每个聚类，基于图中心性选择top-10代表题项：                       │
│        │   • 构建相似度图(阈值0.7)                                          │
│        │   • 计算度中心性 (与多少题项相似)                                  │
│        │   • 计算特征向量中心性 (连接的题项有多重要)                        │
│        │   • 综合得分 = 0.5×度中心性 + 0.5×特征向量中心性                  │
│ 输出   │ graph_top_items.csv (每个聚类的top-10题项+中心性得分)             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 6.2    │ LLM自动标注                                                        │
│ 输入   │ graph_top_items.csv (代表题项)                                     │
│ 方法   │ GPT-4o-mini 分析每个聚类的代表题项，生成：                         │
│        │   • label: 聚类主题标签 (2-5词)                                   │
│        │   • rationale: 一句话解释                                          │
│ 输出   │ graph_component_labels.csv (聚类标签和解释)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 6.3    │ UMAP可视化                                                         │
│ 输入   │ emb_contrastive.npy + graph_component_labels.csv                   │
│ 方法   │ UMAP降维到2D，按聚类着色                                           │
│ 输出   │ umap_graph.png (2D可视化图)                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 7 │ 传统UMAP可视化 (对比参考)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ 输入   │ emb_base.npy (原始嵌入)                                            │
│ 方法   │ UMAP降维                                                           │
│ 输出   │ umap.png (传统UMAP图，用于对比)                                   │
└─────────────────────────────────────────────────────────────────────────────┘





## Output Files
- data/processed/pca_scores.npy — PCA score matrix
- data/processed/pca_loadings.csv — PCA loadings
- data/processed/pca_top_items.csv — Representative items per component
- data/processed/pca_interpretations.csv — LLM interpretations
- data/docs/umap.png — UMAP visualization
- data/docs/revkey_sanity.md — Reverse-key check report

## To Improve
### Contrastive learning
· Incorporate respondent-level information in contrastive objectives.
· Handle both Chinese and English item wordings.
· Direct use of off-the-shelf LLM embeddings may not consistently improve results.

## Documentation
For detailed technical notes, see docs/PIPELINE_COMPARISON.md.
