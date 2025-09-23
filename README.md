# AI Literacy Aligns

## Overview

Analyze AI-literacy survey instruments: extract latent constructs via vector embeddings and PCA, and use an LLM for automatic interpretation and naming.

## 项目结构

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

**Data preprocessing:** add IDs, normalize text, quality control.
**Contrastive learning:** empirically verified that items within the same dimension are closer in semantic space, while items from different dimensions are farther apart.
**Embeddings:** generate 4096-dim vectors with Llama3-8B.
**Reverse-key check:** validate the consistency of reverse-keyed items.
**Similarity computation:** compute pairwise cosine similarity among items.
**PCA analysis:** extract latent constructs.
**LLM interpretation:** use GPT-4o to automatically name and explain constructs.
**Visualization:** generate UMAP plots.

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
