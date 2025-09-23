# AI Literacy Aligns - 人工智能素养评估量表对齐

## 项目概述

用于分析AI素养评估量表，通过向量嵌入和PCA分析提取潜在构念，并使用LLM进行自动解释和命名。

## 项目结构

```
ai-literacy-aligns/
├── main_pipeline.py                    # 主pipeline文件
├── hypergraph_visualization.py        # 超图可视化工具
├── src/                               # 源代码目录
│   ├── preprocessing.py               # 数据预处理
│   ├── embeddings.py                  # 向量嵌入
│   ├── sanity_rev.py                  # 反向题检查
│   ├── neighbors_pca.py               # 相似度计算
│   ├── pca_analysis.py                # PCA分析
│   ├── pca_selection.py               # PCA组件选择
│   ├── llm_interpret.py               # LLM解释
│   ├── umap_plot.py                   # UMAP可视化
│   ├── umap_from_scores.py            # 基于PCA得分的UMAP
│   ├── cluster.py                     # 聚类分析
│   ├── cluster_select.py              # 聚类选择
│   ├── construct_clustering.py        # 构念聚类
│   ├── anchors.py                     # 锚点分析
│   ├── contrastive.py                 # 对比学习
│   ├── contrastive_supcon.py          # 监督对比学习
│   ├── hypergraph.py                  # 超图构建
│   ├── README.md                      # 源代码说明
│   └── requirements.txt               # 源代码依赖
├── data/                              # 数据目录
│   ├── interim/                       # 中间数据
│   │   ├── items_master.csv           # 主数据表(CSV格式)
│   │   └── items_master.parquet       # 主数据表(Parquet格式)
│   ├── processed/                     # 处理后的数据
│   │   ├── emb_base.npy               # 基础嵌入向量
│   │   ├── emb_contrastive.npy        # 对比学习后的向量
│   │   ├── item_ids.txt               # 题项ID列表
│   │   ├── item_ids.csv               # 题项ID表格
│   │   ├── pca_scores.npy             # PCA得分矩阵
│   │   ├── pca_scores.csv             # PCA得分表格
│   │   ├── pca_loadings.csv           # PCA载荷
│   │   ├── pca_top_items.csv          # PCA代表性题项
│   │   ├── pca_component_labels.csv   # PCA组件标签
│   │   ├── pca_metrics.csv            # PCA分析指标
│   │   ├── cluster_labels.csv         # 聚类标签
│   │   ├── cluster_metrics.csv        # 聚类指标
│   │   ├── anchors.npy                # 锚点数据
│   │   ├── anchors.npy.json           # 锚点元数据
│   │   ├── hypergraph_data.json       # 超图JSON数据
│   │   ├── hypergraph_nodes.csv       # 超图节点
│   │   ├── hypergraph_edges.csv       # 超图边
│   │   ├── hyper_nodes.csv            # 超节点数据
│   │   ├── hyper_edges.csv            # 超边数据
│   │   ├── neighbors_pca.parquet      # PCA相似度数据
│   │   ├── S_pca_topk20.npz           # 稀疏相似度矩阵
│   │   └── labels.parquet             # 标签数据
│   ├── models/                        # 模型文件
│   │   └── proj_head.pt               # 投影头模型
│   └── docs/                          # 数据文档和可视化
│       ├── umap.png                   # UMAP降维可视化
│       ├── umap_pca.png               # 基于PCA的UMAP
│       ├── pca_scree.png              # PCA碎石图
│       ├── cluster_elbow.png          # 聚类肘部图
│       ├── hypergraph_static.png      # 静态超图
│       ├── hypergraph_interactive.html # 交互式超图
│       ├── hypergraph_explorer.html   # 超图探索器
│       ├── ai_literacy_hypergraph_real.html # 真实超图
│       ├── hypergraph_visualization_summary.md # 超图可视化总结
│       └── revkey_sanity.md           # 反向题检查报告
├── .gitignore                         # Git忽略文件
├── requirements.txt                   # 项目依赖
├── push_to_github.bat                 # GitHub推送脚本
├── push.ps1                           # PowerShell推送脚本
└── README.md                          # 项目说明文档
```

## 使用方法

### 1. 环境设置

```bash
pip install -r requirements.txt
```

### 2. 设置环境变量

```bash
# 设置OpenAI API密钥（用于LLM解释）
export OPENAI_API_KEY="your-api-key-here"
```

### 3. 运行主Pipeline

```bash
python main_pipeline.py
```


## Pipeline流程

1. **数据预处理**：添加ID、规范化、质量控制
2. **对比学习**：已经验证过同一维度的题项在语义空间上越靠近，不同维度的题项在语义空间中越远离
3. **向量嵌入**：使用Llama3-8B生成4096维向量
4. **反向题检查**：验证反向题项的一致性
5. **相似度计算**：计算题项间的余弦相似度
6. **PCA分析**：提取潜在构念
7. **LLM解释**：使用GPT-4o自动命名和解释构念
8. **可视化**：生成UMAP可视化图

## 输出文件

- `data/processed/pca_scores.npy` - PCA得分矩阵
- `data/processed/pca_loadings.csv` - PCA载荷
- `data/processed/pca_top_items.csv` - 每个主成分的代表性题项
- `data/processed/pca_interpretations.csv` - LLM解释结果
- `data/docs/umap.png` - UMAP可视化图
- `data/docs/revkey_sanity.md` - 反向题检查报告

## 待改进
### 对比学习部分
1. 在对比学习中，需要考虑问卷的respondant
2. 需要处理问卷的中文和英文问题
3. 直接用LLM的embedding可能不会影响一定的结果

## 文档

详细的技术说明请参考 `docs/PIPELINE_COMPARISON.md`。
