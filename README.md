# AI Literacy Aligns - 人工智能素养评估量表对齐

## 项目概述

用于分析AI素养评估量表，通过向量嵌入和PCA分析提取潜在构念，并使用LLM进行自动解释和命名。

## 项目结构

```
ai-literacy-aligns/
├── main_pipeline.py          # 主pipeline文件（推荐使用）
├── src/                      # 源代码目录
│   ├── preprocessing.py      # 数据预处理
│   ├── embeddings.py         # 向量嵌入
│   ├── sanity_rev.py         # 反向题检查
│   ├── neighbors.py          # 相似度计算
│   ├── pca_analysis.py       # PCA分析
│   ├── llm_interpretation.py # LLM解释
│   ├── umap_plot.py          # UMAP可视化
│   ├── umap_from_scores.py   # 基于PCA得分的UMAP
│   ├── cluster.py            # 聚类分析
│   ├── construct_clustering.py # 构念聚类
│   ├── anchors.py            # 锚点分析
│   ├── config.py             # 配置文件
│   └── cli.py                # 命令行接口
├── data/                     # 数据目录
│   ├── interim/              # 中间数据
│   ├── processed/            # 处理后的数据
│   └── docs/                 # 数据文档
├── docs/                     # 项目文档
│   └── PIPELINE_COMPARISON.md # Pipeline对比说明
└── requirements.txt          # 依赖包
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
2. 直接用LLM的embedding可能不一定适合教育领域

## 文档

详细的技术说明请参考 `docs/PIPELINE_COMPARISON.md`。
