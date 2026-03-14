# Corpus Overview 页面重构需求

## 背景

Corpus Overview 页面经历多次迭代修改，累积了以下技术债：冗余的 tab 结构（Density tab 实际不可用、Full corpus tab 与 By framework tab 功能重叠）、Streamlit session state 中残留的 zombie key、客户端 JS 状态与 Python 端 session state 断裂导致联动失效。

核心矛盾在于 **Streamlit 与 Plotly 两套交互模型的冲突**：Streamlit 的任何状态变化都会触发整个 Python 脚本重跑、页面重建；而 Plotly 在浏览器端可以通过 JS 就地修改 DOM、零延迟零刷新。当前代码中两套系统在同一页面交替接管控制权——第一级用 `components.html` 嵌入纯客户端 Plotly（交互流畅），但需要切换视图时又不得不回到 Streamlit 的 `st.rerun()` 逻辑（页面闪烁、状态丢失）。

本次重构的核心决策：**全部交互逻辑统一收归客户端 JS**，彻底消除 Streamlit 交互的介入。

---

## 技术架构：全客户端方案

### 设计原则

Streamlit 仅承担两项职责：首次加载时调用 Python 服务层获取数据、将数据序列化为 JSON 注入 `components.html` 的 iframe。所有用户交互（图例高亮、区域钻取、面包屑返回、阈值过滤）全部在 iframe 内部用 JS 完成，不回调 Streamlit、不触发 `st.rerun()`。

### Python 端职责（仅执行一次）

`render()` 函数精简为以下流程：

1. 调用 `get_all_frameworks()` 获取框架列表
2. 调用 `get_graph_data()` 获取全局图数据（含 UMAP 坐标、construct 归属）
3. 调用 `get_construct_embeddings()` 获取 construct 嵌入向量
4. 将上述数据预处理为两级视图所需的 JSON 结构（见下方"数据契约"）
5. 调用 `components.html(html_template, height=..., scrolling=False)` 一次性注入

Python 端不再维护任何 session state key。`SESSION_VIEW`、`SESSION_FOCUSED_DOMAIN`、`SESSION_CHART_KEY`、`SESSION_FW_ACTIVE` 等全部删除。

### 数据契约：Python → JS 的 JSON 结构

Python 端预计算好以下数据，序列化为 JSON 注入 iframe：

```json
{
  "frameworks": [
    {
      "id": "fw_001",
      "name": "IPIP-NEO",
      "color": "#E8740C",
      "centroid": { "x": 0.45, "y": 0.38 },
      "hull": [ { "x": 0.3, "y": 0.2 }, { "x": 0.6, "y": 0.2 }, ... ],
      "constructs": [
        { "id": "c_001", "name": "Anxiety", "x": 0.42, "y": 0.35 },
        { "id": "c_002", "name": "Depression", "x": 0.38, "y": 0.30 },
        ...
      ],
      "edges": [
        { "source": "c_001", "target": "c_002", "weight": 0.49 },
        ...
      ]
    },
    ...
  ]
}
```

关键点：每个框架的 `edges` 数组包含该框架内所有 construct 两两之间的余弦相似度（全量传入，JS 端按阈值过滤显示）。`constructs` 中的 `x`/`y` 是全局 UMAP 坐标，仅用于第一级展示；第二级的力导向布局由 JS 端实时计算。

### JS 端架构

iframe 内引入两个库：

- **Plotly.js**（CDN `plotly-2.27.0.min.js`）：负责第一级的 UMAP 散点图（凸包、菱形、restyle 动画）和第二级的网络图（节点、连边的 Scatter trace）
- **d3-force**（CDN `d3-force@3` 的 standalone UMD）：仅用于第二级的力导向布局计算，输出节点坐标后交给 Plotly 绑定

JS 内部维护一个状态机：

```
状态枚举: OVERVIEW | HIGHLIGHTED | DETAIL
```

| 当前状态 | 用户动作 | 目标状态 | JS 行为 |
|---------|---------|---------|--------|
| OVERVIEW | 图例点击框架 A | HIGHLIGHTED(A) | `Plotly.restyle` 高亮 A，淡出其余 |
| HIGHLIGHTED(A) | 图例再次点击 A | OVERVIEW | `Plotly.restyle` 恢复全部默认 |
| HIGHLIGHTED(A) | 图例点击框架 B | HIGHLIGHTED(B) | `Plotly.restyle` 切换高亮到 B |
| HIGHLIGHTED(A) | 点击 A 的凸包/菱形 | DETAIL(A) | `Plotly.purge` 清空画布 → 计算力导向布局 → `Plotly.newPlot` 绘制网络图 |
| DETAIL(A) | 面包屑 "Frameworks" | OVERVIEW | `Plotly.purge` 清空 → `Plotly.newPlot` 重绘第一级全景 |
| DETAIL(A) | 拖动阈值 slider | DETAIL(A) | `Plotly.restyle` 更新连边 visible 属性 |

### 第一级的 Plotly 图点击事件绑定

当状态为 HIGHLIGHTED 时，监听 `plotly_click` 事件。在回调中判断点击的 trace 是否属于当前高亮框架的凸包 fill trace 或 diamond trace：

```javascript
gd.on('plotly_click', function(eventData) {
    if (state !== 'HIGHLIGHTED') return;
    var pointIndex = eventData.points[0].curveNumber;
    var activeFw = traceMap.find(fw => fw.id === activeId);
    var isOwnTrace = (
        pointIndex === activeFw.traceIndices.hullFill ||
        pointIndex === activeFw.traceIndices.diamond
    );
    if (isOwnTrace) enterDetail(activeId);
});
```

### 第二级的力导向布局实现

进入 DETAIL 状态后，JS 端执行以下步骤：

1. 从预加载的 JSON 中取出该框架的 `constructs` 和 `edges` 数组
2. 根据当前阈值（默认 0.2）过滤 `edges`
3. 用 `d3.forceSimulation` 计算布局：`d3.forceLink` 以 `(1 - weight)` 为 distance（相似度越高距离越短）、`d3.forceManyBody` 做斥力、`d3.forceCenter` 居中
4. 模拟收敛后（约 300 次 tick），提取每个节点的 `x`/`y` 坐标
5. 用 `Plotly.newPlot` 绘制：节点为 `scatter` trace（mode `markers+text`）、连边为逐条 `scatter` trace（mode `lines`，中点加 annotation 标注权重）

### 第二级的阈值 slider

阈值 slider 不使用 Streamlit 的 `st.slider`，而是在 iframe 内用原生 HTML `<input type="range">` 实现，放置在画布上方（面包屑右侧）。拖动时触发 JS 回调：遍历所有连边 trace，将权重低于阈值的 trace 设为 `visible: false`，通过 `Plotly.restyle` 一次性更新，无页面刷新。

---

## 删除内容

### 删除的 Python 函数与逻辑

- `_tab_overview()` 函数整体（Full corpus 的 Domain map → Constructs drill-down）
- `_show_density_overview()` 函数整体
- `_show_network_plotly()` 函数整体（仅在 fallback 路径调用的死代码）
- `render()` 中的 `st.tabs(["Full corpus", "By framework"])` 双 tab 结构
- `_tab_framework()` 中的 `st.tabs(["Network", "Density"])` 子 tab 结构
- `_filtered_embeddings()` 中对 `domain_id` 的分支逻辑（不再有 domain 级过滤）

### 删除的 Session State Key

- `SESSION_VIEW` (`corpus_overview_view`)
- `SESSION_FOCUSED_DOMAIN` (`corpus_overview_focused_domain_id`)
- `SESSION_CHART_KEY` (`corpus_overview_network`)
- `SESSION_FW_ACTIVE` (`corpus_overview_fw_active_id`)
- `SESSION_FW_SHOW_LABELS` (`corpus_overview_fw_show_labels`)
- `SESSION_FW_SHOW_CROSS_LINKS` (`corpus_overview_fw_show_cross_links`)
- `SESSION_FW_CROSS_THRESHOLD` (`corpus_overview_fw_cross_threshold`)
- 旧残留 key 的清理代码（`overview_expanded_multiselect` 等）

### 删除的 Sidebar 控件

- "Network options" 中的 `st.slider`（`overview_min_edge_weight`）
- "Jump to Domain" 中的 `st.selectbox` + `st.button`（`overview_jump_domain`、`overview_jump_go`）
- "Back to Domain map" 按钮（`overview_back_domains`）

### 删除的 Import

- `ssn.services.density_service` 的全部导入（`estimate_density`、`compute_redundancy_risk`、`intra_domain_density`）
- `ssn.services.decomposition_service.run_decomposition`
- `ssn.components.network_viz.plot_network_from_plotly_data`
- `ssn.components.scatter_viz.plot_density_scatter`
- `ssn.db.schema` 中仅用于 Full corpus 的函数（`get_all_domains`、`get_constructs_by_domain`、`get_domains_by_framework`、`get_items_by_construct`）

---

## 保留内容

- `get_all_frameworks()`、`get_graph_data()`、`get_construct_embeddings()` 的调用（数据获取）
- `build_all_frameworks_umap_figure_for_client()` 的调用（第一级 Plotly figure 构建）
- `_framework_color()` 工具函数
- `_umap_client_html()` 函数（需扩展以包含第二级逻辑）
- `_compute_framework_overlap()` 函数（Overlap Index 面板保留）
- `_convex_hull_2d()` 工具函数

---

## 第一级：框架全景视图

保持当前 By framework 的 UMAP 视图效果不变：

- 画布展示所有框架的菱形质心标记和凸包轮廓线
- 右侧图例为唯一交互入口：点击图例中的框架名称，高亮该框架（凸包变实线、加粗），其余淡出
- 再次点击同一框架图例，取消高亮、恢复全景默认视图
- 客户端 `Plotly.restyle` 实现局部更新，不触发 Streamlit 页面刷新

此级回答的核心问题：框架之间在语义空间中的位置关系和覆盖重叠程度。

### 需修复的已知 Bug

1. **`Plotly.restyle` 不支持 transition 参数**：当前第 165 行传入 `{ transition: { duration: 350 } }` 无效。如需过渡动画，应改用 `Plotly.animate` 或接受无动画的瞬间切换。
2. **`activeId === null` 时全部元素淡出而非恢复正常**：`applyRestyle(null)` 中，每个框架的 `isActive` 都为 `false`，导致所有凸包 opacity 被设为 0.02、所有菱形 opacity 被设为 0.2。应在 `id === null` 时将全部元素恢复为默认 opacity 值。

---

## 第二级：框架内部 Construct 网络视图

### 触发条件

在第一级中，某框架已通过图例点击处于高亮状态时（状态机为 `HIGHLIGHTED`），用户点击该框架的凸包区域或菱形标记，进入第二级（状态机切换为 `DETAIL`）。高亮是前置条件——未高亮的框架不响应区域点击，避免误触。

### 画布行为

调用 `Plotly.purge(gd)` 清除第一级的所有图形元素，然后调用 `Plotly.newPlot(gd, ...)` 完全重绘为一张独立的网络图。不在全局 UMAP 上叠加图层，而是一张全新的、仅包含该框架数据的干净视图。右侧图例区域隐藏（`display: none`），将全部宽度让给网络图。

### 节点

| 属性 | 规格 |
|------|------|
| 数据来源 | 该框架下的全部 construct |
| Plotly trace 类型 | `scatter`，mode = `markers+text` |
| 形状 | 圆形（marker.symbol = `circle`） |
| 颜色 | 继承该框架在第一级中的色相 |
| 大小 | 统一尺寸（marker.size = 14），或可选按节点度数映射（范围 10–24） |
| 标签 | `text` 属性设为 construct 完整名称，`textposition = "top center"`，`textfont.size = 12`，`textfont.color = "#333"` |

### 连边

| 属性 | 规格 |
|------|------|
| 数据来源 | 框架内 construct 两两之间的余弦相似度（从预加载 JSON 的 `edges` 数组读取） |
| 过滤条件 | 相似度 ≥ 当前阈值时 `visible: true`，否则 `visible: false`（默认阈值 0.2） |
| Plotly trace 类型 | 每条连边为一个独立的 `scatter` trace，mode = `lines`，`x`/`y` 为 `[source.x, target.x, null]`/`[source.y, target.y, null]` |
| 粗细 | `line.width` 映射权重值：`1 + (weight - minWeight) / (maxWeight - minWeight) * 3`，范围 1px–4px |
| 颜色 | `line.color = "#CCCCCC"` |
| 权重标注 | 每条连边的中点位置添加一个 Plotly annotation：`text` = 权重值保留两位小数，`font.size = 10`，`font.color = "#999"`，`showarrow = false` |

### 布局

使用 d3-force 力导向布局，在 JS 端实时计算：

```javascript
var simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(filteredEdges)
        .id(d => d.id)
        .distance(d => (1 - d.weight) * 300))  // 相似度越高距离越短
    .force("charge", d3.forceManyBody().strength(-200))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .stop();

for (var i = 0; i < 300; i++) simulation.tick();  // 同步收敛，不做逐帧动画
```

收敛后提取 `nodes[i].x` / `nodes[i].y` 作为 Plotly trace 的坐标。

### 导航栏

画布上方（iframe 内、Plotly 图外）渲染一行 HTML 导航栏：

```
[← Frameworks]    Constructs in IPIP-NEO    Min weight: [====●====] 0.20
```

- 左侧："← Frameworks" 为可点击按钮，点击后调用 `exitDetail()` 返回第一级
- 中间：标题 "Constructs in {框架名称}"，加粗
- 右侧：`<input type="range">` 原生 slider，范围 0.00–0.90，步长 0.05，默认 0.20。拖动时触发 JS 回调，通过 `Plotly.restyle` 批量更新连边 `visible` 属性和 annotation `visible` 属性

导航栏和 slider 均在 iframe 内部用 HTML/CSS 实现，不依赖 Streamlit 控件。

---

## 交互流程总结

```
页面加载
  │
  ├─ Python 端：获取数据 → 序列化为 JSON → 注入 components.html → 职责结束
  │
  └─ JS 端接管（以下全部在 iframe 内完成，零 Streamlit 交互）
      │
      └─ 状态机初始化为 OVERVIEW
          │
          ├─ 图例单击框架 A → HIGHLIGHTED(A)
          │   │  Plotly.restyle: A 高亮，其余淡出
          │   │
          │   ├─ 图例再次点击 A → OVERVIEW
          │   │   Plotly.restyle: 全部恢复默认
          │   │
          │   ├─ 图例点击框架 B → HIGHLIGHTED(B)
          │   │   Plotly.restyle: 切换高亮到 B
          │   │
          │   └─ 点击 A 的凸包/菱形 → DETAIL(A)
          │       Plotly.purge → d3-force 计算布局 → Plotly.newPlot 绘制网络图
          │       显示导航栏 + 阈值 slider，隐藏图例
          │       │
          │       ├─ 拖动阈值 slider → DETAIL(A)（不变）
          │       │   Plotly.restyle: 更新连边和 annotation 的 visible
          │       │
          │       └─ 点击 "← Frameworks" → OVERVIEW
          │           Plotly.purge → Plotly.newPlot 重绘第一级全景
          │           隐藏导航栏，恢复图例显示
          │
          └─ 点击未高亮框架的区域 → 不响应
```

---

## 重构后的 Python 端代码结构

重构后 `06_corpus_overview.py` 应精简至约 150–200 行，结构如下：

```python
"""Corpus Overview – Framework UMAP + Construct network drill-down."""

import json
import streamlit as st
import streamlit.components.v1 as components

from ssn.services.graph_data_service import get_graph_data
from ssn.components.zmlt_plotly import (
    build_all_frameworks_umap_figure_for_client,
    _framework_color,
)
from ssn.services.embedding_service import get_construct_embeddings
from ssn.db.schema import get_all_frameworks


def _prepare_client_data(frameworks, graph_data, embeddings) -> dict:
    """预计算两级视图所需的全部数据，返回可序列化的 dict。"""
    # 第一级: 复用 build_all_frameworks_umap_figure_for_client
    # 第二级: 为每个框架计算 construct 间余弦相似度 edges
    ...


def _build_client_html(fig_dict, trace_map, detail_data) -> str:
    """生成包含第一级 + 第二级全部逻辑的 HTML 模板。"""
    # 注入 Plotly CDN + d3-force CDN
    # 注入 fig_dict (第一级), detail_data (第二级)
    # 注入 JS 状态机 + 交互逻辑
    ...


def render():
    st.title("Corpus Overview")
    frameworks = get_all_frameworks()
    graph_data = get_graph_data()
    embeddings = get_construct_embeddings()

    if not frameworks or not graph_data:
        st.warning("数据未就绪，请先运行 pipeline。")
        return

    fig_dict, trace_map = build_all_frameworks_umap_figure_for_client(
        graph_data, frameworks,
        title="Frameworks in UMAP semantic space",
        height=600,
    )
    detail_data = _prepare_client_data(frameworks, graph_data, embeddings)
    html = _build_client_html(fig_dict, trace_map, detail_data)
    components.html(html, height=660, scrolling=False)


render()
```

无 session state、无 `st.rerun()`、无 sidebar 控件、无 tab 切换。
