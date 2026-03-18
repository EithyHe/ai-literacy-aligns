# Corpus Overview — Requirements Document (Final)

## AI Literacy Construct Corpus Visualization

**Version:** v2.0-final  
**Date:** 2026-03-17  
**Status:** Confirmed  

---

## 1. Project Context

### 1.1 Background

This module is the "Corpus Overview" page of the LLM-Based Semantic Scale Network system. It provides corpus-level visualization and exploratory analysis for an AI literacy construct corpus, enabling researchers to understand the field from three complementary perspectives.

### 1.2 Corpus Data Profile

| Parameter | Value | Notes |
|-----------|-------|-------|
| Construct nodes | ~400 | Normalized constructs extracted from scales |
| Scales | 32 | Published measurement instruments in AI literacy |
| Fine-grained clusters | 168 | Semantic clusters from embedding clustering, avg ~2.4 nodes each |
| Super-clusters (derived) | 10–15 | Second-level aggregation of 168 clusters for visual encoding |

### 1.3 Three-Tab Architecture

The page has three tabs, each with a distinct analytical focus:

| Tab | Name | Core Question | Primary Perspective |
|-----|------|---------------|-------------------|
| Tab 1 | Semantic Map | How do constructs distribute in **meaning** space? | Semantic structure |
| Tab 2 | Measurement Decomposition | How do different scales **decompose** the AI literacy domain? | Measurement practice |
| Tab 3 | Integrated View | Where do semantic structure and measurement practice **align or diverge**? | Semantic × Measurement |

---

## 2. Technology Stack

### 2.1 Per-Tab Tech Choices

| Component | Tab 1: Semantic Map | Tab 2: Measurement Decomposition | Tab 3: Integrated View |
|-----------|--------------------|---------------------------------|----------------------|
| **Visualization library** | Plotly | Cytoscape.js | Plotly |
| **Streamlit integration** | `st.plotly_chart()` | `st.components.v1.html()` | `st.plotly_chart()` |
| **Layout method** | UMAP preset coordinates | Force-directed (`fcose`) + overlap removal post-processing | UMAP preset coordinates |
| **Rationale** | Scatter plot with overlays — Plotly's native strength | Graph-native: force layout, edge class management, tap events | Scatter + edge lines — Plotly natural fit |

## 3. Tab 1: Semantic Map

### 3.1 Purpose

Show how constructs distribute in semantic embedding space. Reveal cluster structure, semantic density, and potential jingle-jangle issues.

**Core questions this tab answers:**
- What are the major semantic regions in AI literacy research?
- Which regions are densely populated (over-measured) vs. sparse (under-explored)?
- Are there constructs with similar names but distant positions (jingle fallacy)?
- Are there constructs with different names but nearby positions (jangle fallacy)?

### 3.2 Visualization Specification

**Chart type:** Plotly `go.Scatter` (mode='markers')

| Visual Element | Data Encoding |
|----------------|--------------|
| Point position (x, y) | UMAP 2D coordinates (precomputed) |
| Point color | Super-cluster membership (discrete colors from shared palette) |
| Point size | Frequency (number of scales measuring this construct) — scaled to radius range 4–16px |
| Default opacity | 0.60 (non-highlighted); 1.0 (hovered or pinned) |

### 3.3 Layout

The page uses a two-column layout:

```
st.columns([3, 1])
├── Left (3): Plotly scatter chart (full height, no axis labels)
└── Right (1):
    ├── Cluster legend (always visible)
    ├── Cluster detail panel (appears below legend on click, collapsible)
    └── Construct detail panel (always visible, updates on hover/click)
```

**Removed controls (compared to earlier spec):**
- Super-cluster filter multiselect — removed (redundant with legend click interaction)
- "Color by" radio button — removed (cluster color is the primary and only encoding)
- "Point size" slider — removed (size encodes frequency, should not be user-adjustable)
- "Run LLM cluster labels" button — removed (labels are precomputed offline; display directly)

### 3.4 Cluster Legend (right panel, top)

A static list of all super-clusters with color dots and LLM-generated names. Each row is **clickable**.

**Default state:** All rows shown at normal weight. Chart shows all clusters at full opacity.

**On cluster row click:**
1. The clicked row becomes bold / highlighted
2. The chart dims all nodes not belonging to that cluster (opacity → 0.12)
3. A **Cluster Detail Panel** appears immediately below the legend (see §3.5)

**On second click of same row (or × button in detail panel):** panel collapses, chart resets to full opacity.

Only one cluster can be active at a time.

### 3.5 Cluster Detail Panel (right panel, below legend, conditional)

Appears when a cluster row is clicked. Disappears when dismissed.

**Content:**

```
● [Cluster Name]                          [×]
─────────────────────────────────────────
[N] constructs in corpus

[Construct tag] [Construct tag] [Construct tag]
[Construct tag] [Construct tag] ...
```

Construct tags are small pill-shaped labels (`background: var(--color-background-secondary)`, font-size 10px). The list shows the canonical construct names belonging to this cluster, taken directly from precomputed cluster metadata (no LLM call at render time).

### 3.6 Hover Tooltip

Appears on node mouseover when no node is pinned. Disappears on mouseleave.

**Content:**

```
[Construct Name]              (bold, 12px)
Cluster: [Cluster Name]       (cluster name in cluster color)
Frequency: [N] scale(s)
Scales: [Scale A]; [Scale B]; [Scale C] +N more
                                         ↑ only if >3 scales
Click to pin
```

**Scale truncation rule:** Show up to 3 scale names separated by semicolons. If `frequency > 3`, append `+{frequency - 3} more` in muted color. Never show the full list in the tooltip.

**Positioning:** Tooltip appears to the right of the cursor by default; flips left if within 220px of the right canvas edge. Clamps vertically to stay within canvas bounds.

### 3.7 Construct Detail Panel (right panel, bottom)

Always visible. Updates on hover (preview) and click (pinned).

**Default state:**
```
Hover to preview · Click to pin
```

**Hover state (preview, not pinned):** Same content as tooltip but in panel form — construct name, cluster, frequency, scale preview with "+N more".

**Pinned state (after click):**
```
[Construct Name]              (bold)
Cluster: [Cluster Name]       (in cluster color)
Frequency: [N] scale(s)

Measured by:
[Scale A] [Scale B] [Scale C]
[Scale D] [Scale E] ...       (all scales as tags, no truncation)

Click canvas to unpin         (hint, muted)
```

All scales shown as tags (`background: var(--color-background-secondary)`, font-size 10px, inline-block with 2px margin). No truncation in pinned state.

**Unpin:** Click anywhere on the canvas that is not a node, or click the pinned node again.

### 3.8 Plotly Implementation Guidance

```python
import plotly.graph_objects as go

fig = go.Figure()

for sc_id, sc_name in super_clusters.items():
    mask = df['super_cluster_id'] == sc_id
    subset = df[mask]

    # Build truncated scale preview strings
    def scale_preview(row):
        scales = row['scale_names']  # list
        preview = '; '.join(scales[:3])
        more = f' +{len(scales)-3} more' if len(scales) > 3 else ''
        return preview + more

    subset = subset.copy()
    subset['scale_preview'] = subset.apply(scale_preview, axis=1)

    fig.add_trace(go.Scatter(
        x=subset['umap_x'],
        y=subset['umap_y'],
        mode='markers',
        name=sc_name,
        marker=dict(
            size=subset['freq_scaled'],   # pre-scaled to 4–16px range
            color=super_cluster_colors[sc_id],
            opacity=0.60,
            line=dict(width=0.5, color='rgba(255,255,255,0.25)')
        ),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'Cluster: %{customdata[1]}<br>'
            'Frequency: %{customdata[2]} scale(s)<br>'
            'Scales: %{customdata[3]}'
            '<extra></extra>'
        ),
        customdata=subset[[
            'construct_name', 'super_cluster_name',
            'frequency', 'scale_preview'
        ]].values,
        # construct_id in customdata[4] for click handling
    ))

fig.update_layout(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    showlegend=False,          # legend replaced by custom Streamlit widget
    margin=dict(l=10, r=10, t=10, b=10),
    height=520
)
```

**Note:** Plotly's built-in legend is disabled (`showlegend=False`). The cluster legend is implemented as a custom Streamlit component in the right column, enabling the click-to-highlight and cluster detail panel interactions.

### 3.9 Click Handling (Streamlit)

```python
# Cluster highlighting: managed in st.session_state
active_cluster = st.session_state.get('active_cluster', None)
pinned_construct = st.session_state.get('pinned_construct', None)

# Rebuild figure with dimming applied
for sc_id, sc_name in super_clusters.items():
    opacity = 0.60 if (active_cluster is None or sc_id == active_cluster) else 0.10
    # ... add trace with computed opacity

# Capture construct click
event = st.plotly_chart(fig, on_select='rerun', selection_mode='points', key='tab1_chart')
if event and event.selection and event.selection.points:
    clicked_id = event.selection.points[0].customdata[4]
    if clicked_id == pinned_construct:
        st.session_state.pinned_construct = None
    else:
        st.session_state.pinned_construct = clicked_id
    st.rerun()
```

### 3.10 Summary Metrics

Display above the chart using `st.columns(3)`:

| Metric | Value |
|--------|-------|
| Constructs | `len(df)` |
| Semantic clusters | `df['super_cluster_id'].nunique()` |
| Scales | `len(scales)` |

*(Fine-grained cluster count removed from summary — too technical for the primary audience.)*

## 4. Acceptance Criteria

### 4.1 Tab 2

- [ ] 400 nodes render with semantic cluster colors (same palette as Tab 1)
- [ ] Node size scales with frequency
- [ ] Community-aware fcose layout produces visually distinct community islands with spatial gaps
- [ ] Intra-community edges visible at moderate opacity; cross-community edges visible but very faint
- [ ] Only top-15% edges (by `shared_scale_count`) used for layout and rendering by default
- [ ] Community labels rendered at centroid of each island, faint
- [ ] Mixed-color islands correctly reflect constructs from different semantic clusters in the same Leiden community
- [ ] Clicking a node: non-adjacent nodes dim to near-invisible
- [ ] Clicking a node: intra-community edges shown in partner's semantic cluster color at full opacity
- [ ] Clicking a node: cross-community edges shown in partner's semantic cluster color at reduced opacity
- [ ] Detail panel shows selected construct's semantic cluster, Leiden community, same-community neighbors, cross-community neighbors
- [ ] Clicking background resets to default state
- [ ] Edge strength percentile slider filters edges and triggers layout re-run
- [ ] Year filter correctly filters scales and updates layout

