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

**Progression logic:** meaning → measurement → meaning × measurement

### 1.4 168-Cluster Visual Encoding Strategy

With 168 fine-grained clusters averaging ~2.4 nodes each, discrete color encoding is infeasible (human vision reliably distinguishes 8–12 colors). The solution:

1. **Pre-compute a second-level aggregation**: cluster the 168 micro-cluster centroids into 10–15 **super-clusters** using agglomerative clustering (optimal k determined by silhouette score)
2. **Color encodes super-cluster** (10–15 discrete colors) in Tab 1 and Tab 3
3. **Spatial proximity** naturally preserves micro-cluster structure (UMAP co-locates same-cluster nodes)
4. **Hover tooltips** show micro-cluster ID and peer count for fine-grained detail

This strategy applies to Tab 1 and Tab 3. Tab 2 uses neutral gray nodes (see below).

---

## 2. Technology Stack

### 2.1 Per-Tab Tech Choices

| Component | Tab 1: Semantic Map | Tab 2: Measurement Decomposition | Tab 3: Integrated View |
|-----------|--------------------|---------------------------------|----------------------|
| **Visualization library** | Plotly | Cytoscape.js | Plotly |
| **Streamlit integration** | `st.plotly_chart()` | `st.components.v1.html()` | `st.plotly_chart()` |
| **Layout method** | UMAP preset coordinates | Force-directed (`fcose`) + overlap removal post-processing | UMAP preset coordinates |
| **Rationale** | Scatter plot with overlays — Plotly's native strength | Graph-native: force layout, edge class management, tap events | Scatter + edge lines — Plotly natural fit |

### 2.2 Rationale for Mixed Stack (Plotly + Cytoscape)

Tab 1 and Tab 3 are fundamentally **scatter plots with overlays** (fixed UMAP coordinates, optional edge lines, color encoding). Plotly offers first-party Streamlit support (`st.plotly_chart`), native hover tooltips, and straightforward color/size encoding.

Tab 2 requires **graph-native capabilities**: force-directed layout, per-class edge styling (different color per scale), and interactive tap-to-reveal. Cytoscape.js is purpose-built for these. Specifically, Cytoscape's edge class system (`cy.edges('.scale-5')`) allows zero-cost per-scale edge color assignment and show/hide toggling.

Each tab renders independently — no cross-tab state synchronization needed.

### 2.3 Required Data Inputs

The following data must be precomputed and available to the Streamlit app:

| Data | Format | Columns / Fields | Used By |
|------|--------|-----------------|---------|
| Construct metadata | CSV/DataFrame | `construct_id`, `construct_name`, `first_year`, `frequency` (number of scales containing it) | All tabs |
| UMAP coordinates | CSV/DataFrame | `construct_id`, `umap_x`, `umap_y` | Tab 1, Tab 3 |
| Micro-cluster assignments | CSV/DataFrame | `construct_id`, `cluster_id` | Tab 1, Tab 3 (hover) |
| Super-cluster assignments | CSV/DataFrame | `cluster_id`, `super_cluster_id`, `super_cluster_name`, `super_cluster_color` | Tab 1, Tab 3 |
| Scale metadata | CSV/DataFrame | `scale_id`, `scale_name`, `year`, `author`, `n_constructs` | Tab 2 |
| Scale–construct membership | CSV/DataFrame | `scale_id`, `construct_id` | Tab 2, Tab 3 |
| Co-measurement edge list | CSV/DataFrame | `construct_a`, `construct_b`, `shared_scale_count`, `jaccard_weight`, `first_coappearance_year`, `shared_scale_ids` (list) | Tab 2, Tab 3 (ego mode) |
| Cluster-pair edge list | CSV/DataFrame | `cluster_a`, `cluster_b`, `total_construct_pairs`, `n_scales`, `dominant_scale`, `dominant_scale_share`, `scale_ids` (list) | Tab 3 (global overview) |
| Super-cluster color palette | Dict/JSON | `super_cluster_id` → hex color (10–15 entries) | Tab 1, Tab 3 |
| Scale color palette | Dict/JSON | `scale_id` → hex color (32 entries, high-saturation HSL, evenly spaced hue) | Tab 2 |

### 2.4 Data Preprocessing Pipeline

#### 2.4.1 Super-Cluster Generation

```
Input:  168 micro-cluster centroids (mean of member construct embeddings)
Method: AgglomerativeClustering(n_clusters=k), k chosen by silhouette score (target 10–15)
Output: cluster_id → super_cluster_id mapping table
        super_cluster_id → super_cluster_name (LLM-generated from member construct names, expert-validated)
```

#### 2.4.2 Co-Measurement Edge Computation

```python
for each pair (construct_a, construct_b):
    shared_scales = set(scales_of[a]) & set(scales_of[b])
    if len(shared_scales) >= 1:
        edge = {
            'construct_a': a,
            'construct_b': b,
            'shared_scale_count': len(shared_scales),
            'jaccard_weight': len(shared_scales) / len(set(scales_of[a]) | set(scales_of[b])),
            'first_coappearance_year': min(year_of[s] for s in shared_scales),
            'shared_scale_ids': list(shared_scales)
        }
```

#### 2.4.3 Backbone Filtering (for Tab 3)

Use Serrano's disparity filter to extract statistically significant edges. Pre-compute backbone at multiple alpha thresholds (0.01, 0.05, 0.10, 0.20) so the user can adjust filtering strength interactively without recomputation.

#### 2.4.4 Scale Color Palette Generation

```python
scale_colors = {}
for i, scale_id in enumerate(sorted(all_scale_ids)):
    hue = (i / len(all_scale_ids)) * 360
    scale_colors[scale_id] = f"hsl({hue}, 85%, 50%)"  # high saturation for edge contrast
```

---

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
| Point color | Super-cluster membership (10–15 discrete colors from palette) |
| Point size | Frequency (number of scales measuring this construct) — `mapData(freq, 1, max_freq, 5, 18)` |
| Hover tooltip | Construct name, super-cluster name, micro-cluster ID + peer count, frequency, first year |

### 3.3 Plotly Implementation Guidance

```python
import plotly.graph_objects as go

fig = go.Figure()

for sc_id, sc_name in super_clusters.items():
    mask = df['super_cluster_id'] == sc_id
    subset = df[mask]
    fig.add_trace(go.Scatter(
        x=subset['umap_x'],
        y=subset['umap_y'],
        mode='markers',
        name=sc_name,
        marker=dict(
            size=subset['freq_scaled'],  # pre-scaled to 5–18 range
            color=super_cluster_colors[sc_id],
            opacity=0.75,
            line=dict(width=0.5, color='rgba(255,255,255,0.3)')
        ),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'Super-cluster: %{customdata[1]}<br>'
            'Micro-cluster: #%{customdata[2]} (%{customdata[3]} peers)<br>'
            'Frequency: %{customdata[4]} scales<br>'
            'First appeared: %{customdata[5]}'
            '<extra></extra>'
        ),
        customdata=subset[['construct_name', 'super_cluster_name', 'cluster_id', 'cluster_peer_count', 'frequency', 'first_year']].values
    ))

fig.update_layout(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    legend=dict(orientation='h', yanchor='top', y=-0.02),
    margin=dict(l=20, r=20, t=20, b=60),
    height=500
)
```

### 3.4 Streamlit Controls

| Control | Streamlit Widget | Function |
|---------|-----------------|----------|
| Super-cluster filter | `st.multiselect("Super-clusters", all_names, default=all_names)` | Show/hide specific super-clusters (filter traces) |
| Color by | `st.radio("Color by", ["Super-cluster", "Frequency heatmap"])` | Toggle between discrete cluster color and continuous frequency colorscale |
| Point size | `st.slider("Point size", 1, 3, 2)` | Global size multiplier |

### 3.5 Summary Metrics

Display above the chart using `st.columns(4)`:

| Metric | Value |
|--------|-------|
| Total constructs | `len(df)` |
| Fine-grained clusters | `df['cluster_id'].nunique()` |
| Super-clusters | `df['super_cluster_id'].nunique()` |
| Scales | `len(scales)` |

---

## 4. Tab 2: Measurement Decomposition

### 4.1 Purpose

Show how 32 scales each decompose the AI literacy domain into different construct combinations. Reveal decomposition strategy differences, consensus constructs, and unique measurement choices.

**Core questions this tab answers:**
- How does each scale "carve up" AI literacy? Which constructs does it include?
- Which constructs are high-consensus (selected by many scales)?
- For a specific construct: what are its co-measurement partners in each scale, and how do they differ?
- Which scales use similar decomposition strategies?

### 4.2 Visualization Specification

**Library:** Cytoscape.js via `st.components.v1.html()`

#### 4.2.1 Node Encoding

| Property | Encoding | Cytoscape Style |
|----------|----------|----------------|
| Node identity | One construct (400 nodes total) | `data: { id, label, freq }` |
| Node color | **Neutral gray** (muted, to maximize edge color contrast on click) | `'background-color': 'rgb(150,150,150)'`, `'background-opacity': 0.25` |
| Node size | Number of scales containing this construct (larger = higher measurement consensus) | `'width': 'mapData(freq, 1, max_freq, 10, 36)'`, same for `'height'` |
| Node label | Show for high-frequency constructs (freq ≥ 4); show for all neighbors on click | `'label': 'data(label)'`, `'font-size': 9` |
| Node border (default) | Thin, subtle | `'border-width': 0.5`, `'border-color': 'rgba(150,150,150,0.3)'` |

#### 4.2.2 Edge Encoding

**Default state (no node selected):** All co-measurement edges shown as very faint lines providing structural context.

| Property | Value |
|----------|-------|
| Color | `rgba(150,150,150,0.06)` (barely visible) |
| Width | `0.3 + min(shared_scale_count * 0.25, 1.5)` |

**Click state (a node is selected):** Only edges from the selected node are shown, colored per scale.

For each scale `s` that contains the selected construct:
- Draw an edge from the selected node to every other construct in scale `s`
- Edge color = `scale_colors[s]` (high-saturation HSL)
- Edge width = `2.5`
- Edge opacity = `0.7`
- Edge curve style = `unbundled-bezier` with `control-point-distances` offset by `(scale_index % 7 - 3) * 5` to prevent overlap when multiple scale edges go to the same target

**Cytoscape edge class strategy:**
```javascript
// Pre-define edge classes for each scale
// Each edge gets classes: 'co-measurement' and 'scale-{scale_id}'
// Default: all edges have 'display: element' with very low opacity
// On node tap: hide all edges, then show only edges with matching scale classes

cy.on('tap', 'node', function(evt) {
    var node = evt.target;
    var scaleIds = node.data('scales'); // array of scale_ids

    // Dim all nodes
    cy.nodes().addClass('dimmed');
    // Hide all edges
    cy.edges().addClass('hidden');

    // Highlight selected node
    node.removeClass('dimmed').addClass('selected');

    // For each scale, show and color its edges
    scaleIds.forEach(function(scaleId) {
        var scaleEdges = cy.edges('.scale-' + scaleId).filter(function(edge) {
            return edge.source().id() === node.id() || edge.target().id() === node.id();
        });
        scaleEdges.removeClass('hidden').addClass('active-scale-' + scaleId);

        // Highlight target nodes with scale color
        scaleEdges.connectedNodes().removeClass('dimmed').addClass('neighbor');
    });
});

// Tap on background to reset
cy.on('tap', function(evt) {
    if (evt.target === cy) {
        cy.nodes().removeClass('dimmed selected neighbor');
        cy.edges().removeClass('hidden');
        // Remove all active-scale classes and reset to default faint style
    }
});
```

#### 4.2.3 Target Node Coloring on Click

When a node is clicked and per-scale edges are shown, each **target node** (the node at the other end of an edge) takes on the color of the scale that connects it to the selected node:

| Scenario | Target Node Style |
|----------|------------------|
| Connected by 1 scale | Fill color = that scale's color, opacity 0.85 |
| Connected by 2+ scales | Fill color = last scale's color; border color = first scale's color, border-width = 2.5 (dual-color indicates multi-scale co-measurement) |
| Not connected (dimmed) | Fill = `rgba(150,150,150,0.05)`, size shrinks to 1.5px |
| Selected node itself | Fill = white (dark mode) or dark gray (light mode), border = white/black 2.5px, label always shown |

#### 4.2.4 Layout: Force-Directed + Overlap Removal

**Step 1: Force-directed layout** using Cytoscape `fcose` extension:

```javascript
var layout = cy.layout({
    name: 'fcose',
    quality: 'proof',           // highest quality
    randomize: true,
    animate: false,
    nodeRepulsion: 8000,        // moderate repulsion
    idealEdgeLength: 80,
    edgeElasticity: 0.1,
    gravityRange: 3.8,
    nestingFactor: 0.1,
    numIter: 2500,
    tile: true,
    tilingPaddingVertical: 20,
    tilingPaddingHorizontal: 20,
});
layout.run();
```

**Step 2: Overlap removal post-processing:**

```javascript
// After fcose layout completes, run overlap removal
function removeOverlaps(cy, minDist, maxPasses) {
    minDist = minDist || 25;  // minimum 25px between node centers
    maxPasses = maxPasses || 50;

    for (var pass = 0; pass < maxPasses; pass++) {
        var moved = false;
        var nodes = cy.nodes();
        for (var i = 0; i < nodes.length; i++) {
            for (var j = i + 1; j < nodes.length; j++) {
                var p1 = nodes[i].position();
                var p2 = nodes[j].position();
                var dx = p1.x - p2.x;
                var dy = p1.y - p2.y;
                var dist = Math.sqrt(dx * dx + dy * dy) || 0.001;
                // Account for node sizes
                var r1 = nodes[i].width() / 2;
                var r2 = nodes[j].width() / 2;
                var requiredDist = r1 + r2 + minDist;

                if (dist < requiredDist) {
                    var push = (requiredDist - dist) / 2 * 0.6;
                    var ux = dx / dist;
                    var uy = dy / dist;
                    nodes[i].position({ x: p1.x + ux * push, y: p1.y + uy * push });
                    nodes[j].position({ x: p2.x - ux * push, y: p2.y - uy * push });
                    moved = true;
                }
            }
        }
        if (!moved) break;
    }
}

layout.one('layoutstop', function() {
    removeOverlaps(cy, 8, 50);
});
```

### 4.3 Streamlit Controls

| Control | Streamlit Widget | Function |
|---------|-----------------|----------|
| Min co-measurement threshold | `st.slider("Min shared scales for edges", 1, 5, 1)` | Filter edges by `shared_scale_count >= threshold`. Triggers layout re-run. |
| Year filter (secondary) | `st.slider("Include scales up to year", 2010, 2025, 2025)` | Filter scales by publication year. Updates edges and layout. |

These controls are implemented in Streamlit sidebar / page controls. When changed, Python recomputes the filtered edge list and re-injects the JSON data into the Cytoscape HTML component, triggering a re-render.

### 4.4 Detail Panel

Below the Cytoscape canvas, display a detail panel (HTML within the same `st.components.v1.html()` block, or a separate `st.container()` updated via JavaScript postMessage):

**When no node selected:**
```
Click any construct to see its per-scale co-measurement partners.
```

**When a node is selected:**
```
[Construct Name] — appears in [N] scales

--- Scale 3 (2019) — 8 constructs ---
Co-measured with: [construct_a], [construct_b], [construct_c], ...
                  (each name colored in Scale 3's color)

--- Scale 7 (2021) — 12 constructs ---
Co-measured with: [construct_d], [construct_e], [construct_f], ...
                  (each name colored in Scale 7's color)

--- Cross-scale partners (co-measured in 2+ scales) ---
[construct_x] (3 scales), [construct_y] (2 scales), ...
```

### 4.5 Data Flow Summary

```
Streamlit (Python)                    Cytoscape (JavaScript)
─────────────────                    ──────────────────────
1. Load construct metadata           
2. Load scale-construct membership   
3. Compute co-measurement edges      
   (filtered by threshold & year)    
4. Build Cytoscape elements JSON:    
   - nodes: [{data: {id, label,     
     freq, scales: [...]}}]          
   - edges: [{data: {source, target,
     count, scale_id},               
     classes: 'scale-{id}'}]         
5. Inject JSON into HTML template    → 6. Parse JSON, init Cytoscape
                                     → 7. Run fcose layout
                                     → 8. Run overlap removal
                                     → 9. Bindtap events
                                     → 10. On tap: class toggling
                                          for edge/node highlighting
```

---

## 5. Tab 3: Integrated View

### 5.1 Purpose

Overlay co-measurement structure onto the semantic map (UMAP layout), revealing where measurement practice aligns with or diverges from semantic structure.

**Core questions this tab answers:**
- Which semantic clusters are frequently co-measured despite being semantically distant? (Cross-cluster amber edges — researchers treat them as complementary despite semantic distinctness)
- Is a strong co-measurement link driven by broad field consensus, or dominated by a single scale's design choices?
- Which semantically similar constructs have never been co-measured? (Dense regions with no edges — potential measurement blind spots)
- For a specific cluster: how does its semantic neighborhood differ from its measurement neighborhood?

### 5.2 Key Design Decision: Cluster-Level Edge Aggregation

**Problem:** There are ~4,473 cross-cluster construct-level co-measurement edges in the corpus. Rendering all of them produces an unreadable hairball regardless of filtering strategy, because 84% of all edges are already cross-cluster. Neither "show only cross-cluster edges" nor backbone filtering at the construct level reduces density sufficiently.

**Solution:** Aggregate construct-level edges to the **super-cluster level** for the global overview. Each pair of super-clusters is represented by a single aggregated edge, encoding two independent dimensions:

| Visual Channel | Encodes | Measurement interpretation |
|----------------|---------|---------------------------|
| **Line width** | Total number of co-measured construct pairs between the two clusters | Intensity of the cross-cluster measurement relationship |
| **Line opacity / color saturation** | Number of distinct scales contributing to those edges (`n_scales`) | Breadth of consensus — high opacity = many scales agree; low opacity = one or few scales dominate |

This reduces the maximum number of visible edges from ~4,473 to at most ~45 (one per super-cluster pair), while preserving the two most important measurement signals.

**The 2×2 signal taxonomy:**

| | Few source scales (low opacity) | Many source scales (high opacity) |
|---|---|---|
| **Few construct pairs (thin line)** | Weak / incidental — safely ignored | Broad but sparse — theoretical consensus exists, limited measurement overlap |
| **Many construct pairs (thick line)** | ⚠️ Single-scale dominated — reflects one instrument's design, not field consensus | ✦ Strong consensus signal — the most meaningful cross-cluster measurement bridges |

Single-scale-dominated edges (where one scale accounts for the majority of construct pairs, e.g. AB5C contributing ~20% of all cross-cluster edges) are additionally rendered as **dashed lines** to visually flag that they require cautious interpretation.

### 5.3 Data Requirements: Cluster-Level Edge Table

In addition to the construct-level `co_measurement_edges.csv`, precompute a cluster-level aggregation:

```python
# Precompute cluster-pair edge table
cluster_edges = []
for (sc_a, sc_b), group in construct_edges.groupby(['source_super_cluster', 'target_super_cluster']):
    if sc_a == sc_b:
        continue  # skip within-cluster edges
    all_scale_ids = set().union(*group['shared_scale_ids'])
    dominant_scale = group.explode('shared_scale_ids')['shared_scale_ids'].value_counts().index[0]
    dominant_count = group.explode('shared_scale_ids')['shared_scale_ids'].value_counts().iloc[0]
    total_pairs = len(group)
    cluster_edges.append({
        'cluster_a': sc_a,
        'cluster_b': sc_b,
        'total_construct_pairs': total_pairs,
        'n_scales': len(all_scale_ids),
        'dominant_scale': dominant_scale,
        'dominant_scale_pair_count': dominant_count,
        'dominant_scale_share': dominant_count / total_pairs,  # fraction dominated by top scale
        'scale_ids': list(all_scale_ids),
    })
cluster_edge_df = pd.DataFrame(cluster_edges)
```

Add `cluster_edge_df` as `cluster_edges.csv` to the data directory.

**Single-scale-dominated flag:** an edge is flagged as dominated when `dominant_scale_share >= 0.5` (one scale contributes ≥ 50% of construct pairs).

### 5.4 Visual Encoding: Global Overview (Mode A)

**Default state — no node selected:**

| Visual Element | Encoding |
|----------------|----------|
| Node position | UMAP coordinates (same as Tab 1) |
| Node color | Super-cluster color (same palette as Tab 1) |
| Node size | Frequency (number of scales measuring this construct) |
| Cluster-pair edge width | `0.5 + (total_construct_pairs / max_pairs) * 7` |
| Cluster-pair edge opacity | `0.12 + (n_scales / max_n_scales) * 0.68` (light mode amber `rgba(186,117,23,α)`) |
| Dominated edge style | Dashed (`stroke-dasharray: 4 3`) when `dominant_scale_share >= 0.5` |
| Within-cluster edges | Not shown in global overview — spatial proximity of nodes conveys within-cluster structure |

**Note on edge rendering with Plotly:** Cluster-pair edges connect super-cluster centroids (mean UMAP position of member constructs), not individual construct nodes. This keeps the edge layer clean and geometrically meaningful.

```python
# Compute super-cluster centroids
sc_centroids = df.groupby('super_cluster_id')[['umap_x', 'umap_y']].mean().to_dict('index')

# Render cluster-pair edges
for _, e in cluster_edge_df.iterrows():
    ca = sc_centroids[e['cluster_a']]
    cb = sc_centroids[e['cluster_b']]
    width = 0.5 + (e['total_construct_pairs'] / max_pairs) * 7
    alpha = 0.12 + (e['n_scales'] / max_n_scales) * 0.68
    dominated = e['dominant_scale_share'] >= 0.5

    fig.add_trace(go.Scatter(
        x=[ca['umap_x'], cb['umap_x'], None],
        y=[ca['umap_y'], cb['umap_y'], None],
        mode='lines',
        line=dict(
            color=f'rgba(186,117,23,{alpha:.2f})',
            width=width,
            dash='dot' if dominated else 'solid'
        ),
        hoverinfo='skip',
        showlegend=False,
        customdata=[[
            e['cluster_a'], e['cluster_b'],
            e['total_construct_pairs'], e['n_scales'],
            e['dominant_scale'], f"{e['dominant_scale_share']:.0%}"
        ]]
    ))
```

**Hover tooltip on cluster-pair edges** (requires Plotly `hovertemplate` on the edge trace):

```
C3 ↔ C6
Co-measured construct pairs: 394
Source scales: 12  →  Strong consensus ✦
Top contributing scale: IPIP-NEO (18%)
```

For dominated edges:
```
C6 ↔ Cn
Co-measured construct pairs: 149
Source scales: 3  →  Single-scale dominated ⚠️
Top contributing scale: AB5C (71%) — interpret with caution
```

### 5.5 Visual Encoding: Ego-Network Mode (Mode C)

When the user clicks a **construct node**, the view switches to ego-network focus at the construct level:

| Visual Element | Encoding |
|----------------|----------|
| Selected node | Large (radius 8), red (`#E24B4A`), labeled |
| Same micro-cluster peers | Medium (radius 5), purple (`#7F77DD`) — semantic neighbors |
| Co-measured neighbors | Medium (radius 5), green (`#1D9E75`) — measurement neighbors |
| All other nodes | Tiny (radius 2), very light gray — dimmed |
| Edges from selected node | Amber (`rgba(186,117,23,0.65)`), width 2, opacity 0.65 |
| All other edges | Hidden |
| Dynamic legend | Selected construct name, same-cluster peer count, co-measurement neighbor count, cross-cluster co-measurement count |

**Exit ego mode:** Click empty canvas area, or click "← Back to overview" button.

The right-side panel in ego mode shows a ranked breakdown of the selected cluster's cross-cluster connections (bar chart of `total_construct_pairs` per partner cluster), replacing the global edge legend.

### 5.6 Plotly Implementation Guidance

```python
import plotly.graph_objects as go

def build_overview(df, cluster_edge_df, sc_centroids, super_cluster_colors):
    fig = go.Figure()
    max_pairs = cluster_edge_df['total_construct_pairs'].max()
    max_scales = cluster_edge_df['n_scales'].max()

    # 1. Cluster-pair edges
    for _, e in cluster_edge_df.iterrows():
        ca = sc_centroids[e['cluster_a']]
        cb = sc_centroids[e['cluster_b']]
        width = 0.5 + (e['total_construct_pairs'] / max_pairs) * 7
        alpha = 0.12 + (e['n_scales'] / max_scales) * 0.68
        dominated = e['dominant_scale_share'] >= 0.5
        signal = ('Strong consensus ✦' if e['n_scales'] >= 8
                  else 'Moderate consensus' if e['n_scales'] >= 4
                  else 'Single-scale dominated ⚠️')
        fig.add_trace(go.Scatter(
            x=[ca['umap_x'], cb['umap_x'], None],
            y=[ca['umap_y'], cb['umap_y'], None],
            mode='lines',
            line=dict(color=f'rgba(186,117,23,{alpha:.2f})', width=width,
                      dash='dot' if dominated else 'solid'),
            hovertemplate=(
                f'<b>{e["cluster_a"]} ↔ {e["cluster_b"]}</b><br>'
                f'Construct pairs: {e["total_construct_pairs"]}<br>'
                f'Source scales: {e["n_scales"]} → {signal}<br>'
                f'Top scale: {e["dominant_scale"]} ({e["dominant_scale_share"]:.0%})'
                '<extra></extra>'
            ),
            showlegend=False
        ))

    # 2. Construct nodes (by super-cluster, same as Tab 1)
    for sc_id, sc_name in super_clusters.items():
        mask = df['super_cluster_id'] == sc_id
        subset = df[mask]
        fig.add_trace(go.Scatter(
            x=subset['umap_x'], y=subset['umap_y'],
            mode='markers', name=sc_name,
            marker=dict(size=subset['freq_scaled'],
                        color=super_cluster_colors[sc_id], opacity=0.8),
            customdata=subset[['construct_id', 'construct_name',
                                'super_cluster_name', 'cluster_id',
                                'cluster_peer_count', 'frequency', 'first_year']].values,
            hovertemplate=(
                '<b>%{customdata[1]}</b><br>'
                'Super-cluster: %{customdata[2]}<br>'
                'Micro-cluster: #%{customdata[3]} (%{customdata[4]} peers)<br>'
                'Frequency: %{customdata[5]} scales<br>'
                'First appeared: %{customdata[6]}'
                '<extra></extra>'
            )
        ))

    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='top', y=-0.02),
        margin=dict(l=20, r=20, t=20, b=60), height=520
    )
    return fig


def build_ego_view(selected_id, df, construct_edges, super_cluster_colors):
    fig = go.Figure()
    selected = df[df['construct_id'] == selected_id].iloc[0]
    cluster_peers = df[df['cluster_id'] == selected['cluster_id']]
    sel_edges = construct_edges[
        (construct_edges['construct_a'] == selected_id) |
        (construct_edges['construct_b'] == selected_id)
    ]
    co_measured_ids = set(
        sel_edges['construct_b'].where(sel_edges['construct_a'] == selected_id,
        sel_edges['construct_a'])
    )

    # Edges from selected construct
    for _, e in sel_edges.iterrows():
        other_id = e['construct_b'] if e['construct_a'] == selected_id else e['construct_a']
        other = df[df['construct_id'] == other_id].iloc[0]
        fig.add_trace(go.Scatter(
            x=[selected['umap_x'], other['umap_x'], None],
            y=[selected['umap_y'], other['umap_y'], None],
            mode='lines', line=dict(color='rgba(186,117,23,0.65)', width=2),
            hoverinfo='skip', showlegend=False
        ))

    # Dimmed nodes
    active_ids = co_measured_ids | set(cluster_peers['construct_id']) | {selected_id}
    dimmed = df[~df['construct_id'].isin(active_ids)]
    fig.add_trace(go.Scatter(x=dimmed['umap_x'], y=dimmed['umap_y'], mode='markers',
        marker=dict(size=2, color='rgba(150,150,150,0.08)'),
        hoverinfo='skip', showlegend=False))

    # Co-measured neighbors (green)
    co_only = co_measured_ids - set(cluster_peers['construct_id']) - {selected_id}
    co_df = df[df['construct_id'].isin(co_only)]
    fig.add_trace(go.Scatter(x=co_df['umap_x'], y=co_df['umap_y'], mode='markers',
        name=f'Co-measured ({len(co_df)})',
        marker=dict(size=8, color='#1D9E75', opacity=0.85),
        hovertemplate='<b>%{customdata[0]}</b><br>Co-measured<extra></extra>',
        customdata=co_df[['construct_name']].values))

    # Cluster peers (purple)
    peers = cluster_peers[cluster_peers['construct_id'] != selected_id]
    fig.add_trace(go.Scatter(x=peers['umap_x'], y=peers['umap_y'], mode='markers',
        name=f'Same cluster ({len(peers)})',
        marker=dict(size=8, color='#7F77DD', opacity=0.85),
        hovertemplate='<b>%{customdata[0]}</b><br>Same cluster<extra></extra>',
        customdata=peers[['construct_name']].values))

    # Selected node (red)
    fig.add_trace(go.Scatter(
        x=[selected['umap_x']], y=[selected['umap_y']], mode='markers+text',
        name='Selected', marker=dict(size=14, color='#E24B4A'),
        text=[selected['construct_name']], textposition='top center',
        hovertemplate='<b>%{text}</b><extra></extra>'))

    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='top', y=-0.02),
        margin=dict(l=20, r=20, t=20, b=60), height=520
    )
    return fig
```

### 5.7 Click Handling for Mode Switching

```python
# In Streamlit
selected_construct = st.session_state.get('selected_construct', None)

if selected_construct is not None:
    fig = build_ego_view(selected_construct, df, construct_edges, super_cluster_colors)
else:
    fig = build_overview(df, cluster_edge_df, sc_centroids, super_cluster_colors)

event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", key="tab3_chart")

if event and event.selection and event.selection.points:
    clicked_id = event.selection.points[0].customdata[0]  # construct_id
    if clicked_id == selected_construct:
        st.session_state.selected_construct = None  # toggle off
    else:
        st.session_state.selected_construct = clicked_id
    st.rerun()

if selected_construct is not None:
    if st.button("← Back to overview"):
        st.session_state.selected_construct = None
        st.rerun()
```

### 5.8 Streamlit Controls

| Control | Streamlit Widget | Default | Function |
|---------|-----------------|---------|----------|
| Min construct pairs | `st.slider("Min construct pairs per cluster edge", 10, 200, 50)` | 50 | Filter weak cluster-pair edges |
| Show dominated edges | `st.checkbox("Show single-scale dominated edges", False)` | Off | Toggle dashed dominated edges |
| Super-cluster filter | `st.selectbox("Focus super-cluster", ["All"] + sc_names)` | All | Dim all other clusters |

### 5.9 Side Panel: Cluster Breakdown

Display alongside the chart using `st.columns([3, 1])`:

**Global overview (no selection):** Top-10 cluster pairs ranked by `n_scales` (consensus strength), with a mini bar chart showing `total_construct_pairs` and `n_scales` side by side.

**Ego mode (construct selected):** Ranked list of partner clusters for the selected construct's super-cluster, showing `total_construct_pairs` and `n_scales` per partner. Each row colored by the partner cluster's color.

### 5.10 Summary Metrics

| Metric | Mode | Description |
|--------|------|-------------|
| Cross-cluster edges shown | Overview | Count of cluster-pair edges after filtering |
| Strong consensus edges | Overview | Edges with `n_scales >= 8` |
| Dominated edges | Overview | Edges with `dominant_scale_share >= 0.5` (flagged ⚠️) |
| Cluster peers | Ego | Count of same-micro-cluster constructs |
| Co-measurement degree | Ego | Number of co-measured construct neighbors |
| Cross-cluster co-measurement | Ego | Co-measured neighbors in different super-clusters |

---

## 6. Cross-Tab Design Specifications

### 6.1 Consistent Color Palette

**Super-cluster palette** (used in Tab 1 and Tab 3): 10–15 high-contrast colors. Must be consistent across both tabs so users can mentally track clusters.

```python
# Example 12-color palette (finalize after actual clustering)
SUPER_CLUSTER_COLORS = {
    0: '#7F77DD',   # label TBD after clustering
    1: '#1D9E75',
    2: '#D85A30',
    3: '#D4537E',
    4: '#378ADD',
    5: '#639922',
    6: '#BA7517',
    7: '#E24B4A',
    8: '#534AB7',
    9: '#0F6E56',
    10: '#993C1D',
    11: '#854F0B',
}
```

**Scale palette** (used in Tab 2 only): 32 colors, evenly spaced in HSL hue wheel, high saturation (85%) for maximum edge contrast against gray nodes.

### 6.2 Consistent Hover Tooltip Structure

All tabs display construct information consistently:

```
[Construct Name]                    (bold)
Super-cluster: [Name]              (with colored dot)
Micro-cluster: #[ID] ([N] peers)
Frequency: [N] scales
First appeared: [Year]
```

Tab 2 additionally shows: `Scales: [Scale 1], [Scale 5], [Scale 12]` (each colored)
Tab 3 (ego mode) additionally shows: `Co-measurement edges: [N] ([M] cross-cluster)`

### 6.3 Streamlit Page Layout

```
st.set_page_config(layout="wide")
st.title("Corpus Overview")

tab1, tab2, tab3 = st.tabs([
    "🗺️ Semantic Map",
    "🔬 Measurement Decomposition",
    "🔗 Integrated View"
])

with tab1:
    # Summary metrics row (st.columns)
    # Controls (st.sidebar or inline)
    # Plotly chart
    pass

with tab2:
    # Controls (inline above chart)
    # Cytoscape HTML component (st.components.v1.html, height=550)
    # Detail panel (part of HTML component)
    pass

with tab3:
    # Summary metrics row
    # Controls (inline)
    # Plotly chart with click handling
    pass
```

---

## 7. Performance Requirements

| Metric | Target |
|--------|--------|
| Tab 1 (Plotly scatter, 400 points) | Initial render ≤ 2s |
| Tab 2 (Cytoscape, 400 nodes + edges) | Layout + render ≤ 5s |
| Tab 2 overlap removal | ≤ 1s post-layout |
| Tab 2 node tap response | ≤ 200ms (class toggling only, no re-layout) |
| Tab 3 (Plotly, 400 points + edge lines) | Initial render ≤ 3s |
| Tab 3 ego-mode switch | ≤ 1s (Streamlit rerun + re-render) |
| Tab switching | ≤ 500ms |

---

## 8. Acceptance Criteria

### 8.1 Tab 1

- [ ] 400 construct nodes render correctly at UMAP coordinates
- [ ] Super-cluster colors match the shared palette
- [ ] Node size scales with frequency
- [ ] Hover tooltip shows all specified fields
- [ ] Super-cluster filter correctly shows/hides nodes
- [ ] Color-by toggle switches between cluster color and frequency heatmap

### 8.2 Tab 2

- [ ] 400 nodes render with gray fill, size proportional to scale count
- [ ] Force-directed layout produces readable graph (no severe clumping)
- [ ] Overlap removal post-processing eliminates node overlap
- [ ] Clicking a node: all non-related nodes dim to near-invisible
- [ ] Clicking a node: per-scale edges appear in distinct high-saturation colors
- [ ] Per-scale edges are visually distinguishable (curved with offset to avoid overlap)
- [ ] Target nodes take on the color of the connecting scale edge
- [ ] Target nodes connected by 2+ scales show dual-color encoding (fill + border)
- [ ] Detail panel below chart lists per-scale co-measurement partners
- [ ] Detail panel shows cross-scale partners (constructs co-measured in 2+ scales)
- [ ] Clicking background resets to default state
- [ ] Min co-measurement slider filters edges and triggers re-layout
- [ ] Year filter correctly filters scales and updates the network

### 8.3 Tab 3

- [ ] Default mode (A): all 400 construct nodes rendered at UMAP coordinates with super-cluster colors
- [ ] Cluster-pair edges rendered as lines connecting super-cluster centroids (not individual construct nodes)
- [ ] Edge width correctly encodes `total_construct_pairs` (thicker = more co-measured pairs)
- [ ] Edge opacity correctly encodes `n_scales` (more opaque = broader consensus)
- [ ] Single-scale-dominated edges (`dominant_scale_share >= 0.5`) rendered as dashed lines
- [ ] Hover on cluster-pair edge shows: cluster pair IDs, construct pair count, source scale count, signal type, top contributing scale and its share
- [ ] "Show dominated edges" checkbox correctly toggles dashed edges on/off (default: off)
- [ ] Min construct pairs slider correctly filters cluster-pair edges
- [ ] Clicking a construct node activates ego-network mode (C)
- [ ] Ego mode: selected node red, same-cluster peers purple, co-measured construct neighbors green, rest dimmed
- [ ] Ego mode: construct-level edges from selected node shown in amber; cluster-pair edges hidden
- [ ] Side panel updates to show partner-cluster breakdown in ego mode
- [ ] Clicking empty space or "← Back to overview" returns to mode A
- [ ] Summary metrics (strong consensus count, dominated edge count) update correctly

### 8.4 Cross-Tab

- [ ] Super-cluster color palette identical in Tab 1 and Tab 3
- [ ] Hover tooltip structure consistent across all tabs
- [ ] Tab switching works without state leakage
- [ ] Page renders correctly at 1200px+ width

---

## 9. File Structure (Suggested)

```
corpus_overview/
├── app.py                          # Main Streamlit page
├── tabs/
│   ├── tab1_semantic_map.py        # Tab 1 rendering logic
│   ├── tab2_measurement.py         # Tab 2 rendering logic (generates Cytoscape HTML)
│   ├── tab3_integrated.py          # Tab 3 rendering logic
│   └── cytoscape_template.html     # Cytoscape.js HTML template for Tab 2
├── data/
│   ├── preprocess.py               # Data preprocessing pipeline
│   ├── constructs.csv              # Construct metadata + UMAP coords + cluster assignments
│   ├── scales.csv                  # Scale metadata
│   ├── scale_construct_map.csv     # Scale-construct membership
│   ├── co_measurement_edges.csv    # Precomputed construct-level edge list
│   ├── cluster_edges.csv           # Precomputed cluster-pair aggregated edge list (Tab 3 overview)
│   ├── super_clusters.json         # Super-cluster definitions + color palette
│   └── scale_colors.json           # Scale color palette
├── utils/
│   ├── colors.py                   # Palette definitions
│   ├── backbone.py                 # Serrano's disparity filter implementation
│   └── clustering.py               # Super-cluster generation
└── requirements.txt                # streamlit, plotly, pandas, scipy, scikit-learn
```
