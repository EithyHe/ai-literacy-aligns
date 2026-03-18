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
| Semantic clusters | ~15 | Construct-level semantic clusters from embedding clustering |

### 1.3 Three-Tab Architecture

The page has three tabs, each with a distinct analytical focus:

| Tab | Name | Core Question | Primary Perspective |
|-----|------|---------------|-------------------|
| Tab 1 | Semantic Map | How do constructs distribute in **meaning** space? | Semantic structure |
| Tab 2 | Measurement Decomposition | How do different scales **decompose** the AI literacy domain? | Measurement practice |
| Tab 3 | Integrated View | Where do semantic structure and measurement practice **align or diverge**? | Semantic × Measurement |

**Progression logic:** meaning → measurement → meaning × measurement

### 1.4 Cluster Visual Encoding Strategy

With ~15 semantic clusters, discrete color encoding works directly — human vision reliably distinguishes 12–15 colors. The strategy:

1. **Color encodes cluster** (15 discrete colors) in Tab 1 and Tab 3
2. **Spatial proximity** reinforces cluster structure (UMAP co-locates same-cluster nodes)
3. **Hover tooltips** show cluster name and peer count for detail

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
| Cluster assignments | CSV/DataFrame | `construct_id`, `cluster_id`, `cluster_name` | Tab 1, Tab 3 |
| Cluster color palette | Dict/JSON | `cluster_id` → hex color (15 entries) | Tab 1, Tab 3 |
| Scale metadata | CSV/DataFrame | `scale_id`, `scale_name`, `year`, `author`, `n_constructs` | Tab 2 |
| Scale–construct membership | CSV/DataFrame | `scale_id`, `construct_id` | Tab 2, Tab 3 |
| Co-measurement edge list | CSV/DataFrame | `construct_a`, `construct_b`, `shared_scale_count`, `jaccard_weight`, `first_coappearance_year`, `shared_scale_ids` (list) | Tab 2, Tab 3 |
| Scale color palette | Dict/JSON | `scale_id` → hex color (32 entries, high-saturation HSL, evenly spaced hue) | Tab 2 |

### 2.4 Data Preprocessing Pipeline

#### 2.4.1 Co-Measurement Edge Computation

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

#### 2.4.2 Backbone Filtering (for Tab 3)

Use Serrano's disparity filter to extract statistically significant edges. Pre-compute backbone at multiple alpha thresholds (0.01, 0.05, 0.10, 0.20) so the user can adjust filtering strength interactively without recomputation.

#### 2.4.3 Scale Color Palette Generation

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
| Point color | Cluster membership (15 discrete colors from palette) |
| Point size | Frequency (number of scales measuring this construct) — `mapData(freq, 1, max_freq, 5, 18)` |
| Hover tooltip | Construct name, cluster name, cluster peer count, frequency, first year |

### 3.3 Plotly Implementation Guidance

```python
import plotly.graph_objects as go

fig = go.Figure()

for cl_id, cl_name in clusters.items():
    mask = df['cluster_id'] == cl_id
    subset = df[mask]
    fig.add_trace(go.Scatter(
        x=subset['umap_x'],
        y=subset['umap_y'],
        mode='markers',
        name=cl_name,
        marker=dict(
            size=subset['freq_scaled'],  # pre-scaled to 5–18 range
            color=cluster_colors[cl_id],
            opacity=0.75,
            line=dict(width=0.5, color='rgba(255,255,255,0.3)')
        ),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'Cluster: %{customdata[1]}<br>'
            'Cluster peers: %{customdata[2]}<br>'
            'Frequency: %{customdata[3]} scales<br>'
            'First appeared: %{customdata[4]}'
            '<extra></extra>'
        ),
        customdata=subset[['construct_name', 'cluster_name', 'cluster_peer_count', 'frequency', 'first_year']].values
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
| Cluster filter | `st.multiselect("Clusters", all_names, default=all_names)` | Show/hide specific clusters (filter traces) |
| Color by | `st.radio("Color by", ["Cluster", "Frequency heatmap"])` | Toggle between discrete cluster color and continuous frequency colorscale |
| Point size | `st.slider("Point size", 1, 3, 2)` | Global size multiplier |

### 3.5 Summary Metrics

Display above the chart using `st.columns(4)`:

| Metric | Value |
|--------|-------|
| Total constructs | `len(df)` |
| Clusters | `df['cluster_id'].nunique()` |
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

Overlay co-measurement edges onto the semantic map (UMAP layout), revealing where measurement practice aligns with or diverges from semantic structure.

**Core questions this tab answers:**
- Which constructs are semantically distant but frequently co-measured? (Long amber edges crossing clusters — researchers view them as complementary despite semantic distinctness)
- Which semantically similar constructs have never been co-measured? (Close nodes with no edge — potential measurement blind spots)
- For a specific construct: how does its semantic neighborhood differ from its measurement neighborhood?

### 5.2 Dual-Mode Interaction

Tab 3 has two interaction modes that the user switches between by clicking:

#### Mode A: Global Overview (default)

All 400 nodes visible with cluster colors. Co-measurement edges shown with backbone filtering. Cross-cluster edges visually highlighted.

| Visual Element | Encoding |
|----------------|----------|
| Node position | UMAP coordinates |
| Node color | Cluster (15 discrete colors, same palette as Tab 1) |
| Node size | Frequency |
| Edge (within cluster) | Very faint gray: `rgba(0,0,0,0.03)` light mode, `rgba(255,255,255,0.04)` dark mode, width 0.4 |
| Edge (across clusters) | Amber highlight: `rgba(186,117,23,0.18)` light mode, `rgba(239,159,39,0.22)` dark mode, width = `0.5 + shared_scale_count * 0.5` (max 2.5) |

#### Mode C: Ego-Network Focus (on click)

When the user clicks a specific construct node:

| Visual Element | Encoding |
|----------------|----------|
| Selected node | Large (radius 6), red (`#E24B4A`), labeled |
| Same cluster peers | Medium (radius 4), purple (`#7F77DD`) — semantic neighbors |
| Co-measured neighbors | Medium (radius 4), green (`#1D9E75`) — measurement neighbors |
| All other nodes | Tiny (radius 1.5), very light gray — dimmed |
| Edges from selected node | Amber, width 1.5, opacity 0.5 |
| All other edges | Hidden |
| Dynamic legend | Shows selected node name, cluster peer count, co-measurement neighbor count |

**Exit:** Click empty canvas area to return to Mode A.

### 5.3 Plotly Implementation Guidance

```python
import plotly.graph_objects as go

# Mode A: Global overview
def build_overview(df, edges, cluster_colors, edge_threshold):
    fig = go.Figure()

    # 1. Within-cluster edges (faint gray)
    within_edges = edges[
        (edges['shared_scale_count'] >= edge_threshold) &
        (edges['source_cluster'] == edges['target_cluster'])
    ]
    for _, e in within_edges.iterrows():
        fig.add_trace(go.Scatter(
            x=[e['source_umap_x'], e['target_umap_x'], None],
            y=[e['source_umap_y'], e['target_umap_y'], None],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.03)', width=0.4),
            hoverinfo='skip',
            showlegend=False
        ))

    # 2. Cross-cluster edges (amber highlight)
    cross_edges = edges[
        (edges['shared_scale_count'] >= edge_threshold) &
        (edges['source_cluster'] != edges['target_cluster'])
    ]
    for _, e in cross_edges.iterrows():
        fig.add_trace(go.Scatter(
            x=[e['source_umap_x'], e['target_umap_x'], None],
            y=[e['source_umap_y'], e['target_umap_y'], None],
            mode='lines',
            line=dict(color='rgba(186,117,23,0.18)', width=min(0.5 + e['shared_scale_count'] * 0.5, 2.5)),
            hoverinfo='skip',
            showlegend=False
        ))

    # 3. Construct nodes (by cluster)
    for cl_id, cl_name in clusters.items():
        mask = df['cluster_id'] == cl_id
        subset = df[mask]
        fig.add_trace(go.Scatter(
            x=subset['umap_x'], y=subset['umap_y'],
            mode='markers', name=cl_name,
            marker=dict(size=subset['freq_scaled'], color=cluster_colors[cl_id], opacity=0.8),
            customdata=subset[['construct_id', 'construct_name', ...]].values,
            hovertemplate='<b>%{customdata[1]}</b><br>...<extra></extra>'
        ))

    return fig


# Mode C: Ego-network focus
def build_ego_view(selected_id, df, edges):
    fig = go.Figure()

    selected = df[df['construct_id'] == selected_id].iloc[0]
    cluster_peers = df[df['cluster_id'] == selected['cluster_id']]
    co_measured = get_co_measured_neighbors(selected_id, edges)  # returns set of construct_ids

    # Edges from selected node
    sel_edges = edges[(edges['construct_a'] == selected_id) | (edges['construct_b'] == selected_id)]
    for _, e in sel_edges.iterrows():
        other_id = e['construct_b'] if e['construct_a'] == selected_id else e['construct_a']
        other = df[df['construct_id'] == other_id].iloc[0]
        fig.add_trace(go.Scatter(
            x=[selected['umap_x'], other['umap_x'], None],
            y=[selected['umap_y'], other['umap_y'], None],
            mode='lines', line=dict(color='rgba(186,117,23,0.5)', width=1.5),
            hoverinfo='skip', showlegend=False
        ))

    # Dimmed nodes
    dimmed = df[~df['construct_id'].isin(co_measured | set(cluster_peers['construct_id']) | {selected_id})]
    fig.add_trace(go.Scatter(x=dimmed['umap_x'], y=dimmed['umap_y'], mode='markers',
        marker=dict(size=2, color='rgba(150,150,150,0.08)'), hoverinfo='skip', showlegend=False))

    # Co-measured neighbors (green)
    co_df = df[df['construct_id'].isin(co_measured - set(cluster_peers['construct_id']) - {selected_id})]
    fig.add_trace(go.Scatter(x=co_df['umap_x'], y=co_df['umap_y'], mode='markers',
        name=f'Co-measured ({len(co_df)})', marker=dict(size=8, color='#1D9E75', opacity=0.8),
        hovertemplate='<b>%{customdata[0]}</b><br>Co-measured<extra></extra>',
        customdata=co_df[['construct_name']].values))

    # Cluster peers (purple)
    peers = cluster_peers[cluster_peers['construct_id'] != selected_id]
    fig.add_trace(go.Scatter(x=peers['umap_x'], y=peers['umap_y'], mode='markers',
        name=f'Same cluster ({len(peers)})', marker=dict(size=8, color='#7F77DD', opacity=0.8),
        hovertemplate='<b>%{customdata[0]}</b><br>Same cluster<extra></extra>',
        customdata=peers[['construct_name']].values))

    # Selected node (red)
    fig.add_trace(go.Scatter(x=[selected['umap_x']], y=[selected['umap_y']], mode='markers+text',
        name='Selected', marker=dict(size=14, color='#E24B4A'),
        text=[selected['construct_name']], textposition='top center',
        hovertemplate='<b>%{text}</b><extra></extra>'))

    return fig
```

### 5.4 Click Handling for Mode Switching

```python
# In Streamlit
selected_construct = st.session_state.get('selected_construct', None)

if selected_construct is not None:
    fig = build_ego_view(selected_construct, df, edges)
else:
    fig = build_overview(df, edges, cluster_colors, edge_threshold)

# Capture click events
event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", key="tab3_chart")

if event and event.selection and event.selection.points:
    clicked_id = event.selection.points[0].customdata[0]  # construct_id
    if clicked_id == selected_construct:
        st.session_state.selected_construct = None  # toggle off
    else:
        st.session_state.selected_construct = clicked_id
    st.rerun()

# Reset button
if selected_construct is not None:
    if st.button("← Back to overview"):
        st.session_state.selected_construct = None
        st.rerun()
```

### 5.5 Streamlit Controls

| Control | Streamlit Widget | Function |
|---------|-----------------|----------|
| Edge threshold | `st.slider("Min shared scales", 1, 5, 1)` | Filter co-measurement edges by shared_scale_count |
| Cross-cluster highlight | `st.checkbox("Highlight cross-cluster edges", True)` | Toggle amber highlight for cross-cluster edges |
| Cluster filter | `st.selectbox("Focus cluster", ["All"] + cluster_names)` | Dim all other clusters |
| Backbone alpha | `st.select_slider("Edge filtering", [0.01, 0.05, 0.10, 0.20], 0.10)` | Control backbone filtering stringency |

### 5.6 Summary Metrics

| Metric | Description |
|--------|-------------|
| Total co-measurement edges | After threshold + backbone filtering |
| Cross-cluster edges | Count and percentage of total |
| Avg UMAP edge length | Mean Euclidean distance of co-measurement edges in UMAP space |
| (Ego mode) Cluster peers | Count of same-cluster constructs |
| (Ego mode) Co-measurement degree | Number of co-measured neighbors |
| (Ego mode) Cross-cluster co-measurement | Count of co-measured neighbors in different clusters |

---

## 6. Cross-Tab Design Specifications

### 6.1 Consistent Color Palette

**Cluster palette** (used in Tab 1 and Tab 3): 15 high-contrast colors. Must be consistent across both tabs so users can mentally track clusters.

```python
# Example 15-color palette (finalize after actual clustering)
CLUSTER_COLORS = {
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
    12: '#2E8B84',
    13: '#A0522D',
    14: '#6B5B95',
}
```

**Scale palette** (used in Tab 2 only): 32 colors, evenly spaced in HSL hue wheel, high saturation (85%) for maximum edge contrast against gray nodes.

### 6.2 Consistent Hover Tooltip Structure

All tabs display construct information consistently:

```
[Construct Name]                    (bold)
Cluster: [Name]                    (with colored dot)
Cluster peers: [N]
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
- [ ] Cluster colors match the shared palette (15 colors)
- [ ] Node size scales with frequency
- [ ] Hover tooltip shows all specified fields
- [ ] Cluster filter correctly shows/hides nodes
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

- [ ] Default mode (A): all 400 nodes with cluster colors at UMAP coordinates
- [ ] Cross-cluster edges highlighted in amber, within-cluster edges faint gray
- [ ] Edge threshold slider correctly filters edges
- [ ] Clicking a node activates ego-network mode (C)
- [ ] Ego mode: selected node red, cluster peers purple, co-measured neighbors green, rest dimmed
- [ ] Ego mode: only edges from selected node shown in amber
- [ ] Clicking empty space or "Back to overview" returns to mode A
- [ ] Summary metrics update correctly in both modes

### 8.4 Cross-Tab

- [ ] Cluster color palette identical in Tab 1 and Tab 3
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
│   ├── co_measurement_edges.csv    # Precomputed edge list
│   ├── cluster_palette.json        # Cluster definitions + color palette (15 entries)
│   └── scale_colors.json           # Scale color palette (32 entries)
├── utils/
│   ├── colors.py                   # Palette definitions
│   └── backbone.py                 # Serrano's disparity filter implementation
└── requirements.txt                # streamlit, plotly, pandas, scipy, scikit-learn
```
