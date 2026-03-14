"""Corpus Overview – Framework UMAP + Construct network drill-down (all client-side)."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from ssn.services.graph_data_service import get_graph_data
from ssn.components.zmlt_plotly import (
    build_all_frameworks_umap_figure_for_client,
    _framework_color,
)
from ssn.services.embedding_service import get_construct_embeddings
from ssn.services.network_service import build_construct_network
from ssn.services.network_service import build_construct_network
from ssn.db.schema import (
    get_all_constructs,
    get_all_frameworks,
    get_constructs_by_domain,
    get_domains_by_framework,
)


def _construct_id_to_name() -> dict[str, str]:
    return {c["construct_id"]: c.get("name", c["construct_id"]) for c in get_all_constructs()}


def _filtered_embeddings(framework_id: str | None, domain_id: str | None) -> dict:
    from ssn.db.schema import get_constructs_by_framework_id

    all_embs = get_construct_embeddings()
    if not all_embs:
        return {}
    if domain_id:
        valid = {c["construct_id"] for c in get_constructs_by_domain(domain_id) if c.get("construct_id")}
        return {k: v for k, v in all_embs.items() if k in valid}
    if framework_id:
        valid = {c["construct_id"] for c in get_constructs_by_framework_id(framework_id)}
        return {k: v for k, v in all_embs.items() if k in valid}
    return all_embs


def _convex_hull_2d(x: list[float], y: list[float]) -> list[tuple[float, float]]:
    """Return convex hull vertices (x,y) for 2D points."""
    if len(x) < 3:
        return []
    try:
        from scipy.spatial import ConvexHull
        pts = np.column_stack([x, y])
        hull = ConvexHull(pts)
        return [(float(pts[i, 0]), float(pts[i, 1])) for i in hull.vertices]
    except Exception:
        return []


def _expand_hull(
    hull: list[tuple[float, float]],
    centroid: tuple[float, float],
    padding: float = 0.35,
) -> list[tuple[float, float]]:
    """Expand hull vertices outward from centroid by (1 + padding)."""
    if not hull:
        return []
    cx, cy = centroid
    return [
        (cx + (hx - cx) * (1 + padding), cy + (hy - cy) * (1 + padding))
        for hx, hy in hull
    ]


def _prepare_client_data(
    frameworks: list[dict],
    graph_data: dict,
) -> dict:
    """Precompute JSON for both levels: centroid, hull, constructs (id, name, x, y), edges per framework."""
    nodes = graph_data.get("nodes") or []
    fw_by_id = {f["framework_id"]: f for f in frameworks}
    id_to_name = _construct_id_to_name()

    constructs_by_fw: dict[str, list[dict]] = {}
    for n in nodes:
        if n.get("level") != "construct" or "x" not in n or "y" not in n:
            continue
        fid = n.get("framework_id")
        if not fid or fid not in fw_by_id:
            continue
        constructs_by_fw.setdefault(fid, []).append(n)

    out_frameworks: list[dict] = []
    for fw in frameworks:
        fid = fw["framework_id"]
        name = fw.get("name", fid)
        construct_nodes = constructs_by_fw.get(fid, [])
        if not construct_nodes:
            continue

        color = _framework_color(fid)
        xs = [float(n["x"]) for n in construct_nodes]
        ys = [float(n["y"]) for n in construct_nodes]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        hull_pts = _convex_hull_2d(xs, ys)
        if len(hull_pts) >= 3:
            hull_pts = _expand_hull(hull_pts, (cx, cy), 0.35)

        embs = _filtered_embeddings(fid, None)
        # 仅对有 embedding 的 construct 生成节点与边，保证两两余弦相似度与连线一一对应
        construct_ids_with_emb = set(embs.keys()) if embs else set()
        construct_nodes_filtered = [n for n in construct_nodes if n.get("id") in construct_ids_with_emb]
        constructs = [
            {
                "id": n["id"],
                "name": id_to_name.get(n["id"], n.get("label", n["id"])),
                "x": float(n["x"]),
                "y": float(n["y"]),
            }
            for n in construct_nodes_filtered
        ]

        edges: list[dict] = []
        if embs:
            G = build_construct_network(embs, use_tmfg=True)
            for u, v, data in G.edges(data=True):
                w = float(data.get("weight", 0.5))
                edges.append({"source": u, "target": v, "weight": round(w, 4)})

        out_frameworks.append({
            "id": fid,
            "name": name,
            "color": color,
            "centroid": {"x": cx, "y": cy},
            "hull": [{"x": hx, "y": hy} for hx, hy in hull_pts],
            "constructs": constructs,
            "edges": edges,
        })

    return {"frameworks": out_frameworks}


def _safe_json_dumps(obj) -> str:
    """JSON that is safe to embed in HTML/JS (no NaN/Infinity, escape </script>)."""
    def default(o):
        if isinstance(o, (np.floating, float)):
            f = float(o)
            if np.isnan(f) or np.isinf(f):
                return 0.0
            return f
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o).__name__)
    s = json.dumps(obj, default=default, ensure_ascii=False)
    return s.replace("</script>", "<\\/script>")


def _build_client_html(
    fig_dict: dict,
    trace_map: list[dict],
    detail_data: dict,
) -> str:
    """Generate HTML with Plotly + d3-force, state machine OVERVIEW | HIGHLIGHTED | DETAIL."""
    fig_json = _safe_json_dumps(fig_dict)
    map_json = _safe_json_dumps(trace_map)
    detail_json = _safe_json_dumps(detail_data)

    legend_rows = []
    for i, fw in enumerate(trace_map):
        fid = fw["id"]
        name = html.escape(str(fw["name"]))
        count = fw["count"]
        color = fw["color"]
        legend_rows.append(
            f'<div class="umap-legend-row" data-fid="{html.escape(str(fid))}" data-fw-index="{i}" role="button" tabindex="0">'
            f'<span class="umap-legend-bar"></span>'
            f'<span class="umap-legend-dot" style="background:{color}"></span>'
            f'<span class="umap-legend-label">{name} ({count})</span>'
            f"</div>"
        )
    legend_html = "".join(legend_rows)

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    .umap-layout {{ display: flex; gap: 12px; width: 100%; min-height: 600px; }}
    .umap-chart-wrap {{ flex: 1; min-width: 0; }}
    .umap-legend {{ flex: 0 0 160px; max-height: 420px; overflow-y: auto; font-size: 12px; }}
    .umap-legend-title {{ font-weight: 600; margin-bottom: 4px; }}
    .umap-legend-caption {{ color: #888; font-size: 11px; margin-bottom: 8px; }}
    .umap-legend-row {{ display: flex; align-items: center; height: 34px; padding: 0 6px; cursor: pointer; border-radius: 4px; transition: background 0.2s; }}
    .umap-legend-row:hover {{ background: rgba(0,0,0,0.04); }}
    .umap-legend-bar {{ width: 18px; display: flex; align-items: center; flex-shrink: 0; }}
    .umap-legend-bar::before {{ content: ''; width: 3px; height: 100%; background: transparent; }}
    .umap-legend-row.active .umap-legend-bar::before {{ background: var(--fw-color); }}
    .umap-legend-dot {{ width: 12px; height: 12px; border-radius: 2px; margin-left: 2px; flex-shrink: 0; }}
    .umap-legend-label {{ margin-left: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #666; transition: color 0.2s, font-weight 0.2s; }}
    .umap-legend-row.active .umap-legend-label {{ font-weight: 700; color: var(--fw-color); }}
    #detail-nav {{ display: none; align-items: center; gap: 12px; margin-bottom: 8px; flex-wrap: wrap; }}
    #detail-nav.visible {{ display: flex; }}
    #detail-nav .nav-back {{ cursor: pointer; color: #1a73e8; font-size: 14px; }}
    #detail-nav .nav-back:hover {{ text-decoration: underline; }}
    #detail-nav .nav-title {{ font-weight: 700; font-size: 14px; flex: 1; }}
    #detail-nav .nav-slider-wrap {{ display: flex; align-items: center; gap: 8px; font-size: 12px; color: #666; }}
    #detail-nav input[type="range"] {{ width: 120px; }}
  </style>
</head>
<body>
  <div id="detail-nav">
    <a class="nav-back" id="nav-back">← Frameworks</a>
    <span class="nav-title" id="nav-title"></span>
    <div class="nav-slider-wrap">
      <label>Min weight:</label>
      <input type="range" id="threshold-slider" min="0" max="0.9" step="0.05" value="0.1">
      <span id="threshold-value">0.10</span>
    </div>
  </div>
  <div class="umap-layout">
    <div class="umap-chart-wrap"><div id="umap-chart"></div></div>
    <div class="umap-legend" id="umap-legend">
      <div class="umap-legend-title">Framework</div>
      <div class="umap-legend-caption">点击展开/收起</div>
      <div id="umap-legend-list">{legend_html}</div>
    </div>
  </div>
  <script>
(function() {{
  var fig, traceMap, detailData, gd, navEl, legendEl, thresholdSlider, thresholdValue, navTitle;
  var state = 'OVERVIEW';
  var activeId = null;
  var detailFrameworkId = null;
  var detailThreshold = 0.1;
  var detailLayout = null;

  try {{
    fig = {fig_json};
    traceMap = {map_json};
    detailData = {detail_json};
  }} catch (e) {{
    console.error('Corpus Overview: invalid embedded data', e);
    return;
  }}
  gd = document.getElementById('umap-chart');
  navEl = document.getElementById('detail-nav');
  legendEl = document.getElementById('umap-legend');
  thresholdSlider = document.getElementById('threshold-slider');
  thresholdValue = document.getElementById('threshold-value');
  navTitle = document.getElementById('nav-title');
  if (!gd || typeof Plotly === 'undefined') return;

  function applyRestyle(id) {{
    const visibility = fig.data.map(function() {{ return true; }});
    const opacity = fig.data.map(function() {{ return 1; }});
    traceMap.forEach(function(fw) {{
      const idx = fw.traceIndices;
      const isActive = (id !== null && fw.id === id);
      if (id === null) {{
        if (idx.hullFill != null) opacity[idx.hullFill] = 0.07;
        if (idx.hullOutline != null) opacity[idx.hullOutline] = 0.2;
        if (idx.diamond != null) opacity[idx.diamond] = 1;
        if (idx.construct != null) visibility[idx.construct] = false;
        (idx.edges || []).forEach(function(ei) {{ visibility[ei] = false; }});
      }} else {{
        if (idx.hullFill != null) opacity[idx.hullFill] = isActive ? 0.07 : 0.02;
        if (idx.hullOutline != null) opacity[idx.hullOutline] = isActive ? 0.7 : 0.12;
        if (idx.diamond != null) opacity[idx.diamond] = isActive ? 1 : 0.2;
        if (idx.construct != null) visibility[idx.construct] = false;
        (idx.edges || []).forEach(function(ei) {{ visibility[ei] = false; }});
      }}
    }});
    Plotly.restyle(gd, {{ visible: visibility, opacity: opacity }}, null);
    var outlineIndices = [];
    var outlineDash = [];
    var outlineWidth = [];
    var outlineColors = [];
    traceMap.forEach(function(fw) {{
      var i = fw.traceIndices.hullOutline;
      if (i == null) return;
      outlineIndices.push(i);
      var isActive = (id !== null && fw.id === id);
      outlineDash.push(isActive ? 'solid' : 'dot');
      outlineWidth.push(isActive ? 2.5 : 1.5);
      outlineColors.push(fw.color || '#999');
    }});
    if (outlineIndices.length) Plotly.restyle(gd, {{ 'line.dash': outlineDash, 'line.width': outlineWidth, 'line.color': outlineColors }}, outlineIndices);
    const annotations = JSON.parse(JSON.stringify(fig.layout.annotations || []));
    traceMap.forEach(function(fw) {{
      var ai = fw.traceIndices.annotationIndex;
      if (annotations[ai]) annotations[ai].opacity = (fw.id === id || id === null) ? 1 : 0.2;
    }});
    Plotly.relayout(gd, {{ annotations: annotations }}, null);
  }}

  function updateLegendStyles() {{
    traceMap.forEach(function(fw, i) {{
      var row = document.querySelector('.umap-legend-row[data-fw-index="' + i + '"]');
      if (!row) return;
      row.style.setProperty('--fw-color', fw.color);
      row.classList.toggle('active', fw.id === activeId);
    }});
  }}

  document.getElementById('umap-legend-list').addEventListener('click', function(e) {{
    var row = e.target.closest('.umap-legend-row');
    if (!row) return;
    e.preventDefault();
    var fid = row.getAttribute('data-fid');
    if (!fid) return;
    activeId = activeId === fid ? null : fid;
    state = activeId ? 'HIGHLIGHTED' : 'OVERVIEW';
    applyRestyle(activeId);
    updateLegendStyles();
  }});
  updateLegendStyles();

  function runForceLayout(fwData, width, height) {{
    if (typeof d3 === 'undefined' || !d3.forceSimulation) return null;
    var constructs = fwData.constructs || [];
    var edges = (fwData.edges || []).filter(function(e) {{ return e.weight >= detailThreshold; }});
    var nodes = constructs.map(function(c) {{ return {{ id: c.id, name: c.name }}; }});
    var links = edges.map(function(e) {{
      return {{ source: e.source, target: e.target, weight: e.weight }};
    }});
    var simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(function(d) {{ return d.id; }}).distance(function(d) {{ return (1 - d.weight) * 300; }}))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .stop();
    for (var i = 0; i < 300; i++) simulation.tick();
    return {{ nodes: nodes, links: links, fwData: fwData }};
  }}

  function loadD3(callback) {{
    if (typeof d3 !== 'undefined' && d3.forceSimulation) {{ callback(); return; }}
    var s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/d3@7';
    s.onload = callback;
    s.onerror = function() {{ console.error('d3 failed to load'); callback(); }};
    document.head.appendChild(s);
  }}

  function drawDetailView(fwData) {{
    var layout = gd._fullLayout || {{}};
    var width = layout.width || 700;
    var height = layout.height || 600;
    detailLayout = runForceLayout(fwData, width, height);
    if (!detailLayout) return;
    var nodes = detailLayout.nodes;
    const links = detailLayout.links;
    const color = fwData.color || '#666';
    const data = [];
    const minW = links.length ? Math.min.apply(null, links.map(function(l) {{ return l.weight; }})) : 0;
    const maxW = links.length ? Math.max.apply(null, links.map(function(l) {{ return l.weight; }})) : 1;
    links.forEach(function(l) {{
      const w = l.weight;
      const lw = maxW > minW ? 1 + (w - minW) / (maxW - minW) * 3 : 1;
      data.push({{
        x: [l.source.x, l.target.x, null],
        y: [l.source.y, l.target.y, null],
        mode: 'lines',
        line: {{ width: lw, color: '#CCCCCC' }},
        hoverinfo: 'skip',
        showlegend: false
      }});
    }});
    const annotations = links.map(function(l) {{
      const mx = (l.source.x + l.target.x) / 2;
      const my = (l.source.y + l.target.y) / 2;
      return {{ x: mx, y: my, text: l.weight.toFixed(2), showarrow: false, font: {{ size: 10, color: '#999' }}, xref: 'x', yref: 'y' }};
    }});
    data.push({{
      x: nodes.map(function(n) {{ return n.x; }}),
      y: nodes.map(function(n) {{ return n.y; }}),
      mode: 'markers+text',
      text: nodes.map(function(n) {{ return n.name; }}),
      textposition: 'top center',
      textfont: {{ size: 12, color: '#333' }},
      marker: {{ size: 14, color: color, line: {{ width: 1, color: 'white' }} }},
      hoverinfo: 'text',
      hovertext: nodes.map(function(n) {{ return n.name; }}),
      showlegend: false
    }});
    Plotly.newPlot(gd, data, {{
      showlegend: false,
      xaxis: {{ showgrid: true, zeroline: false, showticklabels: false }},
      yaxis: {{ showgrid: true, zeroline: false, showticklabels: false }},
      margin: {{ l: 20, r: 20, t: 20, b: 20 }},
      paper_bgcolor: 'white',
      plot_bgcolor: 'rgba(248,248,248,0.5)',
      height: height,
      annotations: annotations
    }}, {{ responsive: true, displayModeBar: true }});
  }}

  function enterDetail(fid) {{
    var fw = detailData.frameworks.find(function(f) {{ return f.id === fid; }});
    if (!fw) return;
    state = 'DETAIL';
    detailFrameworkId = fid;
    detailThreshold = parseFloat(thresholdSlider.value) || 0.1;
    navEl.classList.add('visible');
    legendEl.style.display = 'none';
    navTitle.textContent = 'Constructs in ' + fw.name;
    thresholdSlider.value = detailThreshold.toFixed(2);
    thresholdValue.textContent = detailThreshold.toFixed(2);
    Plotly.purge(gd);
    loadD3(function() {{ drawDetailView(fw); }});
  }}

  function exitDetail() {{
    state = 'OVERVIEW';
    activeId = null;
    detailFrameworkId = null;
    detailLayout = null;
    navEl.classList.remove('visible');
    legendEl.style.display = '';
    Plotly.purge(gd);
    Plotly.newPlot(gd, fig.data, fig.layout, {{ responsive: true, displayModeBar: true }});
    bindPlotlyClick();
    updateLegendStyles();
  }}

  thresholdSlider.addEventListener('input', function() {{
    detailThreshold = parseFloat(thresholdSlider.value) || 0.1;
    thresholdValue.textContent = detailThreshold.toFixed(2);
    if (state !== 'DETAIL' || !detailFrameworkId) return;
    var fw = detailData.frameworks.find(function(f) {{ return f.id === detailFrameworkId; }});
    if (!fw) return;
    Plotly.purge(gd);
    drawDetailView(fw);
  }});

  document.getElementById('nav-back').addEventListener('click', function(e) {{
    e.preventDefault();
    exitDetail();
  }});

  function pointInPolygon(x, y, hull) {{
    if (!hull || hull.length < 3) return false;
    var n = hull.length;
    var inside = false;
    for (var i = 0, j = n - 1; i < n; j = i++) {{
      var xi = hull[i].x, yi = hull[i].y;
      var xj = hull[j].x, yj = hull[j].y;
      if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) inside = !inside;
    }}
    return inside;
  }}

  function bindPlotlyClick() {{
    if (gd.on && typeof gd.on === 'function') {{
      gd.on('plotly_click', function(data) {{
        var pt = data.points && data.points[0];
        var curve = pt ? pt.curveNumber : -1;
        var fw = activeId ? traceMap.find(function(f) {{ return f.id === activeId; }}) : null;
        var idx = fw ? fw.traceIndices : null;
        var clickable = [idx.hullFill, idx.hullOutline, idx.hullClick, idx.diamond];
        if (idx && idx.construct != null) clickable.push(idx.construct);
        if (idx && idx.edges && idx.edges.length) clickable = clickable.concat(idx.edges);
        clickable = clickable.filter(function(x) {{ return x != null; }});
        var isInClickable = idx && clickable.indexOf(curve) !== -1;
        var fwDetail = activeId ? detailData.frameworks.find(function(f) {{ return f.id === activeId; }}) : null;
        var inHull = !!(pt && fwDetail && fwDetail.hull && pointInPolygon(pt.x, pt.y, fwDetail.hull));
        if (state !== 'HIGHLIGHTED' || !activeId) return;
        if (!pt) return;
        if (!fw) return;
        if (!inHull) return;
        enterDetail(activeId);
      }});
    }}
  }}

  try {{
    Plotly.newPlot(gd, fig.data, fig.layout, {{ responsive: true, displayModeBar: true }});
    bindPlotlyClick();
  }} catch (err) {{
    console.error('Plotly.newPlot failed', err);
  }}
}})();
  </script>
</body>
</html>
"""


def _compute_framework_overlap(
    frameworks: list[dict],
) -> list[tuple[str, str, str, str, float]]:
    """Pairwise mean cosine similarity between framework constructs. Returns (name_a, name_b, color_a, color_b, sim)."""
    all_embs = get_construct_embeddings()
    if not all_embs:
        return []
    fw_by_id = {f["framework_id"]: f for f in frameworks}
    ids_by_fw: dict[str, list[str]] = {}
    for fid in fw_by_id:
        embs = _filtered_embeddings(fid, None)
        ids_by_fw[fid] = [cid for cid in embs if cid in all_embs]
    result: list[tuple[str, str, str, str, float]] = []
    fids = list(fw_by_id.keys())
    for i, fid_a in enumerate(fids):
        for fid_b in fids[i + 1 :]:
            ids_a = ids_by_fw.get(fid_a, [])
            ids_b = ids_by_fw.get(fid_b, [])
            if not ids_a or not ids_b:
                continue
            sims: list[float] = []
            for cid_a in ids_a:
                va = np.array(all_embs[cid_a], dtype=np.float64)
                for cid_b in ids_b:
                    vb = np.array(all_embs[cid_b], dtype=np.float64)
                    sims.append(float(np.dot(va, vb)))
            mean_sim = sum(sims) / len(sims) if sims else 0.0
            name_a = fw_by_id[fid_a].get("name", fid_a)
            name_b = fw_by_id[fid_b].get("name", fid_b)
            result.append((name_a, name_b, _framework_color(fid_a), _framework_color(fid_b), mean_sim))
    result.sort(key=lambda x: -x[4])
    return result


def render() -> None:
    st.title("Corpus Overview")
    frameworks = get_all_frameworks()
    graph_data = get_graph_data()
    get_construct_embeddings()

    if not frameworks or not graph_data:
        st.warning("数据未就绪，请先运行 pipeline。")
        return

    fig_dict, trace_map = build_all_frameworks_umap_figure_for_client(
        graph_data,
        frameworks,
        title="Frameworks in UMAP semantic space",
        height=600,
    )
    detail_data = _prepare_client_data(frameworks, graph_data)
    html_content = _build_client_html(fig_dict, trace_map, detail_data)
    components.html(html_content, height=660, scrolling=False)

    with st.expander("Overlap Index", expanded=False):
        st.caption("Mean pairwise cosine similarity between constructs of each framework pair.")
        try:
            overlap_pairs = _compute_framework_overlap(frameworks)
        except Exception as e:
            st.warning(f"Could not compute overlap: {e}")
            overlap_pairs = []
        for name_a, name_b, color_a, color_b, sim in overlap_pairs:
            pct = sim * 100
            high = sim > 0.5
            pct_style = "font-family:monospace;font-weight:600;color:#B45309" if high else "font-family:monospace;color:#999"
            st.markdown(
                f'<span style="color:{color_a};font-weight:600">●</span> '
                f'<span style="color:#999">×</span> '
                f'<span style="color:{color_b};font-weight:600">●</span> '
                f'<span style="color:#777">{name_a} × {name_b}</span> '
                f'<span style="{pct_style}">{pct:.0f}%</span>',
                unsafe_allow_html=True,
            )
            st.progress(min(1.0, float(sim)))


render()
