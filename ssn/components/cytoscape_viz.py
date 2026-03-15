"""Cytoscape.js renderer for construct-level network in Streamlit."""

from __future__ import annotations

import json
from typing import Any

import streamlit.components.v1 as components


def _safe_json(obj: Any) -> str:
    text = json.dumps(obj, ensure_ascii=False)
    return text.replace("</script>", "<\\/script>")


def render_construct_network(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    height: int = 640,
    key: str | None = None,
) -> None:
    """Render an interactive construct network with Cytoscape.

    Node click writes `selected_construct_id` into URL query params and reloads the page.
    """
    node_elements: list[dict[str, Any]] = []
    for node in nodes:
        data = {k: v for k, v in node.items() if k not in {"x", "y"}}
        if "x" in node and "y" in node:
            node_elements.append(
                {
                    "data": data,
                    "position": {"x": float(node["x"]), "y": float(node["y"])},
                }
            )
        else:
            node_elements.append({"data": data})

    elements = {
        "nodes": node_elements,
        "edges": [{"data": e} for e in edges],
    }

    html = f"""
<div id=\"cy-toolbar\">
  <button id=\"btn-fit\">Fit</button>
  <button id=\"btn-reset\">Reset</button>
  <span id=\"edge-count\"></span>
</div>
<div id=\"cy\"></div>
<div id=\"cy-hover\">Hover node/edge to inspect details</div>

<script src=\"https://unpkg.com/cytoscape@3.29.2/dist/cytoscape.min.js\"></script>
<script>
(function() {{
  const elements = {_safe_json(elements)};
  const hover = document.getElementById('cy-hover');
  const edgeCount = document.getElementById('edge-count');
  const hasPreset = elements.nodes.every((n) => n.position && Number.isFinite(n.position.x) && Number.isFinite(n.position.y));

  const cy = cytoscape({{
    container: document.getElementById('cy'),
    elements,
    wheelSensitivity: 0.25,
    textureOnViewport: true,
    hideEdgesOnViewport: true,
    motionBlur: true,
    minZoom: 0.2,
    maxZoom: 4.0,
    style: [
      {{
        selector: 'node',
        style: {{
          'label': 'data(label)',
          'font-size': 10,
          'text-wrap': 'none',
          'text-valign': 'center',
          'text-halign': 'center',
          'background-color': '#2E86AB',
          'color': '#FFFFFF',
          'width': 'mapData(size, 1, 20, 24, 60)',
          'height': 'mapData(size, 1, 20, 24, 60)',
          'border-color': '#FFFFFF',
          'border-width': 1,
        }}
      }},
      {{
        selector: 'edge',
        style: {{
          'width': 'mapData(weight, 0, 1, 0.8, 4)',
          'line-color': '#8AA1B1',
          'curve-style': 'bezier',
          'opacity': 0.55,
        }}
      }},
      {{
        selector: ':selected',
        style: {{
          'border-width': 3,
          'border-color': '#C0392B',
          'line-color': '#C0392B',
          'target-arrow-color': '#C0392B',
        }}
      }}
    ],
    layout: hasPreset
      ? {{
          name: 'preset',
          fit: true,
          padding: 24,
        }}
      : {{
          name: 'cose',
          animate: false,
          fit: true,
          padding: 24,
          nodeRepulsion: 260000,
          idealEdgeLength: 85,
          edgeElasticity: 30,
          gravity: 0.3,
          numIter: 300,
        }},
  }});

  edgeCount.innerText = `${{cy.nodes().length}} nodes / ${{cy.edges().length}} edges`;

  function writeSelectedNode(nodeId) {{
    try {{
      const parentWin = window.parent || window;
      const url = new URL(parentWin.location.href);
      url.searchParams.set('selected_construct_id', nodeId);
      parentWin.location.href = url.toString();
    }} catch (err) {{
      console.error('Failed to write query params:', err);
    }}
  }}

  cy.on('tap', 'node', function(evt) {{
    const d = evt.target.data();
    writeSelectedNode(d.id);
  }});

  cy.on('mouseover', 'node', function(evt) {{
    const d = evt.target.data();
    const label = d.full_label || d.label || d.id;
    hover.innerText = `Node: ${{label}} | Degree: ${{d.degree || 0}} | Avg edge: ${{(d.avg_weight || 0).toFixed(3)}}`;
  }});

  cy.on('mouseover', 'edge', function(evt) {{
    const d = evt.target.data();
    hover.innerText = `Edge: ${{d.source_label}} ↔ ${{d.target_label}} | weight=${{(d.weight || 0).toFixed(3)}}`;
  }});

  cy.on('mouseout', function() {{
    hover.innerText = 'Hover node/edge to inspect details';
  }});

  document.getElementById('btn-fit').addEventListener('click', function() {{
    cy.fit(undefined, 24);
  }});

  document.getElementById('btn-reset').addEventListener('click', function() {{
    if (hasPreset) {{
      cy.fit(undefined, 24);
      return;
    }}
    cy.layout({{ name: 'cose', animate: false, fit: true, padding: 24 }}).run();
  }});
}})();
</script>

<style>
  #cy {{
    width: 100%;
    height: {int(height)-56}px;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    background: linear-gradient(180deg, #fcfdff 0%, #f6f8fb 100%);
  }}
  #cy-toolbar {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    font-size: 12px;
  }}
  #cy-toolbar button {{
    border: 1px solid #d0d7de;
    background: #fff;
    border-radius: 6px;
    padding: 4px 8px;
    cursor: pointer;
  }}
  #edge-count {{
    margin-left: auto;
    color: #475569;
  }}
  #cy-hover {{
    margin-top: 8px;
    color: #334155;
    font-size: 12px;
  }}
</style>
"""
    components.html(html, height=height, scrolling=False)
