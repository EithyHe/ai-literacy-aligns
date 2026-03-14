import React, { useEffect, useRef, useState } from "react";
import Graph from "graphology";
import Sigma from "sigma";
import Fuse from "fuse.js";
import { Streamlit } from "streamlit-component-lib";

const FRAMEWORK_COLORS = [
  "#3498DB", "#E74C3C", "#2ECC71", "#9B59B6", "#F39C12",
  "#1ABC9C", "#34495E", "#E91E63", "#00BCD4", "#8BC34A",
  "#FF5722", "#607D8B",
];

function buildSubgraph(graphData, expandedDomains, minEdgeWeight = 0) {
  const graph = new Graph();
  const nodes = graphData?.nodes || [];
  const edges = graphData?.edges || [];
  const visibleIds = new Set();
  nodes.forEach((n) => {
    if (n.level === "domain") visibleIds.add(n.id);
    else if (n.level === "construct" && expandedDomains.has(n.parent_id)) visibleIds.add(n.id);
  });
  nodes.forEach((n) => {
    if (!visibleIds.has(n.id)) return;
    if (graph.hasNode(n.id)) return;
    const fwId = n.framework_id || "";
    const colorIndex = fwId ? (fwId.split("").reduce((a, c) => a + c.charCodeAt(0), 0) % FRAMEWORK_COLORS.length) : 0;
    const color = FRAMEWORK_COLORS[colorIndex];
    graph.addNode(n.id, {
      label: n.label || n.id,
      x: n.x ?? 0,
      y: n.y ?? 0,
      size: 8 + Math.min((n.n_children || 0) * 1.5, 20),
      level: n.level,
      parent_id: n.parent_id || "",
      color,
    });
  });
  const edgeDone = new Set();
  edges.forEach((e) => {
    if (e.weight < minEdgeWeight) return;
    if (!visibleIds.has(e.source) || !visibleIds.has(e.target)) return;
    const key = [e.source, e.target].sort().join("--");
    if (edgeDone.has(key)) return;
    edgeDone.add(key);
    graph.addEdge(e.source, e.target, { weight: e.weight, is_meta: e.is_meta ?? false });
  });
  return graph;
}

export default function SigmaNetwork(props) {
  const { args } = props;
  const graphData = args?.graph_data || null;
  const config = args?.config || {};
  const minEdgeWeight = config.minEdgeWeight ?? 0.15;

  const containerRef = useRef(null);
  const sigmaRef = useRef(null);
  const graphRef = useRef(null);
  const [expandedDomains, setExpandedDomains] = useState(new Set());
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [searchOpen, setSearchOpen] = useState(false);
  const searchIndexRef = useRef(null);
  const fuseRef = useRef(null);

  // Build search index
  useEffect(() => {
    if (!graphData?.search_index) return;
    searchIndexRef.current = graphData.search_index;
    fuseRef.current = new Fuse(graphData.search_index, {
      keys: ["label", "path"],
      threshold: 0.3,
    });
  }, [graphData]);

  // Search
  useEffect(() => {
    if (!fuseRef.current || !searchQuery.trim()) {
      setSearchResults([]);
      return;
    }
    const results = fuseRef.current.search(searchQuery.trim()).slice(0, 10);
    setSearchResults(results.map((r) => r.item));
  }, [searchQuery]);

  // Sigma lifecycle: create once with initial subgraph
  useEffect(() => {
    if (!containerRef.current || !graphData?.nodes?.length) return;
    const initialGraph = buildSubgraph(graphData, expandedDomains, minEdgeWeight);
    graphRef.current = initialGraph;
    const sigma = new Sigma(initialGraph, containerRef.current, {
      renderLabels: true,
      labelDensity: 0.5,
      labelGridCellSize: 60,
      labelRenderedSizeThreshold: 6,
      defaultNodeColor: "#4a90d9",
      defaultEdgeColor: "#cccccc",
      minCameraRatio: 0.1,
      maxCameraRatio: 20,
      doubleClickZoomingRatio: 1,
    });
    sigmaRef.current = sigma;

    sigma.on("doubleClickNode", (payload) => {
      const { node, event } = payload;
      if (event) {
        event.sigmaDefaultPrevented = true;
        if (event.originalEvent) event.originalEvent.sigmaDefaultPrevented = true;
      }
      const graph = graphRef.current;
      if (!graph || !graph.hasNode(node)) return;
      const attrs = graph.getNodeAttributes(node);
      if (attrs.level === "domain") {
        setExpandedDomains((prev) => {
          const next = new Set(prev);
          if (next.has(node)) next.delete(node);
          else next.add(node);
          return next;
        });
      }
      Streamlit.setComponentValue({ type: "click", nodeId: node, level: attrs.level });
    });

    sigma.on("clickNode", (payload) => {
      const { node } = payload;
      const graph = graphRef.current;
      if (!graph || !graph.hasNode(node)) return;
      const attrs = graph.getNodeAttributes(node);
      Streamlit.setComponentValue({ type: "click", nodeId: node, level: attrs.level });
    });

    Streamlit.setComponentReady();
    Streamlit.setFrameHeight(600);
    return () => {
      sigma.kill();
      sigmaRef.current = null;
    };
  }, [graphData, minEdgeWeight]);

  // When expandedDomains or minEdgeWeight change, replace graph (Sigma v3 setGraph)
  useEffect(() => {
    if (!sigmaRef.current || !graphData?.nodes?.length) return;
    const newGraph = buildSubgraph(graphData, expandedDomains, minEdgeWeight);
    sigmaRef.current.setGraph(newGraph);
    graphRef.current = newGraph;
  }, [graphData, expandedDomains, minEdgeWeight]);

  const handleSearchSelect = (item) => {
    setSearchQuery("");
    setSearchResults([]);
    setSearchOpen(false);
    if (item.level === "domain") setExpandedDomains((prev) => new Set(prev).add(item.id));
    if (item.level === "construct" && item.id) {
      const node = graphData?.nodes?.find((n) => n.id === item.id);
      if (node?.parent_id) setExpandedDomains((prev) => new Set(prev).add(node.parent_id));
    }
    Streamlit.setComponentValue({ type: "navigate", nodeId: item.id, path: item.path });
  };

  if (!graphData?.nodes?.length) {
    return (
      <div style={{ padding: 16 }}>No graph data. Run the graph data pipeline first.</div>
    );
  }

  return (
    <div id="sigma-root" style={{ display: "flex", flexDirection: "column", minHeight: 600, height: "100%" }}>
      <div id="search-container" style={{ padding: 8, flexShrink: 0 }}>
        <input
          type="text"
          placeholder="Search domains, constructs..."
          value={searchQuery}
          onChange={(e) => {
            setSearchQuery(e.target.value);
            setSearchOpen(true);
          }}
          onFocus={() => setSearchOpen(true)}
          style={{ width: "100%", maxWidth: 320, padding: "8px 12px" }}
        />
        {searchOpen && searchResults.length > 0 && (
          <ul
            id="search-results"
            style={{
              position: "absolute",
              zIndex: 10,
              background: "white",
              border: "1px solid #ccc",
              maxHeight: 240,
              overflowY: "auto",
              margin: 0,
              padding: 0,
              listStyle: "none",
            }}
          >
            {searchResults.map((item) => (
              <li
                key={item.id}
                onClick={() => handleSearchSelect(item)}
                style={{ padding: "8px 12px", cursor: "pointer", borderBottom: "1px solid #eee" }}
              >
                <strong>{item.label}</strong>
                <div style={{ fontSize: 12, color: "#666" }}>{item.path}</div>
              </li>
            ))}
          </ul>
        )}
      </div>
      <div ref={containerRef} style={{ flex: 1, minHeight: 560, position: "relative" }} />
    </div>
  );
}
