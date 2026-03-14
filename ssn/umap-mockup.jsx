import { useState, useMemo, useRef } from "react";

// Simulated framework data with constructs
const frameworks = [
  {
    name: "Big Five (IPIP)",
    color: "#2563EB",
    constructs: [
      { name: "Extraversion", x: 0.52, y: 0.42 },
      { name: "Agreeableness", x: 0.48, y: 0.48 },
      { name: "Conscientiousness", x: 0.58, y: 0.52 },
      { name: "Neuroticism", x: 0.42, y: 0.38 },
      { name: "Openness", x: 0.35, y: 0.55 },
    ],
  },
  {
    name: "HEXACO",
    color: "#D97706",
    constructs: [
      { name: "Honesty-Humility", x: 0.38, y: 0.42 },
      { name: "Emotionality", x: 0.44, y: 0.35 },
      { name: "Extraversion (H)", x: 0.55, y: 0.45 },
      { name: "Agreeableness (H)", x: 0.50, y: 0.50 },
      { name: "Conscientiousness (H)", x: 0.60, y: 0.48 },
      { name: "Openness (H)", x: 0.33, y: 0.52 },
    ],
  },
  {
    name: "IPIP-NEO",
    color: "#059669",
    constructs: [
      { name: "Anxiety", x: 0.40, y: 0.30 },
      { name: "Anger", x: 0.45, y: 0.32 },
      { name: "Depression", x: 0.38, y: 0.25 },
      { name: "Self-Consciousness", x: 0.42, y: 0.28 },
      { name: "Warmth", x: 0.55, y: 0.38 },
      { name: "Gregariousness", x: 0.60, y: 0.35 },
      { name: "Assertiveness", x: 0.58, y: 0.40 },
      { name: "Activity", x: 0.62, y: 0.42 },
      { name: "Imagination", x: 0.30, y: 0.58 },
      { name: "Artistic Interest", x: 0.25, y: 0.62 },
    ],
  },
  {
    name: "Oregon Vocational",
    color: "#DC2626",
    constructs: [
      { name: "Interest in Science", x: 0.28, y: 0.70 },
      { name: "Interest in Music", x: 0.20, y: 0.65 },
      { name: "Interest in Sports", x: 0.72, y: 0.75 },
      { name: "Interest in Food", x: 0.78, y: 0.30 },
      { name: "Interest in Outdoors", x: 0.80, y: 0.60 },
      { name: "Interest in Shopping", x: 0.85, y: 0.40 },
      { name: "Interest in Games", x: 0.65, y: 0.80 },
      { name: "Interest in Computers", x: 0.35, y: 0.72 },
      { name: "Interest in Reading", x: 0.22, y: 0.68 },
      { name: "Interest in Gardening", x: 0.15, y: 0.45 },
    ],
  },
  {
    name: "Clinical Scales",
    color: "#7C3AED",
    constructs: [
      { name: "ADHD", x: 0.45, y: 0.40 },
      { name: "Radloff Depression", x: 0.36, y: 0.22 },
      { name: "Foa PTSD", x: 0.40, y: 0.26 },
      { name: "Cognitive Failures", x: 0.50, y: 0.36 },
      { name: "Perfectionism", x: 0.48, y: 0.44 },
    ],
  },
  {
    name: "Values & Motivation",
    color: "#0891B2",
    constructs: [
      { name: "Leadership", x: 0.55, y: 0.68 },
      { name: "Love of Learning", x: 0.28, y: 0.63 },
      { name: "Spirituality", x: 0.48, y: 0.62 },
      { name: "Altruism", x: 0.62, y: 0.55 },
      { name: "Organization", x: 0.65, y: 0.70 },
      { name: "Creativity", x: 0.25, y: 0.58 },
    ],
  },
];

function convexHull(points) {
  if (points.length < 3) return points;
  const pts = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (O, A, B) => (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (const p of pts.reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  upper.pop();
  lower.pop();
  return lower.concat(upper);
}

function expandHull(hull, centroid, padding) {
  return hull.map((p) => ({
    x: centroid.x + (p.x - centroid.x) * (1 + padding),
    y: centroid.y + (p.y - centroid.y) * (1 + padding),
  }));
}

const W = 800;
const H = 640;
const PAD = 60;

function toSvgX(x) { return PAD + x * (W - 2 * PAD); }
function toSvgY(y) { return PAD + y * (H - 2 * PAD); }

export default function UMAPMockup() {
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [activeFramework, setActiveFramework] = useState(null);
  const [showLabels, setShowLabels] = useState(false);
  const svgRef = useRef(null);

  const processed = useMemo(() => {
    return frameworks.map((fw) => {
      const centroid = {
        x: fw.constructs.reduce((s, c) => s + c.x, 0) / fw.constructs.length,
        y: fw.constructs.reduce((s, c) => s + c.y, 0) / fw.constructs.length,
      };
      const hull = convexHull(fw.constructs);
      const expanded = expandHull(hull, centroid, 0.35);
      return { ...fw, centroid, hull: expanded };
    });
  }, []);

  const hullPath = (hull) => {
    if (hull.length < 2) return "";
    const pts = hull.map((p) => ({ x: toSvgX(p.x), y: toSvgY(p.y) }));
    let d = `M ${(pts[pts.length - 1].x + pts[0].x) / 2} ${(pts[pts.length - 1].y + pts[0].y) / 2}`;
    for (let i = 0; i < pts.length; i++) {
      const next = pts[(i + 1) % pts.length];
      const midX = (pts[i].x + next.x) / 2;
      const midY = (pts[i].y + next.y) / 2;
      d += ` Q ${pts[i].x} ${pts[i].y} ${midX} ${midY}`;
    }
    d += " Z";
    return d;
  };

  const isActive = (fwName) => !activeFramework || activeFramework === fwName;

  return (
    <div style={{
      minHeight: "100vh",
      background: "#FAFAFA",
      color: "#1A1A1A",
      fontFamily: "'DM Sans', 'Helvetica Neue', sans-serif",
      padding: "32px",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet" />

      <div style={{ maxWidth: 1100, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 8 }}>
          <div style={{
            fontSize: 11,
            fontFamily: "'DM Mono', monospace",
            color: "#999",
            textTransform: "uppercase",
            letterSpacing: "0.15em",
            marginBottom: 8,
          }}>
            Mockup · 改进后效果示意
          </div>
          <h1 style={{
            fontSize: 28,
            fontWeight: 700,
            margin: 0,
            lineHeight: 1.2,
            color: "#111",
          }}>
            Psychological Frameworks in UMAP Semantic Space
          </h1>
          <p style={{
            fontSize: 14,
            color: "#777",
            margin: "8px 0 0 0",
            maxWidth: 600,
          }}>
            UMAP降维基于construct语义相似度。轮廓线表示框架覆盖范围，重叠区域揭示框架间的冗余。
          </p>
        </div>

        <div style={{ display: "flex", gap: 24, alignItems: "flex-start", marginTop: 24 }}>
          {/* Main chart */}
          <div style={{
            background: "#FFFFFF",
            borderRadius: 12,
            border: "1px solid #E5E5E5",
            padding: 16,
            flex: 1,
            boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
          }}>
            {/* Controls */}
            <div style={{
              display: "flex",
              gap: 12,
              marginBottom: 12,
              alignItems: "center",
            }}>
              <button
                onClick={() => setShowLabels(!showLabels)}
                style={{
                  background: showLabels ? "#F0F0F0" : "#FAFAFA",
                  border: "1px solid #DDD",
                  color: showLabels ? "#333" : "#888",
                  padding: "6px 14px",
                  borderRadius: 6,
                  fontSize: 12,
                  cursor: "pointer",
                  fontFamily: "'DM Mono', monospace",
                  transition: "all 0.2s",
                }}
              >
                {showLabels ? "● 隐藏标签" : "○ 显示Construct标签"}
              </button>
              {activeFramework && (
                <button
                  onClick={() => setActiveFramework(null)}
                  style={{
                    background: "#FAFAFA",
                    border: "1px solid #DDD",
                    color: "#888",
                    padding: "6px 14px",
                    borderRadius: 6,
                    fontSize: 12,
                    cursor: "pointer",
                    fontFamily: "'DM Mono', monospace",
                  }}
                >
                  ✕ 清除筛选
                </button>
              )}
              <span style={{ fontSize: 11, color: "#BBB", marginLeft: "auto", fontFamily: "'DM Mono', monospace" }}>
                hover查看 · 点击图例筛选
              </span>
            </div>

            <svg
              ref={svgRef}
              viewBox={`0 0 ${W} ${H}`}
              style={{ width: "100%", height: "auto", display: "block" }}
            >
              {/* Grid */}
              <defs>
                <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                  <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#F0F0F0" strokeWidth="0.5" />
                </pattern>
              </defs>
              <rect x={PAD} y={PAD} width={W - 2 * PAD} height={H - 2 * PAD} fill="url(#grid)" />
              <rect x={PAD} y={PAD} width={W - 2 * PAD} height={H - 2 * PAD} fill="none" stroke="#E8E8E8" strokeWidth="1" />

              {/* Hull fills */}
              {processed.map((fw) => (
                <path
                  key={`fill-${fw.name}`}
                  d={hullPath(fw.hull)}
                  fill={fw.color}
                  opacity={isActive(fw.name) ? 0.07 : 0.015}
                  style={{ transition: "opacity 0.4s" }}
                />
              ))}

              {/* Hull outlines */}
              {processed.map((fw) => (
                <path
                  key={`hull-${fw.name}`}
                  d={hullPath(fw.hull)}
                  fill="none"
                  stroke={fw.color}
                  strokeWidth={activeFramework === fw.name ? 2.5 : 1.5}
                  strokeDasharray={activeFramework === fw.name ? "none" : "6 4"}
                  opacity={isActive(fw.name) ? 0.7 : 0.1}
                  style={{ transition: "all 0.4s" }}
                />
              ))}

              {/* Points */}
              {processed.map((fw) =>
                fw.constructs.map((c, i) => {
                  const cx = toSvgX(c.x);
                  const cy = toSvgY(c.y);
                  const key = `${fw.name}-${i}`;
                  const isHovered = hoveredPoint === key;
                  const visible = isActive(fw.name);
                  return (
                    <g key={key}>
                      <circle
                        cx={cx}
                        cy={cy}
                        r={isHovered ? 6 : 4}
                        fill={fw.color}
                        opacity={visible ? (isHovered ? 1 : 0.75) : 0.08}
                        stroke={isHovered ? "#333" : "#FFF"}
                        strokeWidth={isHovered ? 1.5 : 0.8}
                        style={{ cursor: "pointer", transition: "all 0.2s" }}
                        onMouseEnter={() => setHoveredPoint(key)}
                        onMouseLeave={() => setHoveredPoint(null)}
                      />
                      {(isHovered || (showLabels && visible)) && (
                        <g>
                          {isHovered && (
                            <rect
                              x={cx + 6}
                              y={cy - 21}
                              width={c.name.length * 6.5 + 14}
                              height={20}
                              rx={4}
                              fill="#333"
                              opacity={0.9}
                            />
                          )}
                          <text
                            x={cx + (isHovered ? 13 : 8)}
                            y={cy - (isHovered ? 7 : 8)}
                            fontSize={isHovered ? 11 : 10}
                            fill={isHovered ? "#FFF" : "#666"}
                            fontFamily="'DM Sans', sans-serif"
                            fontWeight={isHovered ? 600 : 400}
                            style={{ pointerEvents: "none" }}
                          >
                            {c.name}
                          </text>
                        </g>
                      )}
                    </g>
                  );
                })
              )}

              {/* Framework centroid labels */}
              {processed.map((fw) => (
                <g key={`label-${fw.name}`} opacity={isActive(fw.name) ? 1 : 0.12} style={{ transition: "opacity 0.4s" }}>
                  <text
                    x={toSvgX(fw.centroid.x)}
                    y={toSvgY(fw.centroid.y) - 2}
                    textAnchor="middle"
                    fontSize={14}
                    fontWeight={700}
                    fill={fw.color}
                    fontFamily="'DM Sans', sans-serif"
                    style={{ pointerEvents: "none" }}
                  >
                    {fw.name}
                  </text>
                  <text
                    x={toSvgX(fw.centroid.x)}
                    y={toSvgY(fw.centroid.y) + 14}
                    textAnchor="middle"
                    fontSize={10}
                    fill="#AAA"
                    fontFamily="'DM Mono', monospace"
                    style={{ pointerEvents: "none" }}
                  >
                    {fw.constructs.length} constructs
                  </text>
                </g>
              ))}

              {/* Axis labels */}
              <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fill="#BBB" fontFamily="'DM Mono', monospace">
                UMAP Dimension 1
              </text>
              <text x={14} y={H / 2} textAnchor="middle" fontSize={10} fill="#BBB" fontFamily="'DM Mono', monospace"
                transform={`rotate(-90, 14, ${H / 2})`}>
                UMAP Dimension 2
              </text>
            </svg>
          </div>

          {/* Sidebar */}
          <div style={{ width: 240, flexShrink: 0 }}>
            {/* Legend */}
            <div style={{
              background: "#FFFFFF",
              borderRadius: 12,
              border: "1px solid #E5E5E5",
              padding: 16,
              marginBottom: 16,
              boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
            }}>
              <div style={{
                fontSize: 11,
                fontFamily: "'DM Mono', monospace",
                color: "#999",
                textTransform: "uppercase",
                letterSpacing: "0.12em",
                marginBottom: 12,
              }}>
                Frameworks
              </div>
              {processed.map((fw) => (
                <div
                  key={fw.name}
                  onClick={() => setActiveFramework(activeFramework === fw.name ? null : fw.name)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "8px 8px",
                    borderRadius: 6,
                    cursor: "pointer",
                    background: activeFramework === fw.name ? "#F5F5F5" : "transparent",
                    transition: "background 0.2s",
                    marginBottom: 2,
                  }}
                >
                  <div style={{
                    width: 12,
                    height: 12,
                    borderRadius: 3,
                    background: fw.color,
                    opacity: isActive(fw.name) ? 1 : 0.25,
                    transition: "opacity 0.3s",
                    flexShrink: 0,
                  }} />
                  <div>
                    <div style={{
                      fontSize: 13,
                      fontWeight: 500,
                      color: isActive(fw.name) ? "#333" : "#CCC",
                      transition: "color 0.3s",
                      lineHeight: 1.3,
                    }}>
                      {fw.name}
                    </div>
                    <div style={{
                      fontSize: 11,
                      color: "#AAA",
                      fontFamily: "'DM Mono', monospace",
                    }}>
                      {fw.constructs.length} constructs
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Overlap matrix */}
            <div style={{
              background: "#FFFFFF",
              borderRadius: 12,
              border: "1px solid #E5E5E5",
              padding: 16,
              boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
            }}>
              <div style={{
                fontSize: 11,
                fontFamily: "'DM Mono', monospace",
                color: "#999",
                textTransform: "uppercase",
                letterSpacing: "0.12em",
                marginBottom: 12,
              }}>
                Overlap Index
              </div>
              <div style={{ fontSize: 11, color: "#999", lineHeight: 1.5, marginBottom: 12 }}>
                框架间凸包重叠面积占比（模拟数据）
              </div>
              {[
                { a: "Big Five", b: "HEXACO", v: 0.72, c1: "#2563EB", c2: "#D97706" },
                { a: "Big Five", b: "IPIP-NEO", v: 0.65, c1: "#2563EB", c2: "#059669" },
                { a: "Big Five", b: "Clinical", v: 0.38, c1: "#2563EB", c2: "#7C3AED" },
                { a: "HEXACO", b: "IPIP-NEO", v: 0.58, c1: "#D97706", c2: "#059669" },
                { a: "Vocational", b: "Values", v: 0.22, c1: "#DC2626", c2: "#0891B2" },
              ].map((pair, i) => (
                <div key={i} style={{ marginBottom: 10 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11 }}>
                      <span style={{ color: pair.c1, fontWeight: 600 }}>●</span>
                      <span style={{ color: "#CCC" }}>&</span>
                      <span style={{ color: pair.c2, fontWeight: 600 }}>●</span>
                      <span style={{ color: "#777", marginLeft: 2 }}>{pair.a} × {pair.b}</span>
                    </div>
                    <span style={{
                      fontFamily: "'DM Mono', monospace",
                      fontSize: 12,
                      fontWeight: 600,
                      color: pair.v > 0.5 ? "#B45309" : "#999",
                    }}>
                      {(pair.v * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div style={{ height: 4, background: "#F0F0F0", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{
                      height: "100%",
                      width: `${pair.v * 100}%`,
                      background: `linear-gradient(90deg, ${pair.c1}, ${pair.c2})`,
                      borderRadius: 2,
                      opacity: 0.6,
                    }} />
                  </div>
                </div>
              ))}
            </div>

            {/* Design notes */}
            <div style={{
              marginTop: 16,
              background: "#FFFFFF",
              borderRadius: 12,
              border: "1px solid #E5E5E5",
              padding: 16,
              boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
            }}>
              <div style={{
                fontSize: 11,
                fontFamily: "'DM Mono', monospace",
                color: "#999",
                textTransform: "uppercase",
                letterSpacing: "0.12em",
                marginBottom: 10,
              }}>
                改进要点
              </div>
              {[
                "每个框架独立色相，一眼可辨",
                "轮廓线代替填充色块，减少遮挡",
                "框架名在质心位置，大字加粗",
                "Construct标签默认隐藏，按需显示",
                "Overlap量化指标作为补充信息",
                "点击图例可单独高亮某一框架",
              ].map((note, i) => (
                <div key={i} style={{
                  fontSize: 12,
                  color: "#777",
                  lineHeight: 1.6,
                  paddingLeft: 14,
                  position: "relative",
                  marginBottom: 4,
                }}>
                  <span style={{ position: "absolute", left: 0, color: "#CCC" }}>→</span>
                  {note}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
