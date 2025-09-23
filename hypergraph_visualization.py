#!/usr/bin/env python3
"""
超图可交互可视化工具
支持多种可视化方式：Plotly、NetworkX、以及Web界面
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("警告: Plotly未安装，将使用matplotlib进行可视化")

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("警告: NetworkX未安装，将使用简化可视化")

class HypergraphVisualizer:
    def __init__(self, nodes_file: str = None, edges_file: str = None):
        """
        初始化超图可视化器
        
        Args:
            nodes_file: 节点CSV文件路径
            edges_file: 边CSV文件路径
        """
        self.nodes_file = nodes_file
        self.edges_file = edges_file
        self.nodes_df = None
        self.edges_df = None
        self.graph = None
        
    def load_data(self, nodes_file: str = None, edges_file: str = None):
        """加载超图数据"""
        if nodes_file:
            self.nodes_file = nodes_file
        if edges_file:
            self.edges_file = edges_file
            
        try:
            if self.nodes_file and os.path.exists(self.nodes_file):
                self.nodes_df = pd.read_csv(self.nodes_file)
                print(f"已加载节点数据: {len(self.nodes_df)} 个节点")
            else:
                print("节点文件不存在，将生成示例数据")
                self._generate_sample_data()
                
            if self.edges_file and os.path.exists(self.edges_file):
                self.edges_df = pd.read_csv(self.edges_file)
                print(f"已加载边数据: {len(self.edges_df)} 条边")
            else:
                print("边文件不存在，将生成示例数据")
                self._generate_sample_data()
                
        except Exception as e:
            print(f"加载数据时出错: {e}")
            self._generate_sample_data()
    
    def _generate_sample_data(self):
        """生成示例超图数据"""
        print("生成示例超图数据...")
        
        # 生成示例节点
        nodes_data = []
        
        # 项目节点
        for i in range(10):
            nodes_data.append({
                'id': f'it_{i:06d}',
                'label': f'Item {i+1}',
                'type': 'item'
            })
        
        # 构建节点
        constructs = ['AI Literacy', 'Digital Skills', 'Critical Thinking', 'Ethics']
        for construct in constructs:
            nodes_data.append({
                'id': construct,
                'label': construct,
                'type': 'construct'
            })
        
        # 聚类节点
        for i in range(3):
            nodes_data.append({
                'id': f'cluster_{i}',
                'label': f'Cluster {i}',
                'type': 'cluster'
            })
        
        self.nodes_df = pd.DataFrame(nodes_data)
        
        # 生成示例边
        edges_data = []
        
        # 项目-构建边
        for i in range(10):
            construct = constructs[i % len(constructs)]
            edges_data.append({
                'src': f'it_{i:06d}',
                'dst': construct,
                'rel': 'item_construct',
                'weight': 1.0
            })
        
        # 项目-聚类边
        for i in range(10):
            cluster = f'cluster_{i % 3}'
            edges_data.append({
                'src': f'it_{i:06d}',
                'dst': cluster,
                'rel': 'item_cluster',
                'weight': 1.0
            })
        
        # 构建-聚类边
        for construct in constructs:
            for i in range(3):
                weight = np.random.randint(1, 5)
                edges_data.append({
                    'src': construct,
                    'dst': f'cluster_{i}',
                    'rel': 'construct_cluster',
                    'weight': float(weight)
                })
        
        self.edges_df = pd.DataFrame(edges_data)
        print(f"生成了 {len(self.nodes_df)} 个节点和 {len(self.edges_df)} 条边")
    
    def create_plotly_visualization(self, output_file: str = "data/docs/hypergraph_interactive.html"):
        """创建Plotly交互式可视化"""
        if not PLOTLY_AVAILABLE:
            print("Plotly不可用，跳过交互式可视化")
            return
            
        if self.nodes_df is None or self.edges_df is None:
            self.load_data()
        
        # 准备节点位置
        node_positions = self._calculate_node_positions()
        
        # 创建图形
        fig = go.Figure()
        
        # 添加边
        edge_traces = {}
        for rel_type in self.edges_df['rel'].unique():
            rel_edges = self.edges_df[self.edges_df['rel'] == rel_type]
            edge_x, edge_y = [], []
            
            for _, edge in rel_edges.iterrows():
                src_pos = node_positions.get(edge['src'])
                dst_pos = node_positions.get(edge['dst'])
                
                if src_pos and dst_pos:
                    edge_x.extend([src_pos[0], dst_pos[0], None])
                    edge_y.extend([src_pos[1], dst_pos[1], None])
            
            edge_traces[rel_type] = go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=1, color=self._get_edge_color(rel_type)),
                hoverinfo='none',
                showlegend=True,
                name=rel_type.replace('_', ' ').title()
            )
        
        # 添加节点
        node_traces = {}
        for node_type in self.nodes_df['type'].unique():
            type_nodes = self.nodes_df[self.nodes_df['type'] == node_type]
            node_x, node_y = [], []
            node_text = []
            
            for _, node in type_nodes.iterrows():
                pos = node_positions.get(node['id'])
                if pos:
                    node_x.append(pos[0])
                    node_y.append(pos[1])
                    node_text.append(f"{node['label']}<br>Type: {node['type']}")
            
            node_traces[node_type] = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=self._get_node_color(node_type),
                    line=dict(width=2, color='white')
                ),
                text=[node['label'] for _, node in type_nodes.iterrows()],
                textposition="middle center",
                hoverinfo='text',
                hovertext=node_text,
                showlegend=True,
                name=node_type.title()
            )
        
        # 添加所有轨迹
        for trace in edge_traces.values():
            fig.add_trace(trace)
        for trace in node_traces.values():
            fig.add_trace(trace)
        
        # 更新布局
        fig.update_layout(
            title="超图交互式可视化",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="拖拽节点进行交互，悬停查看详情",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor="left", yanchor="bottom",
                font=dict(color="gray", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        # 保存为HTML
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        fig.write_html(output_file)
        print(f"交互式可视化已保存到: {output_file}")
        
        return fig
    
    def create_networkx_visualization(self, output_file: str = "data/docs/hypergraph_static.png"):
        """创建NetworkX静态可视化"""
        if not NETWORKX_AVAILABLE:
            print("NetworkX不可用，跳过静态可视化")
            return
            
        if self.nodes_df is None or self.edges_df is None:
            self.load_data()
        
        # 创建图形
        G = nx.Graph()
        
        # 添加节点
        for _, node in self.nodes_df.iterrows():
            G.add_node(node['id'], label=node['label'], type=node['type'])
        
        # 添加边
        for _, edge in self.edges_df.iterrows():
            G.add_edge(edge['src'], edge['dst'], 
                      rel=edge['rel'], weight=edge['weight'])
        
        # 计算布局
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 创建图形
        plt.figure(figsize=(15, 10))
        
        # 按类型绘制节点
        node_types = self.nodes_df['type'].unique()
        colors = {'item': '#FF6B6B', 'construct': '#4ECDC4', 'cluster': '#45B7D1'}
        
        for node_type in node_types:
            nodes = self.nodes_df[self.nodes_df['type'] == node_type]['id'].tolist()
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                 node_color=colors.get(node_type, '#95A5A6'),
                                 node_size=500, alpha=0.8, label=node_type.title())
        
        # 绘制边
        edge_types = self.edges_df['rel'].unique()
        edge_styles = {'item_construct': '-', 'item_cluster': '--', 'construct_cluster': ':'}
        
        for edge_type in edge_types:
            edges = [(row['src'], row['dst']) for _, row in 
                    self.edges_df[self.edges_df['rel'] == edge_type].iterrows()]
            nx.draw_networkx_edges(G, pos, edgelist=edges,
                                 style=edge_styles.get(edge_type, '-'),
                                 alpha=0.6, width=1.5)
        
        # 添加标签
        labels = {row['id']: row['label'] for _, row in self.nodes_df.iterrows()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        plt.title("超图静态可视化", fontsize=16, fontweight='bold')
        plt.legend(scatterpoints=1, loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        
        # 保存图形
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"静态可视化已保存到: {output_file}")
    
    def create_web_interface(self, output_file: str = "data/docs/hypergraph_explorer.html"):
        """创建Web探索界面"""
        if self.nodes_df is None or self.edges_df is None:
            self.load_data()
        
        # 生成HTML内容
        html_content = self._generate_web_interface_html()
        
        # 保存HTML文件
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Web探索界面已保存到: {output_file}")
    
    def _calculate_node_positions(self) -> Dict[str, Tuple[float, float]]:
        """计算节点位置"""
        positions = {}
        
        # 按类型分组
        items = self.nodes_df[self.nodes_df['type'] == 'item']
        constructs = self.nodes_df[self.nodes_df['type'] == 'construct']
        clusters = self.nodes_df[self.nodes_df['type'] == 'cluster']
        
        # 项目节点 - 圆形排列
        n_items = len(items)
        for i, (_, item) in enumerate(items.iterrows()):
            angle = 2 * np.pi * i / n_items
            radius = 3
            positions[item['id']] = (radius * np.cos(angle), radius * np.sin(angle))
        
        # 构建节点 - 上部排列
        n_constructs = len(constructs)
        for i, (_, construct) in enumerate(constructs.iterrows()):
            x = (i - n_constructs/2) * 2
            positions[construct['id']] = (x, 5)
        
        # 聚类节点 - 下部排列
        n_clusters = len(clusters)
        for i, (_, cluster) in enumerate(clusters.iterrows()):
            x = (i - n_clusters/2) * 3
            positions[cluster['id']] = (x, -5)
        
        return positions
    
    def _get_node_color(self, node_type: str) -> str:
        """获取节点颜色"""
        colors = {
            'item': '#FF6B6B',
            'construct': '#4ECDC4', 
            'cluster': '#45B7D1'
        }
        return colors.get(node_type, '#95A5A6')
    
    def _get_edge_color(self, rel_type: str) -> str:
        """获取边颜色"""
        colors = {
            'item_construct': '#E74C3C',
            'item_cluster': '#3498DB',
            'construct_cluster': '#2ECC71'
        }
        return colors.get(rel_type, '#95A5A6')
    
    def _generate_web_interface_html(self) -> str:
        """生成Web界面HTML"""
        # 准备数据
        nodes_data = self.nodes_df.to_dict('records')
        edges_data = self.edges_df.to_dict('records')
        
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>超图探索器</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .content {
            padding: 20px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        #graph-container {
            height: 600px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .controls {
            margin-top: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        label {
            font-weight: bold;
            color: #333;
        }
        select, input {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .export-buttons {
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        .btn-primary {
            background: #667eea;
            color: white;
        }
        .btn-primary:hover {
            background: #5a6fd8;
        }
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        .btn-secondary:hover {
            background: #5a6268;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔗 超图探索器</h1>
            <p>交互式探索项目、构建和聚类之间的关系</p>
        </div>
        
        <div class="content">
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number" id="total-nodes">0</div>
                    <div class="stat-label">总节点数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="total-edges">0</div>
                    <div class="stat-label">总边数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="node-types">0</div>
                    <div class="stat-label">节点类型</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="edge-types">0</div>
                    <div class="stat-label">关系类型</div>
                </div>
            </div>
            
            <div id="graph-container"></div>
            
            <div class="controls">
                <div class="control-group">
                    <label>显示节点类型:</label>
                    <select id="node-type-filter" multiple>
                        <option value="item">项目</option>
                        <option value="construct">构建</option>
                        <option value="cluster">聚类</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>显示边类型:</label>
                    <select id="edge-type-filter" multiple>
                        <option value="item_construct">项目-构建</option>
                        <option value="item_cluster">项目-聚类</option>
                        <option value="construct_cluster">构建-聚类</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>节点大小:</label>
                    <input type="range" id="node-size" min="5" max="30" value="15">
                </div>
            </div>
            
            <div class="export-buttons">
                <button class="btn btn-primary" onclick="exportData('all')">导出所有数据</button>
                <button class="btn btn-secondary" onclick="exportData('nodes')">导出节点</button>
                <button class="btn btn-secondary" onclick="exportData('edges')">导出边</button>
                <button class="btn btn-primary" onclick="downloadImage()">下载图片</button>
            </div>
        </div>
    </div>

    <script>
        // 数据
        const nodesData = """ + json.dumps(nodes_data, ensure_ascii=False) + """;
        const edgesData = """ + json.dumps(edges_data, ensure_ascii=False) + """;
        
        // 初始化统计信息
        document.getElementById('total-nodes').textContent = nodesData.length;
        document.getElementById('total-edges').textContent = edgesData.length;
        document.getElementById('node-types').textContent = new Set(nodesData.map(n => n.type)).size;
        document.getElementById('edge-types').textContent = new Set(edgesData.map(e => e.rel)).size;
        
        // 初始化所有选择框为选中状态
        document.querySelectorAll('#node-type-filter option').forEach(option => option.selected = true);
        document.querySelectorAll('#edge-type-filter option').forEach(option => option.selected = true);
        
        // 创建可视化
        function createVisualization() {
            const filteredNodes = getFilteredNodes();
            const filteredEdges = getFilteredEdges();
            
            // 计算节点位置
            const positions = calculatePositions(filteredNodes);
            
            // 准备Plotly数据
            const traces = [];
            
            // 添加边
            const edgeTypes = [...new Set(filteredEdges.map(e => e.rel))];
            edgeTypes.forEach(relType => {
                const relEdges = filteredEdges.filter(e => e.rel === relType);
                const edgeX = [], edgeY = [];
                
                relEdges.forEach(edge => {
                    const srcPos = positions[edge.src];
                    const dstPos = positions[edge.dst];
                    if (srcPos && dstPos) {
                        edgeX.push(srcPos[0], dstPos[0], null);
                        edgeY.push(srcPos[1], dstPos[1], null);
                    }
                });
                
                traces.push({
                    x: edgeX,
                    y: edgeY,
                    mode: 'lines',
                    line: { width: 1, color: getEdgeColor(relType) },
                    hoverinfo: 'none',
                    showlegend: true,
                    name: relType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())
                });
            });
            
            // 添加节点
            const nodeTypes = [...new Set(filteredNodes.map(n => n.type))];
            nodeTypes.forEach(nodeType => {
                const typeNodes = filteredNodes.filter(n => n.type === nodeType);
                const nodeX = [], nodeY = [], nodeText = [], hoverText = [];
                
                typeNodes.forEach(node => {
                    const pos = positions[node.id];
                    if (pos) {
                        nodeX.push(pos[0]);
                        nodeY.push(pos[1]);
                        nodeText.push(node.label);
                        hoverText.push(`${node.label}<br>Type: ${node.type}<br>ID: ${node.id}`);
                    }
                });
                
                traces.push({
                    x: nodeX,
                    y: nodeY,
                    mode: 'markers+text',
                    marker: {
                        size: parseInt(document.getElementById('node-size').value),
                        color: getNodeColor(nodeType),
                        line: { width: 2, color: 'white' }
                    },
                    text: nodeText,
                    textposition: "middle center",
                    hoverinfo: 'text',
                    hovertext: hoverText,
                    showlegend: true,
                    name: nodeType.charAt(0).toUpperCase() + nodeType.slice(1)
                });
            });
            
            // 创建图形
            Plotly.newPlot('graph-container', traces, {
                title: '超图交互式可视化',
                showlegend: true,
                hovermode: 'closest',
                margin: { b: 20, l: 5, r: 5, t: 40 },
                annotations: [{
                    text: "拖拽节点进行交互，悬停查看详情",
                    showarrow: false,
                    xref: "paper", yref: "paper",
                    x: 0.005, y: -0.002,
                    xanchor: "left", yanchor: "bottom",
                    font: { color: "gray", size: 12 }
                }],
                xaxis: { showgrid: false, zeroline: false, showticklabels: false },
                yaxis: { showgrid: false, zeroline: false, showticklabels: false }
            });
        }
        
        // 获取过滤后的节点
        function getFilteredNodes() {
            const selectedTypes = Array.from(document.getElementById('node-type-filter').selectedOptions)
                .map(option => option.value);
            return nodesData.filter(node => selectedTypes.includes(node.type));
        }
        
        // 获取过滤后的边
        function getFilteredEdges() {
            const selectedTypes = Array.from(document.getElementById('edge-type-filter').selectedOptions)
                .map(option => option.value);
            const filteredNodes = getFilteredNodes();
            const filteredNodeIds = new Set(filteredNodes.map(n => n.id));
            
            return edgesData.filter(edge => 
                selectedTypes.includes(edge.rel) &&
                filteredNodeIds.has(edge.src) &&
                filteredNodeIds.has(edge.dst)
            );
        }
        
        // 计算节点位置
        function calculatePositions(nodes) {
            const positions = {};
            const items = nodes.filter(n => n.type === 'item');
            const constructs = nodes.filter(n => n.type === 'construct');
            const clusters = nodes.filter(n => n.type === 'cluster');
            
            // 项目节点 - 圆形排列
            items.forEach((item, i) => {
                const angle = 2 * Math.PI * i / items.length;
                const radius = 3;
                positions[item.id] = [radius * Math.cos(angle), radius * Math.sin(angle)];
            });
            
            // 构建节点 - 上部排列
            constructs.forEach((construct, i) => {
                const x = (i - constructs.length/2) * 2;
                positions[construct.id] = [x, 5];
            });
            
            // 聚类节点 - 下部排列
            clusters.forEach((cluster, i) => {
                const x = (i - clusters.length/2) * 3;
                positions[cluster.id] = [x, -5];
            });
            
            return positions;
        }
        
        // 获取节点颜色
        function getNodeColor(nodeType) {
            const colors = {
                'item': '#FF6B6B',
                'construct': '#4ECDC4',
                'cluster': '#45B7D1'
            };
            return colors[nodeType] || '#95A5A6';
        }
        
        // 获取边颜色
        function getEdgeColor(relType) {
            const colors = {
                'item_construct': '#E74C3C',
                'item_cluster': '#3498DB',
                'construct_cluster': '#2ECC71'
            };
            return colors[relType] || '#95A5A6';
        }
        
        // 导出数据
        function exportData(type) {
            let data, filename;
            if (type === 'nodes') {
                data = nodesData;
                filename = 'hypergraph_nodes.csv';
            } else if (type === 'edges') {
                data = edgesData;
                filename = 'hypergraph_edges.csv';
            } else {
                data = { nodes: nodesData, edges: edgesData };
                filename = 'hypergraph_all.json';
            }
            
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }
        
        // 下载图片
        function downloadImage() {
            Plotly.downloadImage('graph-container', {
                format: 'png',
                width: 1200,
                height: 800,
                filename: 'hypergraph_visualization'
            });
        }
        
        // 事件监听
        document.getElementById('node-type-filter').addEventListener('change', createVisualization);
        document.getElementById('edge-type-filter').addEventListener('change', createVisualization);
        document.getElementById('node-size').addEventListener('input', createVisualization);
        
        // 初始化可视化
        createVisualization();
    </script>
</body>
</html>
        """
        return html_template
    
    def export_data(self, output_dir: str = "data/processed"):
        """导出超图数据"""
        if self.nodes_df is None or self.edges_df is None:
            self.load_data()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 导出CSV
        nodes_file = os.path.join(output_dir, "hypergraph_nodes.csv")
        edges_file = os.path.join(output_dir, "hypergraph_edges.csv")
        
        self.nodes_df.to_csv(nodes_file, index=False, encoding='utf-8')
        self.edges_df.to_csv(edges_file, index=False, encoding='utf-8')
        
        print(f"数据已导出到:")
        print(f"  节点: {nodes_file}")
        print(f"  边: {edges_file}")
        
        # 导出JSON格式
        json_file = os.path.join(output_dir, "hypergraph_data.json")
        data = {
            "nodes": self.nodes_df.to_dict('records'),
            "edges": self.edges_df.to_dict('records'),
            "metadata": {
                "total_nodes": len(self.nodes_df),
                "total_edges": len(self.edges_df),
                "node_types": self.nodes_df['type'].value_counts().to_dict(),
                "edge_types": self.edges_df['rel'].value_counts().to_dict()
            }
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"  JSON: {json_file}")
        
        return {
            "nodes_file": nodes_file,
            "edges_file": edges_file,
            "json_file": json_file
        }

def main():
    """主函数"""
    print("🔗 超图可视化工具启动")
    
    # 创建可视化器
    visualizer = HypergraphVisualizer()
    
    # 尝试加载真实数据
    visualizer.load_data(
        "data/processed/hyper_nodes.csv",
        "data/processed/hyper_edges.csv"
    )
    
    # 创建多种可视化
    print("\n📊 创建可视化...")
    
    # 1. Plotly交互式可视化
    try:
        visualizer.create_plotly_visualization()
    except Exception as e:
        print(f"Plotly可视化失败: {e}")
    
    # 2. NetworkX静态可视化
    try:
        visualizer.create_networkx_visualization()
    except Exception as e:
        print(f"NetworkX可视化失败: {e}")
    
    # 3. Web探索界面
    try:
        visualizer.create_web_interface()
    except Exception as e:
        print(f"Web界面创建失败: {e}")
    
    # 4. 导出数据
    try:
        visualizer.export_data()
    except Exception as e:
        print(f"数据导出失败: {e}")
    
    print("\n✅ 可视化完成!")
    print("生成的文件:")
    print("  - data/docs/hypergraph_interactive.html (交互式可视化)")
    print("  - data/docs/hypergraph_static.png (静态图片)")
    print("  - data/docs/hypergraph_explorer.html (Web探索界面)")
    print("  - data/processed/hypergraph_*.csv (数据文件)")

if __name__ == "__main__":
    main()
