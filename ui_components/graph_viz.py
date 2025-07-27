# graph_viz.py - PyVis graph builder and helpers

from pyvis.network import Network
import tempfile
import os

def draw_graph(G, highlight_info=None):
    """Build and return PyVis HTML for a NetworkX graph with optional node highlighting.
    
    Args:
        G: NetworkX graph
        highlight_info: Dict with highlighting information:
            {
                'direct_matches': [list of node IDs],
                'graph_related': [list of node IDs],
                'target_topic': node_id (optional)
            }
    """
    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
    
    # Extract highlighting information
    direct_matches = set()
    graph_related = set()
    target_topic = None
    
    if highlight_info:
        direct_matches = set(highlight_info.get('direct_matches', []))
        graph_related = set(highlight_info.get('graph_related', []))
        target_topic = highlight_info.get('target_topic')
    
    # Add nodes with styling and highlighting
    for node_id, node_data in G.nodes(data=True):
        title = node_data.get('title', node_id)
        node_type = node_data.get('type', 'Unknown')
        summary = node_data.get('summary', '')
        weight = node_data.get('weight', 0.0)
        
        # Base color by type
        base_color_map = {
            'Concept': '#ff6b6b',
            'Approach': '#4ecdc4', 
            'Method': '#45b7d1',
            'Tool': '#96ceb4',
            'Topic': '#74b9ff',
            'Category': '#9b59b6'
        }
        base_color = base_color_map.get(node_type, '#ffeaa7')
        
        # Determine highlighting and styling
        node_color = base_color
        border_color = base_color
        border_width = 1
        node_size = 10
        font_size = 12
        
        # Create enhanced tooltip
        tooltip_parts = [f"{title}", f"Type: {node_type}"]
        if summary:
            tooltip_parts.append(f"Summary: {summary}")
        if weight > 0:
            tooltip_parts.append(f"Weight: {weight:.3f}")
        
        # Apply highlighting styles
        if node_id == target_topic:
            # Target topic - special gold highlighting
            node_color = '#FFD700'  # Gold
            border_color = '#FF8C00'  # Dark orange
            border_width = 4
            node_size = 20
            font_size = 14
            tooltip_parts.append("🎯 TARGET TOPIC")
            
        elif node_id in direct_matches:
            # Direct matches - bright green highlighting
            node_color = '#00FF7F'  # Spring green
            border_color = '#228B22'  # Forest green
            border_width = 3
            node_size = 15
            font_size = 13
            tooltip_parts.append("🎯 DIRECT MATCH (High Semantic Similarity)")
            
        elif node_id in graph_related:
            # Graph-related - orange highlighting
            node_color = '#FF6347'  # Tomato
            border_color = '#FF4500'  # Orange red
            border_width = 2
            node_size = 12
            font_size = 12
            tooltip_parts.append("🔗 GRAPH RELATED (Connected via Graph Traversal)")
        
        # Create final tooltip
        tooltip = "\n".join(tooltip_parts)
        
        net.add_node(node_id, 
                    label=title,
                    title=tooltip,
                    color={'background': node_color, 'border': border_color},
                    borderWidth=border_width,
                    size=node_size,
                    font={'size': font_size})
    
    # Add edges with styling
    for source, target, edge_data in G.edges(data=True):
        edge_type = edge_data.get('type', 'related_to')
        
        # Highlight edges connecting to matched nodes
        edge_color = '#666666'  # Default gray
        edge_width = 1
        
        # Check if edge connects highlighted nodes
        if (source in direct_matches or target in direct_matches or 
            source in graph_related or target in graph_related or
            source == target_topic or target == target_topic):
            edge_color = '#FFFFFF'  # White for highlighted connections
            edge_width = 2
        
        net.add_edge(source, target, 
                    title=edge_type,
                    color=edge_color,
                    width=edge_width)
    
    # Configure physics with better layout for highlighted nodes
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 150},
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200
      }
    }
    """)
    
    # Generate HTML
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        net.save_graph(tmp_file.name)
        with open(tmp_file.name, 'r', encoding='utf-8') as f:
            html_content = f.read()
        os.remove(tmp_file.name)
        return html_content

def build_pyvis_graph(G, highlight_info=None):
    """Alternative function name for compatibility."""
    return draw_graph(G, highlight_info)
