# graph_viz.py - PyVis graph builder and helpers

from pyvis.network import Network
import tempfile
import os

def draw_graph(G):
    """Build and return PyVis HTML for a NetworkX graph."""
    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
    
    # Add nodes with styling
    for node_id, node_data in G.nodes(data=True):
        title = node_data.get('title', node_id)
        node_type = node_data.get('type', 'Unknown')
        summary = node_data.get('summary', '')
        
        # Color by type
        color_map = {
            'Concept': '#ff6b6b',
            'Approach': '#4ecdc4', 
            'Method': '#45b7d1',
            'Tool': '#96ceb4',
            'Topic': '#74b9ff'  # Added Topic type with blue color
        }
        color = color_map.get(node_type, '#ffeaa7')
        
        net.add_node(node_id, 
                    label=title,
                    title=f"{title}\nType: {node_type}\n{summary}",
                    color=color)
    
    # Add edges with styling
    for source, target, edge_data in G.edges(data=True):
        edge_type = edge_data.get('type', 'related_to')
        net.add_edge(source, target, title=edge_type)
    
    # Configure physics
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
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

def build_pyvis_graph(G):
    """Alternative function name for compatibility."""
    return draw_graph(G)
