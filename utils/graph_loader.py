import yaml
import networkx as nx

def load_graph(path):
    """Load a knowledge graph from YAML into NetworkX."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for node in data.get('nodes', []):
        G.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
    
    # Add edges with attributes
    for edge in data.get('edges', []):
        G.add_edge(edge['from'], edge['to'], **{k: v for k, v in edge.items() if k not in ['from', 'to']})
    
    return G

def load_excerpts(excerpts_dir):
    """Load text excerpts from directory."""
    import os
    texts = []
    for filename in os.listdir(excerpts_dir):
        if filename.endswith(('.txt', '.md')):
            with open(os.path.join(excerpts_dir, filename), 'r') as f:
                texts.append(f.read())
    return texts

def load_metas(metas_path):
    """Load metadata from JSON file."""
    import json
    with open(metas_path, 'r') as f:
        return json.load(f)