import yaml
import networkx as nx

def load_graph(path):
    """Load a knowledge graph from YAML into NetworkX."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    G = nx.DiGraph()
    
    # Handle new structure with 'topics' and 'prerequisites'
    if 'topics' in data:
        # New structure: topics with prerequisites
        for topic_data in data['topics']:
            topic_name = topic_data['topic']
            prerequisites = topic_data.get('prerequisites', [])
            description = topic_data.get('description', f"Mathematical topic: {topic_name}")
            
            # Add node with topic name as both id and title
            # Use description if available, otherwise create default summary
            G.add_node(topic_name, 
                      title=topic_name,
                      type='Topic',
                      summary=description,
                      weight=0.5,  # Default weight
                      difficulty='intermediate')
            
            # Add edges from prerequisites to this topic
            for prereq in prerequisites:
                G.add_edge(prereq, topic_name, 
                          type='prerequisite',
                          weight=1.0)
    
    # Handle old structure with 'nodes' and 'edges' (for backward compatibility)
    elif 'nodes' in data:
        # Old structure: nodes with explicit edges
        for node in data.get('nodes', []):
            G.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
        
        for edge in data.get('edges', []):
            G.add_edge(edge['from'], edge['to'], **{k: v for k, v in edge.items() if k not in ['from', 'to']})
    
    print(f"📊 Loaded graph: {G.number_of_nodes()} topics, {G.number_of_edges()} prerequisite relationships")
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