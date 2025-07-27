from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
import re
import os
import torch
import networkx as nx

# Use Qwen2.5-0.5B-Instruct for text generation
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# =============================================================================
# PROMPT TEMPLATES - For problem generation and assessment only
# =============================================================================

MATH_PROBLEM_PROMPT = """You are a mathematics educator. Create a specific math problem to test knowledge of: {topic}

Topic: {topic}
Difficulty Level: {level}
Context: This is to assess prerequisite knowledge for learning {target_goal}

Create ONE clear, specific math problem that tests understanding of {topic}.
Make it:
- Concrete with specific numbers/examples
- Appropriate for {level} level
- Solvable in a few steps
- Tests core concepts of {topic}
- ONLY provide the problem statement, NO solution or answer

Important: Do NOT include the solution, answer, or any hints about how to solve it. Only provide the problem statement.

Problem:"""

PROBLEM_SOLUTION_PROMPT = """You are a mathematics educator. Solve this math problem step by step.

Problem: {problem}
Topic: {topic}

Provide a clear, step-by-step solution showing all work. Be thorough and educational.

Solution:"""

ANSWER_ASSESSMENT_PROMPT = """You are a mathematics educator who assesses student answers by comparing them to correct solutions.

Problem: {problem}
Correct Solution: {correct_solution}
Student Answer: {student_answer}
Topic: {topic}

Compare the student's answer to the correct solution and provide an assessment.

Respond in this format:
ASSESSMENT: [CORRECT/PARTIAL/INCORRECT]
EXPLANATION: [Brief explanation of why the answer is correct, partially correct, or incorrect]

Assessment:"""

ROADMAP_GENERATION_PROMPT = """You are a mathematics educator who creates personalized learning paths based on student assessments.

Learning Goal: {goal}
Target Topic: {target_topic}
Current Level: {level}

Assessment Results:
{assessment_results}

Known Topics: {known_topics}
Topics to Learn: {unknown_topics}

Create a step-by-step learning roadmap from the student's current knowledge to their goal.
Each step should be specific and actionable.

Format each step as:
Step X: [Topic Name] - [Learning Activity] - [Skill to Master]

Provide 4-6 steps maximum. Focus on the logical progression from known to unknown topics.

Roadmap:"""

# =============================================================================
# CORE LLM FUNCTIONS
# =============================================================================

# Global pipeline instance for efficiency
_pipeline = None

def get_pipeline():
    """Get or initialize the Hugging Face pipeline for text generation."""
    global _pipeline
    if _pipeline is None:
        try:
            print(f"🤖 Loading {model_id} model...")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Create pipeline
            _pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            _pipeline = None
    
    return _pipeline

def _create_qwen_prompt(system_prompt: str, user_input: str) -> str:
    """Create a properly formatted prompt for Qwen model."""
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

def generate_math_problems(target_topic: str, related_topics: list, level: str, target_goal: str) -> list:
    """Generate math problems for assessment using LLM."""
    pipe = get_pipeline()
    
    if pipe is None:
        return []
    
    problems = []
    system_prompt = "You are a mathematics educator who creates assessment problems."
    
    # Generate problems for related topics (limit to 3-4 problems)
    topics_to_test = related_topics[:4] if len(related_topics) > 4 else related_topics
    
    for topic in topics_to_test:
        user_prompt = MATH_PROBLEM_PROMPT.format(
            topic=topic,
            level=level,
            target_goal=target_goal
        )
        full_prompt = _create_qwen_prompt(system_prompt, user_prompt)
        
        try:
            result = pipe(full_prompt, max_new_tokens=200)
            generated_text = result[0]["generated_text"]
            
            # Extract only the assistant's response
            assistant_response = generated_text.split("<|im_start|>assistant")[-1].strip()
            
            # Clean up the problem text
            problem_text = assistant_response.replace("Problem:", "").strip()
            problem_text = _filter_solutions_from_problem(problem_text)
            
            if len(problem_text) > 10 and not _contains_solution_indicators(problem_text):
                problems.append({
                    'topic': topic,
                    'problem': problem_text,
                    'level': level
                })
            
        except Exception as e:
            print(f"Error generating problem for {topic}: {e}")
            continue
    
    return problems

def _filter_solutions_from_problem(problem_text: str) -> str:
    """Remove any solution hints or answers that may have leaked into the problem."""
    # Remove common solution indicators
    lines = problem_text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip lines that look like solutions
        if any(indicator in line.lower() for indicator in [
            'solution:', 'answer:', 'result:', 'therefore:', 'thus:', 'so the answer is',
            'the solution is', 'solving:', 'step 1:', 'step 2:', 'first,', 'next,'
        ]):
            break
        if line:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines).strip()

def _contains_solution_indicators(text: str) -> bool:
    """Check if text contains solution indicators."""
    solution_words = ['=', 'answer', 'solution', 'result', 'therefore', 'thus', 'step 1', 'first,']
    return any(word in text.lower() for word in solution_words)

def assess_student_answer(problem: str, student_answer: str, topic: str) -> dict:
    """Assess student's answer using LLM by first solving the problem internally."""
    pipe = get_pipeline()
    
    if pipe is None:
        # Fallback assessment
        return {
            'assessment': 'PARTIAL',
            'explanation': 'Unable to assess - LLM not available'
        }
    
    # First, solve the problem internally to get the correct solution
    correct_solution = _solve_problem_with_llm(problem, topic, pipe)
    
    # Now assess the student's answer against the correct solution
    system_prompt = "You are a mathematics educator who assesses student answers by comparing them to correct solutions."
    user_prompt = ANSWER_ASSESSMENT_PROMPT.format(
        problem=problem,
        correct_solution=correct_solution,
        student_answer=student_answer,
        topic=topic
    )
    full_prompt = _create_qwen_prompt(system_prompt, user_prompt)
    
    try:
        result = pipe(full_prompt, max_new_tokens=200)
        generated_text = result[0]["generated_text"]
        
        # Extract only the assistant's response
        assistant_response = generated_text.split("<|im_start|>assistant")[-1].strip()
        
        # Parse the assessment
        assessment = {'assessment': 'PARTIAL', 'explanation': 'Could not parse assessment'}
        
        for line in assistant_response.split('\n'):
            if 'ASSESSMENT:' in line:
                assessment['assessment'] = line.split('ASSESSMENT:')[1].strip()
            elif 'EXPLANATION:' in line:
                assessment['explanation'] = line.split('EXPLANATION:')[1].strip()
        
        return assessment
        
    except Exception as e:
        print(f"Error assessing answer: {e}")
        return {
            'assessment': 'PARTIAL',
            'explanation': f'Assessment error: {str(e)}'
        }

def _solve_problem_with_llm(problem: str, topic: str, pipe) -> str:
    """Internal function to solve a math problem using LLM."""
    system_prompt = "You are a mathematics educator who solves problems step by step."
    user_prompt = PROBLEM_SOLUTION_PROMPT.format(problem=problem, topic=topic)
    full_prompt = _create_qwen_prompt(system_prompt, user_prompt)
    
    try:
        result = pipe(full_prompt, max_new_tokens=300)
        generated_text = result[0]["generated_text"]
        
        # Extract only the assistant's response
        solution = generated_text.split("<|im_start|>assistant")[-1].strip()
        return solution
        
    except Exception as e:
        print(f"Error solving problem: {e}")
        return "Unable to solve problem internally"

def generate_learning_roadmap(intent: dict, assessment_results: dict, known_topics: list, unknown_topics: list) -> dict:
    """Generate a learning roadmap from known topics to the goal using LLM."""
    pipe = get_pipeline()
    
    if pipe is None:
        # Simple fallback roadmap
        steps = {}
        for i, topic in enumerate(unknown_topics[:6], 1):
            steps[f"Step {i}"] = [
                f"📚 **Topic**: {topic}",
                f"🎯 **Activity**: Learn fundamental concepts of {topic}",
                f"💡 **Skill**: Understand and apply {topic} principles"
            ]
        return steps

    system_prompt = "You are a mathematics educator who creates personalized learning paths based on student assessments."
    
    # Format assessment results
    assessment_summary = []
    for topic, result in assessment_results.items():
        assessment_summary.append(f"{topic}: {result['assessment']} - {result['explanation']}")
    
    user_prompt = ROADMAP_GENERATION_PROMPT.format(
        goal=intent.get('goal', ''),
        target_topic=intent.get('target_topic', ''),
        level=intent.get('current_level', 'beginner'),
        assessment_results='\n'.join(assessment_summary),
        known_topics=', '.join(known_topics) if known_topics else 'None identified',
        unknown_topics=', '.join(unknown_topics) if unknown_topics else 'None identified'
    )
    
    full_prompt = _create_qwen_prompt(system_prompt, user_prompt)
    
    try:
        result = pipe(full_prompt, max_new_tokens=600)
        generated_text = result[0]["generated_text"]
        
        # Extract only the assistant's response
        assistant_response = generated_text.split("<|im_start|>assistant")[-1].strip()
        
        # Parse roadmap steps
        roadmap = {}
        lines = assistant_response.split('\n')
        
        for line in lines:
            line = line.strip()
            step_match = re.match(r'Step (\d+):\s*(.+)', line)
            if step_match:
                step_num = step_match.group(1)
                step_content = step_match.group(2)
                
                # Parse the detailed format: [topic] - [activity] - [skill]
                parts = step_content.split(' - ')
                if len(parts) >= 3:
                    topic = parts[0].strip()
                    activity = parts[1].strip()
                    skill = parts[2].strip()
                    roadmap[f"Step {step_num}"] = [f"📚 **Topic**: {topic}", f"🎯 **Activity**: {activity}", f"💡 **Skill**: {skill}"]
                else:
                    # Fallback to single content if format doesn't match
                    roadmap[f"Step {step_num}"] = [step_content]
        
        # If no steps found, create a basic structure
        if not roadmap:
            for i, topic in enumerate(unknown_topics[:6], 1):
                roadmap[f"Step {i}"] = [
                    f"📚 **Topic**: {topic}",
                    f"🎯 **Activity**: Learn fundamental concepts of {topic}",
                    f"💡 **Skill**: Understand and apply {topic} principles"
                ]
        
        return roadmap
        
    except Exception as e:
        print(f"Error generating roadmap: {e}")
        # Fallback roadmap
        steps = {}
        for i, topic in enumerate(unknown_topics[:6], 1):
            steps[f"Step {i}"] = [
                f"📚 **Topic**: {topic}",
                f"🎯 **Activity**: Learn fundamental concepts of {topic}",
                f"💡 **Skill**: Understand and apply {topic} principles"
            ]
        return steps

# =============================================================================
# SEMANTIC TOPIC MATCHING (using Sentence Transformers)
# =============================================================================

from sentence_transformers import SentenceTransformer, util
import torch

# Global model instance for efficiency
_semantic_model = None

def get_semantic_model():
    """Get or initialize the sentence transformer model."""
    global _semantic_model
    if _semantic_model is None:
        print("🤖 Loading sentence transformer model...")
        _semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Sentence transformer model loaded successfully")
    return _semantic_model

def find_related_topics_by_semantic_similarity(user_input: str, graph: nx.Graph, max_direct: int = 5, min_similarity: float = 0.3) -> dict:
    """Find related topics using semantic similarity with sentence transformers, then expand using graph structure."""
    
    # Get the semantic model
    model = get_semantic_model()
    
    # Encode user input
    print(f"\n🔍 Semantic Topic Matching for: '{user_input}'")
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    
    # Collect all topic texts and metadata
    topic_data = []
    topic_texts = []
    
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get('type') != 'Topic':
            continue
            
        title = node_data.get('title', node_id)
        summary = node_data.get('summary', '')
        
        # Create a comprehensive text representation for each topic
        topic_text = f"{title}. {summary}".strip()
        if topic_text.endswith('.'):
            topic_text = topic_text[:-1]  # Remove double period
            
        topic_texts.append(topic_text)
        topic_data.append({
            'id': node_id,
            'title': title,
            'summary': summary,
            'difficulty': node_data.get('difficulty', 'intermediate'),
            'weight': node_data.get('weight', 0.0),  # Include weight from new YAML
            'text': topic_text
        })
    
    if not topic_texts:
        print("❌ No topics found in graph")
        return {'direct_matches': [], 'graph_related': []}
    
    print(f"📊 Analyzing {len(topic_texts)} topics from knowledge graph...")
    
    # Encode all topic texts
    topic_embeddings = model.encode(topic_texts, convert_to_tensor=True)
    
    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(user_embedding, topic_embeddings)[0]
    
    # Create results with similarity scores
    all_results = []
    for i, topic in enumerate(topic_data):
        similarity_score = similarities[i].item()
        all_results.append({
            'id': topic['id'],
            'title': topic['title'],
            'summary': topic['summary'],
            'difficulty': topic['difficulty'],
            'weight': topic['weight'],
            'similarity_score': similarity_score,
            'text': topic['text']
        })
    
    # Filter by minimum similarity threshold
    filtered_results = [r for r in all_results if r['similarity_score'] >= min_similarity]
    
    # Sort by similarity score (highest first)
    filtered_results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Take top direct matches within the limit (reduced to 5)
    direct_matches = filtered_results[:max_direct]
    
    # Debug output for direct matches
    print(f"🎯 Found {len(filtered_results)} topics above {min_similarity} similarity threshold")
    print(f"📋 Top {len(direct_matches)} direct matches:")
    for i, topic in enumerate(direct_matches):
        print(f"{i+1:2d}. {topic['title']:30s} (similarity: {topic['similarity_score']:.3f}, weight: {topic['weight']:.3f}) - {topic['summary'][:50]}...")
    
    # Now find graph-related topics (second-level nodes) with weight prioritization
    graph_related_topics = find_graph_related_topics_weighted(graph, direct_matches, max_related=10)
    
    return {
        'direct_matches': direct_matches,
        'graph_related': graph_related_topics,
        'total_filtered': len(filtered_results),
        'similarity_threshold': min_similarity
    }

def find_graph_related_topics_weighted(graph: nx.Graph, seed_topics: list, max_related: int = 10) -> list:
    """Find related topics by traversing graph structure, prioritizing by shortest path + highest weight + similarity path quality."""
    
    print(f"\n🗺️ Finding graph-related topics from {len(seed_topics)} seed topics (prioritizing by path length + weight + similarity path quality)...")
    
    # Create a mapping of seed topic IDs to their similarity scores for path quality calculation
    seed_similarity_map = {topic['id']: topic['similarity_score'] for topic in seed_topics}
    
    # Collect all neighbor nodes with path information, weights, and path quality
    related_candidates = {}
    seed_ids = {topic['id'] for topic in seed_topics}
    
    # For each seed topic, find connected topics with path lengths and quality scores
    for seed_topic in seed_topics:
        seed_id = seed_topic['id']
        seed_similarity = seed_topic['similarity_score']
        
        # Use BFS to find topics within 2 steps, tracking path lengths and intermediate nodes
        visited = set()
        queue = [(seed_id, 0, [], seed_similarity)]  # (node_id, distance, path, path_quality)
        visited.add(seed_id)
        
        while queue:
            current_node, distance, path, current_path_quality = queue.pop(0)
            
            if distance >= 2:  # Don't go beyond 2 steps
                continue
                
            # Get neighbors (both directions for directed graphs)
            neighbors = set()
            
            # Get successors
            for neighbor in graph.neighbors(current_node):
                neighbors.add(neighbor)
                
            # For directed graphs, also get predecessors
            if graph.is_directed():
                for neighbor in graph.predecessors(current_node):
                    neighbors.add(neighbor)
            
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in seed_ids:
                    visited.add(neighbor)
                    new_distance = distance + 1
                    new_path = path + [current_node]
                    
                    # Calculate path quality based on intermediate nodes
                    if new_distance == 1:
                        # Direct connection - path quality is just the seed similarity
                        new_path_quality = seed_similarity
                    else:
                        # 2-step connection - consider the quality of the intermediate node
                        intermediate_node = new_path[-1]  # The node we're going through
                        
                        # If intermediate node is also a seed topic, use its similarity
                        if intermediate_node in seed_similarity_map:
                            intermediate_quality = seed_similarity_map[intermediate_node]
                        else:
                            # Use the weight of the intermediate node as a proxy for quality
                            intermediate_data = graph.nodes.get(intermediate_node, {})
                            intermediate_weight = intermediate_data.get('weight', 0.0)
                            # Convert weight to a similarity-like score (0.0-1.0 range)
                            intermediate_quality = min(intermediate_weight, 1.0)
                        
                        # Path quality is average of seed similarity and intermediate quality
                        new_path_quality = (seed_similarity + intermediate_quality) / 2
                    
                    if new_distance <= 2:  # Only consider topics within 2 steps
                        queue.append((neighbor, new_distance, new_path, new_path_quality))
                        
                        # Track this as a candidate
                        if neighbor not in related_candidates:
                            related_candidates[neighbor] = {
                                'min_distance': new_distance,
                                'max_path_quality': new_path_quality,
                                'connections': [],
                                'connection_count': 0,
                                'best_path_info': None
                            }
                        
                        # Update minimum distance and maximum path quality
                        if new_distance < related_candidates[neighbor]['min_distance']:
                            related_candidates[neighbor]['min_distance'] = new_distance
                        
                        if new_path_quality > related_candidates[neighbor]['max_path_quality']:
                            related_candidates[neighbor]['max_path_quality'] = new_path_quality
                            related_candidates[neighbor]['best_path_info'] = {
                                'seed_topic': seed_topic['title'],
                                'seed_similarity': seed_similarity,
                                'path_quality': new_path_quality,
                                'distance': new_distance,
                                'intermediate_nodes': new_path[1:] if len(new_path) > 1 else []
                            }
                        
                        related_candidates[neighbor]['connections'].append({
                            'seed_topic': seed_topic['title'],
                            'seed_id': seed_id,
                            'seed_similarity': seed_similarity,
                            'distance': new_distance,
                            'path_quality': new_path_quality,
                            'path': new_path + [neighbor]
                        })
                        related_candidates[neighbor]['connection_count'] += 1
    
    print(f"  📍 Found {len(related_candidates)} candidate topics within 2 steps")
    
    # Convert to topic data with weights and enhanced prioritization scoring
    graph_related_topics = []
    for node_id, connection_info in related_candidates.items():
        if node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            if node_data.get('type') == 'Topic':
                
                # Get topic weight (higher is better)
                topic_weight = node_data.get('weight', 0.0)
                
                # Enhanced priority calculation with path quality
                min_distance = connection_info['min_distance']
                connection_count = connection_info['connection_count']
                max_path_quality = connection_info['max_path_quality']
                
                # Enhanced priority score: 
                # - topic weight (0-1): intrinsic importance
                # - distance bonus (2 for distance 1, 1 for distance 2): shorter is better
                # - connection bonus (0.1 per connection): more connections is better  
                # - path quality bonus (0-1): higher similarity path is better
                priority_score = (
                    topic_weight + 
                    (3 - min_distance) + 
                    (connection_count * 0.1) +
                    max_path_quality  # This is the new component!
                )
                
                # Create enhanced connection paths for display
                connection_paths = []
                best_path_info = connection_info.get('best_path_info')
                
                # Show the best quality path first
                if best_path_info:
                    if best_path_info['distance'] == 1:
                        path_desc = f"{best_path_info['seed_topic']} → {node_data.get('title', node_id)} (1 step, sim: {best_path_info['seed_similarity']:.3f})"
                    else:
                        intermediate_names = [graph.nodes.get(n, {}).get('title', n) for n in best_path_info['intermediate_nodes']]
                        path_desc = f"{best_path_info['seed_topic']} → {' → '.join(intermediate_names)} → {node_data.get('title', node_id)} ({best_path_info['distance']} steps, quality: {best_path_info['path_quality']:.3f})"
                    connection_paths.append(f"🏆 BEST: {path_desc}")
                
                # Add other example paths (up to 2 more)
                other_connections = [conn for conn in connection_info['connections'] 
                                   if conn.get('path_quality', 0) != max_path_quality][:2]
                for conn in other_connections:
                    if conn['distance'] == 1:
                        path_desc = f"{conn['seed_topic']} → {node_data.get('title', node_id)} (1 step, sim: {conn['seed_similarity']:.3f})"
                    else:
                        path_desc = f"{conn['seed_topic']} → ... → {node_data.get('title', node_id)} ({conn['distance']} steps, quality: {conn['path_quality']:.3f})"
                    connection_paths.append(path_desc)
                
                graph_related_topics.append({
                    'id': node_id,
                    'title': node_data.get('title', node_id),
                    'summary': node_data.get('summary', ''),
                    'difficulty': node_data.get('difficulty', 'intermediate'),
                    'weight': topic_weight,
                    'min_distance': min_distance,
                    'connection_count': connection_count,
                    'max_path_quality': max_path_quality,
                    'priority_score': priority_score,
                    'connection_paths': connection_paths,
                    'graph_distance': f"{min_distance}_steps",
                    'best_path_info': best_path_info
                })
    
    # Sort by enhanced priority score (highest first: high weight + short distance + many connections + high path quality)
    graph_related_topics.sort(key=lambda x: x['priority_score'], reverse=True)
    
    # Limit results to max_related (10)
    graph_related_topics = graph_related_topics[:max_related]
    
    print(f"🔗 Selected top {len(graph_related_topics)} graph-related topics (by weight + path + similarity quality):")
    for i, topic in enumerate(graph_related_topics[:5]):  # Show top 5 in debug
        print(f"  {i+1}. {topic['title']:30s} (weight: {topic['weight']:.3f}, dist: {topic['min_distance']}, quality: {topic['max_path_quality']:.3f}, priority: {topic['priority_score']:.3f})")
    
    return graph_related_topics

def conduct_semantic_assessment(goal: str, graph: nx.Graph) -> dict:
    """Semantic assessment: uses sentence transformers + graph traversal with weight prioritization."""
    
    print(f"\n🎯 Analyzing goal with semantic similarity + weighted graph traversal: '{goal}'")
    
    # Semantic topic matching with graph expansion (5 direct + 10 graph-related = 15 total)
    topic_results = find_related_topics_by_semantic_similarity(goal, graph, max_direct=5, min_similarity=0.3)
    
    direct_matches = topic_results['direct_matches']
    graph_related = topic_results['graph_related']
    
    # Create enhanced intent with both semantic and graph analysis info
    semantic_intent = {
        'goal': goal,
        'analysis_method': 'semantic_similarity_plus_weighted_graph_traversal',
        'model_used': 'sentence-transformers/all-MiniLM-L6-v2',
        'direct_matches_found': len(direct_matches),
        'graph_related_found': len(graph_related),
        'similarity_threshold': topic_results['similarity_threshold'],
        'total_above_threshold': topic_results['total_filtered'],
        'top_similarity_score': direct_matches[0]['similarity_score'] if direct_matches else 0.0,
        'limits': {'direct_matches': 5, 'graph_related': 10, 'total': 15}
    }
    
    # Skip problem generation for debugging
    print("⏭️ Skipping problem generation for debugging")
    
    return {
        'intent': semantic_intent,
        'direct_matches': direct_matches,
        'graph_related': graph_related,
        'related_topics': direct_matches,  # For backward compatibility
        'target_topic': direct_matches[0]['id'] if direct_matches else None,
        'problems': [],  # Empty for debugging
        'debug_info': {
            'matching_method': 'semantic_similarity_plus_weighted_graph_traversal',
            'model_name': 'all-MiniLM-L6-v2',
            'user_input': goal,
            'total_graph_nodes': graph.number_of_nodes(),
            'similarity_threshold': 0.3,
            'result_limits': {'direct': 5, 'graph_related': 10, 'total': 15},
            'direct_similarity_range': {
                'highest': direct_matches[0]['similarity_score'] if direct_matches else 0.0,
                'lowest': direct_matches[-1]['similarity_score'] if direct_matches else 0.0
            },
            'graph_analysis': {
                'seed_topics': len(direct_matches),
                'related_topics_found': len(graph_related),
                'max_connections': max([t['connection_count'] for t in graph_related], default=0),
                'weight_range': {
                    'highest': max([t['weight'] for t in graph_related], default=0.0),
                    'lowest': min([t['weight'] for t in graph_related], default=0.0)
                } if graph_related else {'highest': 0.0, 'lowest': 0.0}
            }
        }
    }

# =============================================================================
# WIKIPEDIA CATEGORY EXTRACTION AND SELF-ASSESSMENT
# =============================================================================

def extract_wikipedia_categories(graph: nx.Graph) -> list:
    """Extract foundational topics (with 0-1 prerequisites) from the graph for self-assessment."""
    
    categories = []
    
    # Use topics with 0-1 prerequisites as foundational categories
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get('type') == 'Topic':
            # Count prerequisites (incoming edges)
            prerequisite_count = graph.in_degree(node_id)
            
            # Include topics with 0-1 prerequisites as "foundational categories"
            if prerequisite_count <= 1:
                categories.append({
                    'id': node_id,
                    'title': node_data.get('title', node_id),
                    'weight': node_data.get('weight', 0.5),  # Default weight
                    'prerequisite_count': prerequisite_count,
                    'summary': node_data.get('summary', f"Mathematical topic: {node_id}")
                })
    
    # Sort by prerequisite count (ascending) then by title for consistency
    categories.sort(key=lambda x: (x['prerequisite_count'], x['title']))
    
    print(f"📚 Found {len(categories)} foundational topics (0-1 prerequisites) for self-assessment")
    print("🏗️ Foundational topics by prerequisite count:")
    
    # Group by prerequisite count for display
    by_prereq_count = {}
    for cat in categories:
        count = cat['prerequisite_count']
        if count not in by_prereq_count:
            by_prereq_count[count] = []
        by_prereq_count[count].append(cat)
    
    for count in sorted(by_prereq_count.keys()):
        topics = by_prereq_count[count]
        print(f"  {count} prerequisites: {len(topics)} topics")
        for i, cat in enumerate(topics[:3]):  # Show first 3 examples
            print(f"    • {cat['title']}")
        if len(topics) > 3:
            print(f"    ... and {len(topics) - 3} more")
    
    return categories

def map_categories_to_related_nodes(graph: nx.Graph, category_knowledge: dict) -> dict:
    """Map Wikipedia categories to their related nodes based on graph layers (0-3 levels)."""
    
    print(f"\n🗺️ Mapping {len(category_knowledge)} categories to related nodes by graph layers...")
    
    known_nodes = set()
    partially_known_nodes = set()
    
    for category_id, knowledge_level in category_knowledge.items():
        if knowledge_level == 0:  # No knowledge
            continue
            
        # Find nodes at different graph distances based on knowledge level
        connected_nodes = set()
        
        # Level 1: Direct neighbors only
        if knowledge_level >= 1:
            for neighbor in graph.neighbors(category_id):
                if graph.nodes.get(neighbor, {}).get('type') == 'Topic':
                    connected_nodes.add(neighbor)
            
            # For directed graphs, also get predecessors
            if graph.is_directed():
                for neighbor in graph.predecessors(category_id):
                    if graph.nodes.get(neighbor, {}).get('type') == 'Topic':
                        connected_nodes.add(neighbor)
        
        # Level 2: 2-step connections
        if knowledge_level >= 2:
            # Get all direct neighbors first
            direct_neighbors = set()
            for neighbor in graph.neighbors(category_id):
                direct_neighbors.add(neighbor)
            if graph.is_directed():
                for neighbor in graph.predecessors(category_id):
                    direct_neighbors.add(neighbor)
            
            # Then get their neighbors
            for direct_neighbor in direct_neighbors:
                for second_neighbor in graph.neighbors(direct_neighbor):
                    if graph.nodes.get(second_neighbor, {}).get('type') == 'Topic':
                        connected_nodes.add(second_neighbor)
                if graph.is_directed():
                    for second_neighbor in graph.predecessors(direct_neighbor):
                        if graph.nodes.get(second_neighbor, {}).get('type') == 'Topic':
                            connected_nodes.add(second_neighbor)
        
        # Level 3: 3-step connections
        if knowledge_level >= 3:
            # Get 2-step neighbors first
            two_step_neighbors = set()
            direct_neighbors = set()
            for neighbor in graph.neighbors(category_id):
                direct_neighbors.add(neighbor)
            if graph.is_directed():
                for neighbor in graph.predecessors(category_id):
                    direct_neighbors.add(neighbor)
            
            for direct_neighbor in direct_neighbors:
                for second_neighbor in graph.neighbors(direct_neighbor):
                    two_step_neighbors.add(second_neighbor)
                if graph.is_directed():
                    for second_neighbor in graph.predecessors(direct_neighbor):
                        two_step_neighbors.add(second_neighbor)
            
            # Then get their neighbors (3-step)
            for two_step_neighbor in two_step_neighbors:
                for third_neighbor in graph.neighbors(two_step_neighbor):
                    if graph.nodes.get(third_neighbor, {}).get('type') == 'Topic':
                        connected_nodes.add(third_neighbor)
                if graph.is_directed():
                    for third_neighbor in graph.predecessors(two_step_neighbor):
                        if graph.nodes.get(third_neighbor, {}).get('type') == 'Topic':
                            connected_nodes.add(third_neighbor)
        
        # Categorize based on knowledge level
        if knowledge_level >= 3:  # Expert level (3+ steps)
            known_nodes.update(connected_nodes)
        elif knowledge_level >= 2:  # Good knowledge (2+ steps)
            known_nodes.update(connected_nodes)
        elif knowledge_level >= 1:  # Basic knowledge (1+ step)
            partially_known_nodes.update(connected_nodes)
        
        level_names = {1: "Basic", 2: "Good", 3: "Expert"}
        level_name = level_names.get(int(knowledge_level), "Unknown")
        
        print(f"  📂 {graph.nodes.get(category_id, {}).get('title', category_id):25s} "
              f"(level: {knowledge_level} - {level_name}) → {len(connected_nodes)} topics")
    
    # Remove overlaps (known takes precedence over partial)
    partially_known_nodes = partially_known_nodes - known_nodes
    
    print(f"\n✅ Knowledge mapping complete:")
    print(f"  🟢 Known topics: {len(known_nodes)}")
    print(f"  🟡 Partially known topics: {len(partially_known_nodes)}")
    
    return {
        'known_nodes': list(known_nodes),
        'partially_known_nodes': list(partially_known_nodes),
        'total_mapped': len(known_nodes) + len(partially_known_nodes)
    }

def generate_learning_path_from_known(graph: nx.Graph, known_nodes: list, target_topics: list, max_path_length: int = 6) -> dict:
    """Generate comprehensive learning path that covers ALL prerequisites of the target topic."""
    
    print(f"\n🎯 Generating comprehensive learning path from {len(known_nodes)} known nodes...")
    
    # If no known nodes provided, use topics with 0 prerequisites as starting points
    original_known_count = len(known_nodes)
    if not known_nodes:
        print("⚠️ No known nodes provided - using topics with 0 prerequisites as starting points")
        zero_prereq_topics = []
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get('type') == 'Topic' and graph.in_degree(node_id) == 0:
                zero_prereq_topics.append(node_id)
        
        if zero_prereq_topics:
            known_nodes = zero_prereq_topics
            print(f"✅ Found {len(zero_prereq_topics)} topics with 0 prerequisites to use as starting points:")
            for topic in zero_prereq_topics[:5]:  # Show first 5
                print(f"    • {topic}")
            if len(zero_prereq_topics) > 5:
                print(f"    ... and {len(zero_prereq_topics) - 5} more")
        else:
            print("❌ No topics with 0 prerequisites found")
            return {'path': [], 'message': 'No starting points found in the knowledge graph. The graph may be incomplete.'}
    
    if not target_topics:
        print("❌ No target topics provided")
        return {'path': [], 'message': 'No learning targets identified.'}
    
    # ONLY use the highest semantic matching score (first item in direct matches)
    if len(target_topics) > 0 and isinstance(target_topics[0], dict) and 'similarity_score' in target_topics[0]:
        # Sort by similarity score and take only the top one
        sorted_targets = sorted(target_topics, key=lambda x: x.get('similarity_score', 0), reverse=True)
        single_target = [sorted_targets[0]]  # Only the highest scoring target
        target_title = single_target[0]['title']
        target_id = single_target[0]['id']
        
        print(f"🎯 Focusing on HIGHEST semantic match only: '{target_title}' (similarity: {single_target[0]['similarity_score']:.3f})")
    else:
        # Fallback for non-dict targets
        single_target = target_topics[:1]
        target_title = single_target[0] if single_target else "Unknown"
        target_id = target_title
    
    # Convert known_nodes to set for faster lookup
    known_set = set(known_nodes)
    
    # If target is already known, return success message
    if target_id in known_set:
        if original_known_count > 0:
            return {
                'path': [],
                'message': f'Great! You already know: {target_title}. No additional learning needed!'
            }
        else:
            return {
                'path': [],
                'message': f'Target "{target_title}" is a foundational topic with no prerequisites. You can start learning it directly!'
            }
    
    if target_id not in graph.nodes():
        return {'path': [], 'message': f'Target topic "{target_title}" not found in knowledge graph.'}
    
    # Find ALL prerequisites for the target topic using topological traversal
    print(f"🔍 Finding all prerequisites for '{target_title}'...")
    
    def find_all_prerequisites(node_id, visited=None):
        """Recursively find all prerequisites for a given node."""
        if visited is None:
            visited = set()
        
        if node_id in visited:
            return set()  # Avoid cycles
        
        visited.add(node_id)
        all_prereqs = set()
        
        # Get direct prerequisites (predecessors in the graph)
        for prereq in graph.predecessors(node_id):
            if graph.nodes.get(prereq, {}).get('type') == 'Topic':
                all_prereqs.add(prereq)
                # Recursively find prerequisites of prerequisites
                sub_prereqs = find_all_prerequisites(prereq, visited.copy())
                all_prereqs.update(sub_prereqs)
        
        return all_prereqs
    
    # Get all prerequisites for the target
    all_prerequisites = find_all_prerequisites(target_id)
    print(f"📚 Found {len(all_prerequisites)} total prerequisites for '{target_title}'")
    
    # Filter out prerequisites that are already known
    unknown_prerequisites = all_prerequisites - known_set
    print(f"📖 {len(unknown_prerequisites)} prerequisites need to be learned (excluding {len(all_prerequisites - unknown_prerequisites)} already known)")
    
    if not unknown_prerequisites:
        return {
            'path': [{
                'id': target_id,
                'title': target_title,
                'summary': graph.nodes.get(target_id, {}).get('summary', ''),
                'weight': graph.nodes.get(target_id, {}).get('weight', 0.0),
                'difficulty': graph.nodes.get(target_id, {}).get('difficulty', 'intermediate'),
                'step_number': 1,
                'is_target': True,
                'is_starting_point': False,
                'prerequisite_level': 0
            }],
            'message': f'All prerequisites for "{target_title}" are already known! You can learn it directly.',
            'path_details': {
                'total_steps': 1,
                'target_topic': target_title,
                'prerequisites_covered': len(all_prerequisites),
                'already_known': len(all_prerequisites - unknown_prerequisites),
                'path_type': 'direct_to_target'
            }
        }
    
    # Create a topological ordering of the unknown prerequisites + target
    learning_sequence = []
    remaining_topics = unknown_prerequisites.copy()
    remaining_topics.add(target_id)
    
    # Keep track of prerequisite levels for better organization
    topic_levels = {}
    
    def calculate_prerequisite_level(topic_id, known_topics):
        """Calculate how many prerequisite layers deep a topic is."""
        if topic_id in known_topics:
            return 0
        
        max_level = 0
        for prereq in graph.predecessors(topic_id):
            if graph.nodes.get(prereq, {}).get('type') == 'Topic':
                if prereq in known_topics:
                    level = 1
                else:
                    level = calculate_prerequisite_level(prereq, known_topics) + 1
                max_level = max(max_level, level)
        
        return max_level
    
    # Calculate levels for all topics
    for topic in remaining_topics:
        topic_levels[topic] = calculate_prerequisite_level(topic, known_set)
    
    print(f"📊 Prerequisite levels calculated:")
    level_counts = {}
    for topic, level in topic_levels.items():
        level_counts[level] = level_counts.get(level, 0) + 1
    for level in sorted(level_counts.keys()):
        print(f"  Level {level}: {level_counts[level]} topics")
    
    # Build learning sequence by processing topics level by level
    processed = known_set.copy()
    current_level = 1
    
    while remaining_topics:
        # Find topics at current level that have all prerequisites satisfied
        ready_topics = []
        for topic in remaining_topics:
            if topic_levels[topic] == current_level:
                # Check if all prerequisites are satisfied
                prereqs_satisfied = True
                for prereq in graph.predecessors(topic):
                    if (graph.nodes.get(prereq, {}).get('type') == 'Topic' and 
                        prereq not in processed):
                        prereqs_satisfied = False
                        break
                
                if prereqs_satisfied:
                    ready_topics.append(topic)
        
        if not ready_topics:
            # Move to next level if no topics ready at current level
            current_level += 1
            if current_level > max(topic_levels.values()) + 1:
                # Safety break to avoid infinite loop
                print("⚠️ Breaking potential infinite loop in prerequisite resolution")
                break
            continue
        
        # Sort ready topics by weight (higher weight = more important)
        ready_topics.sort(key=lambda x: graph.nodes.get(x, {}).get('weight', 0.0), reverse=True)
        
        # Add ready topics to learning sequence
        for topic in ready_topics:
            node_data = graph.nodes.get(topic, {})
            learning_sequence.append({
                'id': topic,
                'title': node_data.get('title', topic),
                'summary': node_data.get('summary', ''),
                'weight': node_data.get('weight', 0.0),
                'difficulty': node_data.get('difficulty', 'intermediate'),
                'step_number': len(learning_sequence) + 1,
                'is_target': topic == target_id,
                'is_starting_point': False,
                'prerequisite_level': topic_levels[topic]
            })
            
            processed.add(topic)
            remaining_topics.remove(topic)
    
    print(f"🛤️ Generated comprehensive learning path with {len(learning_sequence)} steps:")
    for i, step in enumerate(learning_sequence[:10]):  # Show first 10 steps
        target_indicator = " 🎯" if step['is_target'] else ""
        level_indicator = f" (L{step['prerequisite_level']})"
        print(f"  {i+1:2d}. {step['title']:30s} (weight: {step['weight']:.3f}){level_indicator}{target_indicator}")
    if len(learning_sequence) > 10:
        print(f"    ... and {len(learning_sequence) - 10} more steps")
    
    return {
        'path': learning_sequence,
        'message': f'Generated comprehensive learning path with {len(learning_sequence)} steps covering all prerequisites for "{target_title}".',
        'path_details': {
            'total_steps': len(learning_sequence),
            'target_topic': target_title,
            'prerequisites_covered': len(all_prerequisites),
            'already_known': len(all_prerequisites - unknown_prerequisites),
            'max_prerequisite_level': max(topic_levels.values()) if topic_levels else 0,
            'path_type': 'comprehensive_prerequisites'
        }
    }

def conduct_slider_based_assessment(goal: str, graph: nx.Graph, category_knowledge: dict) -> dict:
    """Conduct assessment using slider-based self-evaluation of Wikipedia categories."""
    
    print(f"\n🎯 Conducting slider-based assessment for goal: '{goal}'")
    
    # Get semantic matches for target topics (same as before)
    topic_results = find_related_topics_by_semantic_similarity(goal, graph, max_direct=5, min_similarity=0.3)
    direct_matches = topic_results['direct_matches']
    graph_related = topic_results['graph_related']
    
    # Map categories to related nodes based on slider values (0-3 levels)
    knowledge_mapping = map_categories_to_related_nodes(graph, category_knowledge)
    
    # Generate learning path from known nodes to ONLY the highest semantic match (direct matches only)
    # Combine known and partially known nodes as starting points
    all_starting_nodes = knowledge_mapping['known_nodes'] + knowledge_mapping['partially_known_nodes']
    
    print(f"🚀 Using {len(knowledge_mapping['known_nodes'])} known + {len(knowledge_mapping['partially_known_nodes'])} partially known = {len(all_starting_nodes)} total starting points")
    
    learning_path = generate_learning_path_from_known(
        graph, 
        all_starting_nodes,  # Use both known and partially known as starting points
        direct_matches,  # Only use direct matches, not graph_related
        max_path_length=6
    )
    
    # Create enhanced intent with slider-based assessment info
    slider_intent = {
        'goal': goal,
        'analysis_method': 'slider_based_self_assessment_with_graph_layers',
        'categories_assessed': len(category_knowledge),
        'knowledge_levels_used': {
            'level_0': len([k for k in category_knowledge.values() if k == 0]),
            'level_1': len([k for k in category_knowledge.values() if k == 1]),
            'level_2': len([k for k in category_knowledge.values() if k == 2]),
            'level_3': len([k for k in category_knowledge.values() if k == 3])
        },
        'mapped_nodes': {
            'known': len(knowledge_mapping['known_nodes']),
            'partially_known': len(knowledge_mapping['partially_known_nodes']),
            'total_starting_points': len(all_starting_nodes),
            'total': knowledge_mapping['total_mapped']
        },
        'learning_path': {
            'steps': len(learning_path['path']),
            'message': learning_path['message'],
            'focused_on': 'highest_semantic_match_only'
        }
    }
    
    return {
        'intent': slider_intent,
        'direct_matches': direct_matches,
        'graph_related': graph_related,  # Still return for display, but not used in path generation
        'category_knowledge': category_knowledge,
        'knowledge_mapping': knowledge_mapping,
        'learning_path': learning_path,
        'target_topic': direct_matches[0]['id'] if direct_matches else None,
        'assessment_method': 'slider_based_discrete_levels',
        'all_starting_nodes': all_starting_nodes  # Include for graph highlighting
    }


