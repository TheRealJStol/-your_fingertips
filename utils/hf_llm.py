from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
import re
import os
import torch
import networkx as nx

# Use Qwen2.5-0.5B-Instruct for text generation
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# =============================================================================
# PROMPT TEMPLATES - Easily tweakable without touching UI code
# =============================================================================

INTENT_EXTRACTION_PROMPT = """Analyze this math learning goal and identify the target mathematical concept:

Goal: "{goal}"

Based on this goal, identify:
- Target Topic: The main mathematical concept they want to learn
- Current Level: Their likely starting level (beginner, intermediate, advanced)
- Learning Context: Why they want to learn this (academic, professional, personal)

Respond in this format:
TARGET_TOPIC: [exact mathematical concept name]
LEVEL: [beginner/intermediate/advanced]
CONTEXT: [brief explanation]

Analysis:"""

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

Provide a complete solution with:
- Step-by-step working
- Final answer
- Key concepts used

Solution:"""

ANSWER_ASSESSMENT_PROMPT = """You are a mathematics educator assessing a student's answer.

Problem: {problem}
Correct Solution: {correct_solution}
Student's Answer: {student_answer}
Topic: {topic}

Compare the student's answer to the correct solution and assess if the student demonstrates understanding of {topic}.

Consider:
- Is their final answer correct?
- Is their approach mathematically sound?
- Do they show understanding of key concepts?
- Are there minor computational errors vs fundamental misunderstandings?
- Did they show appropriate work/reasoning?

Respond with:
ASSESSMENT: [CORRECT/PARTIAL/INCORRECT]
EXPLANATION: [brief explanation of their understanding level and what they got right/wrong]

Assessment:"""

ROADMAP_GENERATION_PROMPT = """Create a detailed learning roadmap for a mathematics student.

Student Profile:
- Goal: {goal}
- Target Topic: {target_topic}
- Current Level: {level}

Assessment Results:
{assessment_results}

Known Topics (student has demonstrated understanding):
{known_topics}

Unknown/Weak Topics (student needs to learn):
{unknown_topics}

Create a step-by-step learning path from their current knowledge to their goal.
For each step, provide:
- The specific mathematical concept to learn
- Concrete learning activities (not just "study" or "practice")
- Specific skills or understanding to develop
- How it builds toward the target goal

Focus on:
- Building from what they know
- Addressing knowledge gaps systematically
- Logical progression of mathematical concepts
- Specific, actionable learning activities

Format exactly as:
Step 1: [specific topic] - [specific learning activity] - [specific skill to develop]
Step 2: [specific topic] - [specific learning activity] - [specific skill to develop]
[continue until reaching the goal]

Learning Path:"""

RESOURCE_GENERATION_PROMPT = """You are an expert mathematics educator recommending specific learning resources.

Student Context:
- Target Topic: {target_topic}
- Current Learning Step: {current_step}
- Specific Learning Need: {learning_need}
- Student Level: {level}

Recommend 3-4 SPECIFIC resources that would help with "{learning_need}" for learning "{target_topic}".

For each resource, provide:
- Resource Type (video, interactive tool, practice problems, textbook chapter, etc.)
- Specific Topic/Title
- Why it's helpful for this learning need
- Approximate study time

Format as:
1. [Resource Type]: [Specific Title/Topic] - [Why helpful] - [Time needed]
2. [Resource Type]: [Specific Title/Topic] - [Why helpful] - [Time needed]
3. [Resource Type]: [Specific Title/Topic] - [Why helpful] - [Time needed]

Resources:"""

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

_pipeline = None
_tokenizer = None

def get_pipeline():
    """Get text generation pipeline with proper Qwen model initialization."""
    global _pipeline, _tokenizer
    
    if _pipeline is None:
        try:
            print(f"Loading {model_id}...")
            
            # Load tokenizer
            _tokenizer = AutoTokenizer.from_pretrained(model_id)
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            
            # Load model with appropriate settings for Qwen
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Create pipeline
            _pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=_tokenizer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            print("✅ Qwen model loaded successfully!")
            return _pipeline
            
        except Exception as e:
            print(f"❌ Error loading Qwen model: {e}")
            print("This might be due to:")
            print("1. Model not available locally")
            print("2. Insufficient system resources") 
            print("3. Network connectivity issues")
            return None
    
    return _pipeline

def _create_qwen_prompt(system_prompt: str, user_input: str) -> str:
    """Create properly formatted prompt for Qwen model."""
    return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""

def find_target_topic_in_graph(goal: str, graph: nx.Graph) -> str:
    """Find the most relevant topic in the graph based on the learning goal."""
    goal_lower = goal.lower()
    
    # Get all topic nodes from the graph
    topic_nodes = [node for node, data in graph.nodes(data=True) 
                   if data.get('type') == 'Topic']
    
    if not topic_nodes:
        print("Warning: No Topic nodes found in graph")
        return None
    
    # Score nodes based on how well they match the goal
    best_match = None
    best_score = 0
    
    print(f"Searching for '{goal}' among {len(topic_nodes)} topics...")
    
    for node in topic_nodes:
        node_data = graph.nodes[node]
        node_id = node.lower()
        node_title = node_data.get('title', '').lower()
        node_summary = node_data.get('summary', '').lower()
        
        score = 0
        
        # Check for exact matches first (highest priority)
        if goal_lower == node_id or goal_lower == node_title:
            score += 100
            print(f"Exact match found: {node} (title: {node_data.get('title', '')})")
        
        # Check for direct substring matches
        if goal_lower in node_id or goal_lower in node_title:
            score += 50
            
        # Check if goal is contained in node_id or title
        if node_id in goal_lower or node_title in goal_lower:
            score += 30
        
        # Score based on individual word matches
        goal_words = [word for word in goal_lower.split() if len(word) > 2]  # Skip short words
        
        for word in goal_words:
            # High score for word matches in title
            if word in node_title:
                score += 10
            # Medium score for word matches in node ID
            if word in node_id:
                score += 8
            # Lower score for word matches in summary
            if word in node_summary:
                score += 3
        
        # Bonus for multiple word matches
        matching_words = sum(1 for word in goal_words if word in node_title or word in node_id)
        if matching_words > 1:
            score += matching_words * 5
            
        if score > best_score:
            best_score = score
            best_match = node
            print(f"New best match: {node} (title: {node_data.get('title', '')}) - Score: {score}")
    
    if best_match:
        print(f"Final match: {best_match} (title: {graph.nodes[best_match].get('title', '')}) - Score: {best_score}")
    else:
        print("No suitable match found, using first topic")
        best_match = topic_nodes[0]
    
    return best_match

def get_related_topics(target_topic: str, graph: nx.Graph, max_topics: int = 5) -> list:
    """Get topics related to the target topic in the graph."""
    if target_topic not in graph:
        print(f"Warning: Target topic '{target_topic}' not found in graph")
        return []
    
    related_topics = []
    
    # Get direct neighbors
    neighbors = list(graph.neighbors(target_topic))
    print(f"Found {len(neighbors)} direct neighbors for '{target_topic}'")
    
    # Get topics that are prerequisites or related
    for neighbor in neighbors:
        if graph.has_edge(target_topic, neighbor):
            edge_data = graph[target_topic][neighbor]
        elif graph.has_edge(neighbor, target_topic):
            edge_data = graph[neighbor][target_topic]
        else:
            edge_data = {}
            
        edge_type = edge_data.get('type', 'related_to')
        
        # Accept all relationship types since the graph uses 'related_to'
        if edge_type in ['prerequisite_of', 'related_to', 'supports', 'depends_on']:
            if neighbor not in related_topics:
                related_topics.append(neighbor)
                print(f"Added related topic: {neighbor} (relationship: {edge_type})")
    
    # If we need more topics, get second-degree neighbors
    if len(related_topics) < max_topics:
        print(f"Need more topics, searching second-degree neighbors...")
        for neighbor in neighbors[:10]:  # Limit search to avoid performance issues
            if neighbor in graph:
                second_neighbors = list(graph.neighbors(neighbor))
                for second_neighbor in second_neighbors:
                    if (second_neighbor != target_topic and 
                        second_neighbor not in related_topics and 
                        len(related_topics) < max_topics):
                        related_topics.append(second_neighbor)
                        print(f"Added second-degree topic: {second_neighbor}")
    
    print(f"Final related topics for '{target_topic}': {related_topics[:max_topics]}")
    return related_topics[:max_topics]

# =============================================================================
# MAIN LLM FUNCTIONS
# =============================================================================

def extract_intent_from_goal(goal: str) -> dict:
    """Extract target topic and learning intent from goal using LLM."""
    pipe = get_pipeline()
    
    if pipe is None:
        return {
            "goal": goal,
            "target_topic": "unknown",
            "current_level": "beginner",
            "learning_context": "personal interest"
        }
    
    system_prompt = "You are an expert mathematics educator who analyzes learning goals to identify target topics and learning context."
    user_prompt = INTENT_EXTRACTION_PROMPT.format(goal=goal)
    
    full_prompt = _create_qwen_prompt(system_prompt, user_prompt)
    
    try:
        result = pipe(full_prompt, max_new_tokens=200)
        generated_text = result[0]["generated_text"]
        
        # Extract only the assistant's response
        assistant_response = generated_text.split("<|im_start|>assistant")[-1].strip()
        
        # Parse structured response
        target_topic = "unknown"
        level = "beginner"
        context = "personal interest"
        
        lines = assistant_response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('TARGET_TOPIC:'):
                target_topic = line.split(':', 1)[1].strip()
            elif line.startswith('LEVEL:'):
                level = line.split(':', 1)[1].strip()
            elif line.startswith('CONTEXT:'):
                context = line.split(':', 1)[1].strip()
        
        return {
            "goal": goal,
            "target_topic": target_topic,
            "current_level": level,
            "learning_context": context
        }
        
    except Exception as e:
        print(f"LLM intent extraction error: {e}")
        return {
            "goal": goal,
            "target_topic": "unknown",
            "current_level": "beginner",
            "learning_context": "personal interest"
        }

def generate_math_problems(target_topic: str, related_topics: list, level: str, target_goal: str) -> list:
    """Generate actual math problems for related topics using LLM."""
    pipe = get_pipeline()
    
    if pipe is None:
        return [
            {
                "topic": topic,
                "problem": f"Solve a basic problem involving {topic}.",
                "type": "general"
            } for topic in related_topics[:3]
        ]
    
    problems = []
    system_prompt = "You are a mathematics educator who creates specific, concrete math problems to test student knowledge. You ONLY provide problem statements, never solutions."
    
    for topic in related_topics[:5]:  # Limit to 5 problems
        user_prompt = MATH_PROBLEM_PROMPT.format(
            topic=topic,
            level=level,
            target_goal=target_goal
        )
        
        full_prompt = _create_qwen_prompt(system_prompt, user_prompt)
        
        try:
            result = pipe(full_prompt, max_new_tokens=200)  # Reduced tokens to prevent solutions
            generated_text = result[0]["generated_text"]
            
            # Extract only the assistant's response
            assistant_response = generated_text.split("<|im_start|>assistant")[-1].strip()
            
            # Clean up the problem text
            problem_text = assistant_response.replace("Problem:", "").strip()
            
            # Filter out solutions/answers that might have leaked through
            problem_text = _filter_solutions_from_problem(problem_text)
            
            if len(problem_text) > 10 and not _contains_solution_indicators(problem_text):
                problems.append({
                    "topic": topic,
                    "problem": problem_text,
                    "type": "math_problem"
                })
            
        except Exception as e:
            print(f"Error generating problem for {topic}: {e}")
            # Add a simple fallback problem
            problems.append({
                "topic": topic,
                "problem": f"Create and solve a problem that demonstrates your understanding of {topic}. Show your work.",
                "type": "general"
            })
    
    return problems

def _filter_solutions_from_problem(problem_text: str) -> str:
    """Remove any solution content that might have leaked into the problem."""
    # Split by common solution indicators and take only the first part (the problem)
    solution_indicators = [
        "Solution:", "Answer:", "Step 1:", "First,", "To solve", 
        "The answer is", "Therefore", "So the result", "= "
    ]
    
    lines = problem_text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        # Stop at first solution indicator
        if any(indicator in line for indicator in solution_indicators):
            break
        if line and not line.startswith('='):  # Skip lines that start with equals
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines).strip()

def _contains_solution_indicators(text: str) -> bool:
    """Check if text contains solution indicators."""
    solution_words = [
        "the answer is", "solution:", "step 1:", "therefore", "so the result",
        "equals", "= ", "solve:", "working:"
    ]
    text_lower = text.lower()
    return any(word in text_lower for word in solution_words)

def assess_student_answer(problem: str, student_answer: str, topic: str) -> dict:
    """Use LLM to assess if student answered correctly and understands the topic."""
    pipe = get_pipeline()
    
    if pipe is None:
        # Simple keyword-based assessment as fallback
        answer_lower = student_answer.lower()
        if len(answer_lower) > 20 and any(word in answer_lower for word in ['solve', 'answer', 'result', 'equals']):
            return {"assessment": "PARTIAL", "explanation": "Response provided but cannot verify accuracy"}
        else:
            return {"assessment": "INCORRECT", "explanation": "Insufficient response"}
    
    # Step 1: Get the correct solution first
    correct_solution = _solve_problem_with_llm(problem, topic, pipe)
    
    # Step 2: Assess student's answer against correct solution
    system_prompt = "You are a mathematics educator who assesses student answers by comparing them to correct solutions."
    user_prompt = ANSWER_ASSESSMENT_PROMPT.format(
        problem=problem,
        correct_solution=correct_solution,
        student_answer=student_answer,
        topic=topic
    )
    
    full_prompt = _create_qwen_prompt(system_prompt, user_prompt)
    
    try:
        result = pipe(full_prompt, max_new_tokens=250)
        generated_text = result[0]["generated_text"]
        
        # Extract only the assistant's response
        assistant_response = generated_text.split("<|im_start|>assistant")[-1].strip()
        
        # Parse assessment
        assessment = "PARTIAL"
        explanation = "Assessment completed"
        
        lines = assistant_response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('ASSESSMENT:'):
                assessment = line.split(':', 1)[1].strip()
            elif line.startswith('EXPLANATION:'):
                explanation = line.split(':', 1)[1].strip()
        
        return {
            "assessment": assessment,
            "explanation": explanation,
            "correct_solution": correct_solution  # Include for debugging/transparency
        }
        
    except Exception as e:
        print(f"Error assessing answer: {e}")
        return {
            "assessment": "PARTIAL",
            "explanation": "Could not complete assessment",
            "correct_solution": correct_solution
        }

def _solve_problem_with_llm(problem: str, topic: str, pipe) -> str:
    """Use LLM to solve the problem and get the correct solution."""
    system_prompt = "You are a mathematics expert who solves problems step-by-step with complete accuracy."
    user_prompt = PROBLEM_SOLUTION_PROMPT.format(
        problem=problem,
        topic=topic
    )
    
    full_prompt = _create_qwen_prompt(system_prompt, user_prompt)
    
    try:
        result = pipe(full_prompt, max_new_tokens=300)
        generated_text = result[0]["generated_text"]
        
        # Extract only the assistant's response
        assistant_response = generated_text.split("<|im_start|>assistant")[-1].strip()
        
        # Clean up the solution
        solution = assistant_response.replace("Solution:", "").strip()
        
        return solution if solution else "Could not generate solution"
        
    except Exception as e:
        print(f"Error solving problem: {e}")
        return f"Error generating solution for {topic} problem"

def generate_learning_roadmap(intent: dict, assessment_results: dict, known_topics: list, unknown_topics: list) -> dict:
    """Generate a learning roadmap from known topics to the goal using LLM."""
    pipe = get_pipeline()
    
    if pipe is None:
        # Simple fallback roadmap
        steps = {}
        for i, topic in enumerate(unknown_topics[:6], 1):
            steps[f"Step {i}"] = [f"Study {topic}", f"Practice {topic} problems", f"Master {topic} concepts"]
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
            steps[f"Step {i}"] = [f"Study {topic}", f"Practice {topic} problems", f"Master {topic} concepts"]
        return steps

def generate_learning_resources(intent: dict, assessment_results: dict, current_step: str, learning_need: str) -> dict:
    """Generate specific learning resources for a given step using LLM."""
    pipe = get_pipeline()
    
    if pipe is None:
        return {
            "resources": [
                {"type": "Video", "title": "Introduction to Mathematics", "why_helpful": "Overview of key concepts", "time": "10 min"},
                {"type": "Interactive Tool", "title": "Math Practice Platform", "why_helpful": "Interactive exercises", "time": "15 min"}
            ]
        }
    
    system_prompt = "You are an expert mathematics educator who recommends specific learning resources."
    
    # Format assessment results
    assessment_summary = []
    for topic, result in assessment_results.items():
        assessment_summary.append(f"{topic}: {result['assessment']} - {result['explanation']}")
    
    user_prompt = RESOURCE_GENERATION_PROMPT.format(
        target_topic=intent.get('target_topic', ''),
        current_step=current_step,
        learning_need=learning_need,
        level=intent.get('current_level', 'beginner')
    )
    
    full_prompt = _create_qwen_prompt(system_prompt, user_prompt)
    
    try:
        result = pipe(full_prompt, max_new_tokens=400)
        generated_text = result[0]["generated_text"]
        
        # Extract only the assistant's response
        assistant_response = generated_text.split("<|im_start|>assistant")[-1].strip()
        
        # Parse resources
        resources = []
        lines = assistant_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('1.'):
                parts = line.split('1.', 1)
                if len(parts) > 1:
                    resource_line = parts[1].strip()
                    resource_type_match = re.match(r'\[Resource Type\]:\s*(.+?)\s*-\s*\[Why helpful\]:\s*(.+?)\s*-\s*\[Time needed\]:\s*(.+)', resource_line)
                    if resource_type_match:
                        resource_type = resource_type_match.group(1).strip()
                        title = resource_type_match.group(2).strip()
                        time = resource_type_match.group(3).strip()
                        resources.append({"type": resource_type, "title": title, "why_helpful": "Resource not specified", "time": time})
        
        # If no resources found, provide a basic recommendation
        if not resources:
            resources.append({"type": "Video", "title": "Introduction to Mathematics", "why_helpful": "Overview of key concepts", "time": "10 min"})
            resources.append({"type": "Interactive Tool", "title": "Math Practice Platform", "why_helpful": "Interactive exercises", "time": "15 min"})
        
        return {"resources": resources}
        
    except Exception as e:
        print(f"Error generating resources: {e}")
        return {
            "resources": [
                {"type": "Video", "title": "Introduction to Mathematics", "why_helpful": "Overview of key concepts", "time": "10 min"},
                {"type": "Interactive Tool", "title": "Math Practice Platform", "why_helpful": "Interactive exercises", "time": "15 min"}
            ]
        }

# =============================================================================
# GRAPH-BASED ASSESSMENT WORKFLOW
# =============================================================================

def conduct_graph_based_assessment(goal: str, graph: nx.Graph) -> dict:
    """Complete assessment workflow: find target, generate problems, assess answers."""
    
    # Step 1: Extract intent and find target topic in graph
    intent = extract_intent_from_goal(goal)
    target_topic = find_target_topic_in_graph(goal, graph)
    
    if target_topic:
        intent['target_topic'] = target_topic
    
    # Step 2: Get related topics from graph
    related_topics = get_related_topics(target_topic, graph) if target_topic else []
    
    # Step 3: Generate math problems for related topics
    problems = generate_math_problems(
        target_topic or "mathematics",
        related_topics,
        intent.get('current_level', 'beginner'),
        goal
    )
    
    return {
        'intent': intent,
        'target_topic': target_topic,
        'related_topics': related_topics,
        'problems': problems
    }

def process_assessment_answers(problems: list, answers: dict) -> dict:
    """Process student answers and determine known/unknown topics."""
    assessment_results = {}
    known_topics = []
    unknown_topics = []
    
    for problem_data in problems:
        topic = problem_data['topic']
        problem_text = problem_data['problem']
        
        if topic in answers and answers[topic].strip():
            # Assess the answer using LLM
            result = assess_student_answer(problem_text, answers[topic], topic)
            assessment_results[topic] = result
            
            # Categorize based on assessment
            if result['assessment'] == 'CORRECT':
                known_topics.append(topic)
            elif result['assessment'] == 'PARTIAL':
                # Partial understanding - could go either way
                if 'good' in result['explanation'].lower() or 'understand' in result['explanation'].lower():
                    known_topics.append(topic)
                else:
                    unknown_topics.append(topic)
            else:
                unknown_topics.append(topic)
        else:
            # No answer provided
            assessment_results[topic] = {
                'assessment': 'NO_ANSWER',
                'explanation': 'No response provided'
            }
            unknown_topics.append(topic)
    
    return {
        'assessment_results': assessment_results,
        'known_topics': known_topics,
        'unknown_topics': unknown_topics
    }

# =============================================================================
# LEGACY FUNCTIONS (for backward compatibility)
# =============================================================================

def llm_intake(goal: str):
    """Legacy function - use extract_intent_from_goal instead."""
    return extract_intent_from_goal(goal)

def llm_generate_questions(intent: dict):
    """Legacy function - now returns empty list since we use graph-based problems."""
    return []

def llm_generate_roadmap(intent, approach, context):
    """Legacy function - use generate_learning_roadmap instead."""
    return generate_learning_roadmap(intent, {}, [], [])

def llm_text(prompt, max_tokens=100):
    """General text generation function."""
    pipe = get_pipeline()
    if pipe is None:
        return "LLM not available - model loading failed"
    
    try:
        result = pipe(prompt, max_new_tokens=max_tokens)
        return result[0]["generated_text"]
    except Exception as e:
        return f"Error generating text: {e}"

# =============================================================================
# GRAPH-BASED PATHFINDING FOR LEARNING ROADMAPS
# =============================================================================

def generate_pathfinding_roadmap(graph: nx.Graph, known_topics: list, target_topic: str, intent: dict) -> dict:
    """Generate learning roadmap using graph pathfinding from known topics to target."""
    
    if not target_topic or target_topic not in graph.nodes():
        print(f"⚠️ Target topic '{target_topic}' not found in graph")
        return {}
    
    # If no known topics, find some basic starting points
    if not known_topics:
        # Find topics with low in-degree (likely foundational)
        in_degrees = dict(graph.in_degree())
        potential_starts = sorted(in_degrees.items(), key=lambda x: x[1])[:5]
        known_topics = [topic for topic, _ in potential_starts]
        print(f"🔍 No known topics provided, using foundational topics: {known_topics}")
    
    # Generate paths from each known topic to target
    path_results = []
    print(f"\n🎯 Target: {target_topic}")
    
    for start_topic in known_topics:
        if start_topic not in graph.nodes():
            print(f"⚠️ Start topic '{start_topic}' not found in graph")
            continue
            
        try:
            path = nx.shortest_path(graph, source=start_topic, target=target_topic)
            path_length = len(path) - 1
            path_results.append({
                "start": start_topic,
                "target": target_topic,
                "length": path_length,
                "path": " → ".join(path),
                "path_nodes": path
            })
            print(f"✓ {start_topic:35s} → {target_topic:25s} = {path_length} steps")
        except nx.NetworkXNoPath:
            path_results.append({
                "start": start_topic,
                "target": target_topic,
                "length": None,
                "path": "NO PATH",
                "path_nodes": []
            })
            print(f"⟂ No path  {start_topic:35s} → {target_topic}")
    
    # Find the shortest valid path
    valid_paths = [p for p in path_results if p["length"] is not None]
    if not valid_paths:
        print("❌ No valid paths found to target topic")
        return {}
    
    # Use the shortest path as our primary roadmap
    best_path = min(valid_paths, key=lambda x: x["length"])
    print(f"\n🏆 Best path: {best_path['path']} ({best_path['length']} steps)")
    
    # Convert path to roadmap steps
    roadmap = {}
    path_nodes = best_path["path_nodes"]
    
    # Skip the first node (starting point) since user already knows it
    learning_path = path_nodes[1:]  
    
    for i, topic in enumerate(learning_path, 1):
        # Get topic details from graph
        topic_data = graph.nodes.get(topic, {})
        topic_title = topic_data.get('title', topic)
        topic_summary = topic_data.get('summary', f'Learn about {topic}')
        difficulty = topic_data.get('difficulty', 'intermediate')
        
        # Create step description
        step_description = f"**{topic_title}**"
        if topic_summary:
            step_description += f": {topic_summary}"
        
        # Add context about prerequisites (previous step)
        if i == 1:
            prereq_info = f"Building on your knowledge of {graph.nodes.get(path_nodes[0], {}).get('title', path_nodes[0])}"
        else:
            prereq_info = f"After mastering {graph.nodes.get(path_nodes[i-1], {}).get('title', path_nodes[i-1])}"
        
        roadmap[f"Step {i}"] = [
            f"📚 **Topic**: {step_description}",
            f"🔗 **Prerequisites**: {prereq_info}",
            f"📊 **Difficulty**: {difficulty.capitalize()}",
            f"🎯 **Goal**: Master this concept to progress toward {target_topic}"
        ]
    
    # Add metadata about the path
    roadmap_metadata = {
        "total_steps": len(learning_path),
        "starting_from": best_path["start"],
        "target_topic": target_topic,
        "path_summary": best_path["path"],
        "alternative_paths": len(valid_paths) - 1
    }
    
    return {
        "roadmap": roadmap,
        "metadata": roadmap_metadata,
        "all_paths": path_results  # For debugging/alternative routes
    }

# =============================================================================
# ENHANCED ROADMAP GENERATION (combines LLM + pathfinding)
# =============================================================================

def generate_enhanced_roadmap(graph: nx.Graph, intent: dict, assessment_results: dict, known_topics: list, unknown_topics: list) -> dict:
    """Generate enhanced roadmap combining graph pathfinding with LLM insights."""
    
    target_topic = intent.get('target_topic')
    if not target_topic:
        print("⚠️ No target topic found in intent")
        return generate_learning_roadmap(intent, assessment_results, known_topics, unknown_topics)
    
    # First, get the graph-based optimal path
    pathfinding_result = generate_pathfinding_roadmap(graph, known_topics, target_topic, intent)
    
    if not pathfinding_result or "roadmap" not in pathfinding_result:
        print("⚠️ Pathfinding failed, falling back to LLM-only roadmap")
        return generate_learning_roadmap(intent, assessment_results, known_topics, unknown_topics)
    
    # Enhance the pathfinding roadmap with LLM-generated learning activities
    enhanced_roadmap = {}
    base_roadmap = pathfinding_result["roadmap"]
    
    pipe = get_pipeline()
    
    for step_key, step_content in base_roadmap.items():
        # Extract topic from step content
        topic_line = step_content[0]  # First line has the topic
        topic_match = re.search(r'\*\*(.+?)\*\*', topic_line)
        topic_name = topic_match.group(1) if topic_match else "unknown topic"
        
        if pipe:
            # Generate specific learning activities using LLM
            activity_prompt = f"""Create 3 specific learning activities for mastering: {topic_name}

Target Level: {intent.get('current_level', 'beginner')}
Learning Goal: {intent.get('goal', '')}

Provide exactly 3 activities in this format:
1. [Concrete learning activity]
2. [Practice exercise or problem type]
3. [Assessment or application method]

Activities:"""
            
            full_prompt = _create_qwen_prompt("You are a mathematics educator creating specific learning activities.", activity_prompt)
            
            try:
                result = pipe(full_prompt, max_new_tokens=200)
                generated_text = result[0]["generated_text"]
                assistant_response = generated_text.split("<|im_start|>assistant")[-1].strip()
                
                # Parse activities
                activities = []
                for line in assistant_response.split('\n'):
                    line = line.strip()
                    if re.match(r'^\d+\.', line):
                        activity = re.sub(r'^\d+\.\s*', '', line)
                        activities.append(f"🎯 {activity}")
                
                # Combine pathfinding info with LLM activities
                enhanced_step = step_content[:3]  # Keep first 3 lines from pathfinding
                if activities:
                    enhanced_step.extend(activities)
                else:
                    enhanced_step.append(f"🎯 Study and practice {topic_name} concepts")
                
                enhanced_roadmap[step_key] = enhanced_step
                
            except Exception as e:
                print(f"Error enhancing step {step_key}: {e}")
                enhanced_roadmap[step_key] = step_content
        else:
            # No LLM available, use pathfinding result as-is
            enhanced_roadmap[step_key] = step_content
    
    return {
        "roadmap": enhanced_roadmap,
        "metadata": pathfinding_result.get("metadata", {}),
        "pathfinding_info": pathfinding_result
    }

# =============================================================================


