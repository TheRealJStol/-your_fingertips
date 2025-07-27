import requests
from bs4 import BeautifulSoup
import time
import random
import re # Added for parsing LLM output
from .hf_llm import get_pipeline, _create_qwen_prompt

def generate_personalized_resources(target_topic: str, assessment_results: dict, known_topics: list, unknown_topics: list, level: str = "beginner") -> list:
    """Generate personalized learning resources using LLM based on assessment results."""
    pipe = get_pipeline()
    
    if pipe is None:
        return _fallback_resources(target_topic)
    
    # Create context from assessment
    learning_gaps = []
    strengths = []
    
    for topic, result in assessment_results.items():
        if result['assessment'] in ['INCORRECT', 'NO_ANSWER']:
            learning_gaps.append(f"{topic} (needs work: {result['explanation']})")
        elif result['assessment'] == 'CORRECT':
            strengths.append(f"{topic} (strong: {result['explanation']})")
        else:  # PARTIAL
            learning_gaps.append(f"{topic} (partial understanding: {result['explanation']})")
    
    system_prompt = """You are an expert mathematics educator who creates personalized learning resource recommendations.
    
Based on a student's assessment results, recommend specific learning resources that address their exact needs.
Focus on practical, actionable resources that build from their current knowledge."""
    
    user_prompt = f"""Create personalized learning resources for a student wanting to learn: {target_topic}

Student Assessment Results:
- Target Goal: {target_topic}
- Current Level: {level}
- Topics They Know Well: {', '.join(strengths) if strengths else 'None identified'}
- Topics Needing Work: {', '.join(learning_gaps) if learning_gaps else 'None identified'}

Generate 5-6 specific learning resources that:
1. Address their knowledge gaps directly
2. Build from what they already know
3. Are appropriate for their level
4. Progress logically toward their goal

For each resource, provide:
- Type (Video Tutorial, Interactive Exercise, Practice Problems, Concept Explanation, etc.)
- Specific Title/Description
- Why it helps with their specific needs
- Estimated time commitment

Format as:
1. [Type]: [Specific Title] - [Why it helps their specific situation] - [Time: X minutes]
2. [Type]: [Specific Title] - [Why it helps their specific situation] - [Time: X minutes]
...

Resources:"""
    
    full_prompt = _create_qwen_prompt(system_prompt, user_prompt)
    
    try:
        result = pipe(full_prompt, max_new_tokens=500)
        generated_text = result[0]["generated_text"]
        
        # Extract only the assistant's response
        assistant_response = generated_text.split("<|im_start|>assistant")[-1].strip()
        
        # Parse resources
        resources = _parse_generated_resources(assistant_response)
        
        if len(resources) >= 3:
            return resources
        else:
            return _fallback_resources(target_topic)
            
    except Exception as e:
        print(f"Error generating personalized resources: {e}")
        return _fallback_resources(target_topic)

def _parse_generated_resources(text: str) -> list:
    """Parse LLM-generated resources into structured format."""
    resources = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for numbered resources
        resource_match = re.match(r'^\d+\.\s*\[(.+?)\]:\s*(.+?)\s*-\s*(.+?)\s*-\s*Time:\s*(.+)', line)
        if resource_match:
            resource_type = resource_match.group(1).strip()
            title = resource_match.group(2).strip()
            why_helpful = resource_match.group(3).strip()
            time_needed = resource_match.group(4).strip()
            
            resources.append({
                "type": resource_type,
                "title": title,
                "description": why_helpful,
                "time": time_needed,
                "personalized": True
            })
        else:
            # Try simpler format
            simple_match = re.match(r'^\d+\.\s*(.+?)\s*-\s*(.+)', line)
            if simple_match:
                title = simple_match.group(1).strip()
                description = simple_match.group(2).strip()
                
                resources.append({
                    "type": "Learning Resource",
                    "title": title,
                    "description": description,
                    "time": "15-30 minutes",
                    "personalized": True
                })
    
    return resources

def generate_step_specific_resources(step_topic: str, step_activity: str, step_skill: str, level: str = "beginner") -> list:
    """Generate resources for a specific learning step."""
    pipe = get_pipeline()
    
    if pipe is None:
        return _fallback_step_resources(step_topic)
    
    system_prompt = """You are a mathematics educator who finds specific learning resources for individual learning steps.
    
Create targeted resources that help students complete a specific learning activity and develop a specific skill."""
    
    user_prompt = f"""Find specific learning resources for this learning step:

Learning Step Details:
- Topic: {step_topic}
- Activity: {step_activity}
- Skill to Develop: {step_skill}
- Student Level: {level}

Recommend 3-4 specific resources that directly support this activity and skill development.

For each resource:
- Type (Video, Interactive Tool, Practice Set, Reading, etc.)
- Specific Title/Content
- How it supports the activity
- How it develops the skill
- Time needed

Format as:
1. [Type]: [Title] - Supports: [how it helps activity] - Develops: [how it builds skill] - [Time]
2. [Type]: [Title] - Supports: [how it helps activity] - Develops: [how it builds skill] - [Time]
...

Resources:"""
    
    full_prompt = _create_qwen_prompt(system_prompt, user_prompt)
    
    try:
        result = pipe(full_prompt, max_new_tokens=400)
        generated_text = result[0]["generated_text"]
        
        # Extract only the assistant's response
        assistant_response = generated_text.split("<|im_start|>assistant")[-1].strip()
        
        # Parse step-specific resources
        resources = _parse_step_resources(assistant_response)
        
        if len(resources) >= 2:
            return resources
        else:
            return _fallback_step_resources(step_topic)
            
    except Exception as e:
        print(f"Error generating step resources: {e}")
        return _fallback_step_resources(step_topic)

def _parse_step_resources(text: str) -> list:
    """Parse step-specific resources."""
    resources = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for the detailed format
        resource_match = re.match(r'^\d+\.\s*\[(.+?)\]:\s*(.+?)\s*-\s*Supports:\s*(.+?)\s*-\s*Develops:\s*(.+?)\s*-\s*(.+)', line)
        if resource_match:
            resource_type = resource_match.group(1).strip()
            title = resource_match.group(2).strip()
            supports = resource_match.group(3).strip()
            develops = resource_match.group(4).strip()
            time_needed = resource_match.group(5).strip()
            
            resources.append({
                "type": resource_type,
                "title": title,
                "supports_activity": supports,
                "develops_skill": develops,
                "time": time_needed,
                "step_specific": True
            })
    
    return resources

def _fallback_resources(topic: str) -> list:
    """Minimal fallback resources when LLM is unavailable."""
    topic_lower = topic.lower()
    
    # Basic resource types
    resources = [
        {
            "type": "Concept Introduction",
            "title": f"Understanding {topic} Fundamentals",
            "description": f"Core concepts and principles of {topic}",
            "time": "20-30 minutes",
            "personalized": False
        },
        {
            "type": "Practice Problems",
            "title": f"{topic} Problem Set",
            "description": f"Guided practice problems for {topic}",
            "time": "30-45 minutes",
            "personalized": False
        },
        {
            "type": "Visual Learning",
            "title": f"{topic} Visual Guide",
            "description": f"Diagrams and visual explanations of {topic}",
            "time": "15-25 minutes",
            "personalized": False
        }
    ]
    
    return resources

def _fallback_step_resources(topic: str) -> list:
    """Fallback resources for specific learning steps."""
    return [
        {
            "type": "Tutorial",
            "title": f"{topic} Step-by-Step Guide",
            "supports_activity": "Provides structured learning approach",
            "develops_skill": "Builds foundational understanding",
            "time": "25-35 minutes",
            "step_specific": False
        },
        {
            "type": "Practice",
            "title": f"{topic} Exercises",
            "supports_activity": "Reinforces concepts through practice",
            "develops_skill": "Develops problem-solving abilities",
            "time": "20-30 minutes",
            "step_specific": False
        }
    ]

# Legacy functions for backward compatibility
def scrape_learning_links(approach):
    """Legacy function - now generates personalized resources."""
    return generate_personalized_resources(approach, {}, [], [], "beginner")

def scrape_resources(topic):
    """Legacy function - now generates personalized resources."""
    return generate_personalized_resources(topic, {}, [], [], "beginner")

def scrape_url(url, max_retries=3):
    """Keep original URL scraping functionality for actual web scraping needs."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title and description
            title = soup.find('title')
            title = title.text.strip() if title else 'No title'
            
            description = soup.find('meta', {'name': 'description'})
            description = description.get('content', 'No description') if description else 'No description'
            
            return {
                'url': url,
                'title': title,
                'description': description,
                'status': 'success'
            }
            
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    'url': url,
                    'title': 'Failed to load',
                    'description': f'Error: {str(e)}',
                    'status': 'error'
                }
            time.sleep(random.uniform(1, 3))  # Random delay between retries
    
    return None
