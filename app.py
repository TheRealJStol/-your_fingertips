import streamlit as st
from utils.graph_loader import load_graph, load_excerpts, load_metas
from utils.hf_llm import (
    conduct_graph_based_assessment, 
    conduct_semantic_assessment,
    process_assessment_answers, 
    generate_learning_roadmap,
    generate_enhanced_roadmap
)
from utils.retrieval import build_index, retrieve
from ui_components.graph_viz import draw_graph
from utils.scraper import scrape_resources
import os

st.set_page_config(layout="wide")
st.title("📚 Math Learning Navigator")
st.caption("🤖 AI-Powered Graph-Based Math Assessment & Learning Roadmaps")

# Initialize session state
if 'assessment_data' not in st.session_state:
    st.session_state.assessment_data = None
if 'problems_generated' not in st.session_state:
    st.session_state.problems_generated = False

# Load graph + build index once
@st.cache_data
def load_data():
    try:
        G = load_graph("/Users/jiawenyang/Documents/GitHub/-your_fingertips/math_firstyear-2.yaml")  # Use the new weighted graph
        
        # Load excerpts and metas if they exist
        excerpts_dir = "data/excerpts"
        metas_path = "data/metas.json"
        
        texts = []
        metas = []
        
        if os.path.exists(excerpts_dir) and os.listdir(excerpts_dir):
            texts = load_excerpts(excerpts_dir)
        
        if os.path.exists(metas_path):
            try:
                metas = load_metas(metas_path)
            except:
                metas = []
        
        embed_model, idx = build_index(texts)
        return G, texts, metas, embed_model, idx
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, [], [], None, None

G, texts, metas, embed_model, idx = load_data()

if G is None:
    st.error("Failed to load math knowledge graph. Please check your data files.")
    st.stop()

# --- Learning Goal Input ---
st.header("🎯 What mathematical concept do you want to learn?")
goal = st.text_input(
    "Describe your learning goal", 
    placeholder="e.g., I want to understand calculus derivatives, learn about prime numbers, master linear algebra"
)

# Reset assessment if goal changes
if 'previous_goal' not in st.session_state:
    st.session_state.previous_goal = ""

if goal != st.session_state.previous_goal:
    st.session_state.assessment_data = None
    st.session_state.problems_generated = False
    st.session_state.previous_goal = goal

if goal and not st.session_state.problems_generated:
    if st.button("🤖 Find Topics (Semantic + Graph Traversal)"):
        with st.spinner("🤖 Analyzing with semantic similarity + graph traversal..."):
            # Use semantic assessment with sentence transformers
            st.session_state.assessment_data = conduct_semantic_assessment(goal, G)
            st.session_state.problems_generated = True
            st.rerun()

if st.session_state.problems_generated and st.session_state.assessment_data:
    assessment_data = st.session_state.assessment_data
    
    st.success("✅ Related topics found using semantic similarity!")
    
    # Show what the semantic analysis found
    st.subheader("🤖 Semantic Analysis + Weighted Graph Traversal Results:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Analysis Method:** Semantic + Weighted Graph")
        st.write(f"**Direct Matches:** {assessment_data['intent'].get('direct_matches_found', 0)}/5")
    with col2:
        st.write(f"**Graph Related:** {assessment_data['intent'].get('graph_related_found', 0)}/10")
        st.write(f"**Total Results:** {assessment_data['intent'].get('direct_matches_found', 0) + assessment_data['intent'].get('graph_related_found', 0)}/15")
    with col3:
        st.write(f"**Top Similarity:** {assessment_data['intent'].get('top_similarity_score', 0):.3f}")
        st.write(f"**Similarity Threshold:** ≥{assessment_data['intent'].get('similarity_threshold', 0.3)}")
    
    # Display direct matches (semantic similarity > 0.3)
    direct_matches = assessment_data.get('direct_matches', [])
    if direct_matches:
        st.subheader("🎯 Direct Matches (Semantic Similarity > 0.3)")
        st.caption("Topics with high semantic similarity to your input")
        
        # Show similarity range for direct matches
        debug_info = assessment_data.get('debug_info', {})
        similarity_range = debug_info.get('direct_similarity_range', {})
        if similarity_range:
            st.info(f"📈 **Direct Match Range:** {similarity_range.get('highest', 0):.3f} (highest) → {similarity_range.get('lowest', 0):.3f} (lowest)")
        
        for i, topic in enumerate(direct_matches, 1):
            similarity_score = topic.get('similarity_score', 0)
            # Create a visual indicator for similarity strength
            if similarity_score >= 0.7:
                similarity_icon = "🟢"
                similarity_label = "Very High"
            elif similarity_score >= 0.5:
                similarity_icon = "🟡"
                similarity_label = "High"
            elif similarity_score >= 0.3:
                similarity_icon = "🟠"
                similarity_label = "Medium"
            else:
                similarity_icon = "🔴"
                similarity_label = "Low"
            
            with st.expander(f"#{i} **{topic['title']}** {similarity_icon} {similarity_score:.3f} ({similarity_label})", expanded=i<=2):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**ID:** `{topic['id']}`")
                    st.write(f"**Summary:** {topic['summary']}")
                    st.write(f"**Difficulty:** {topic['difficulty'].title()}")
                with col2:
                    st.write(f"**Semantic Similarity:** {similarity_score:.3f}")
                    st.write(f"**Topic Weight:** {topic.get('weight', 0.0):.3f}")
                
                # Show the full text that was used for matching
                if 'text' in topic:
                    with st.expander("🔍 Text used for similarity matching", expanded=False):
                        st.code(topic['text'])
    
    # Display graph-related topics (weighted and prioritized)
    graph_related = assessment_data.get('graph_related', [])
    if graph_related:
        st.subheader("🗺️ Graph-Related Topics (Weighted & Quality-Prioritized)")
        st.caption("Topics discovered through graph traversal, prioritized by weight + shortest path + similarity path quality")
        
        # Show graph analysis summary with weight information
        graph_analysis = debug_info.get('graph_analysis', {})
        if graph_analysis:
            weight_range = graph_analysis.get('weight_range', {})
            st.info(f"🔗 **Graph Analysis:** Selected top 10 from candidates, prioritized by weight + path length + similarity quality. Weight range: {weight_range.get('lowest', 0):.3f} → {weight_range.get('highest', 0):.3f}")
        
        for i, topic in enumerate(graph_related, 1):
            connection_count = topic.get('connection_count', 0)
            topic_weight = topic.get('weight', 0.0)
            min_distance = topic.get('min_distance', 2)
            max_path_quality = topic.get('max_path_quality', 0.0)
            priority_score = topic.get('priority_score', 0.0)
            
            # Create visual indicator for priority (weight + distance + path quality)
            if priority_score >= 3.0:
                priority_icon = "🏆"
                priority_label = "High Priority"
            elif priority_score >= 2.5:
                priority_icon = "🥈"
                priority_label = "Medium Priority"
            else:
                priority_icon = "🥉"
                priority_label = "Lower Priority"
            
            # Distance indicator
            distance_icon = "🔗" if min_distance == 1 else "🔗🔗"
            
            # Path quality indicator
            if max_path_quality >= 0.7:
                quality_icon = "⭐⭐⭐"
            elif max_path_quality >= 0.5:
                quality_icon = "⭐⭐"
            else:
                quality_icon = "⭐"
            
            with st.expander(f"#{i} **{topic['title']}** {priority_icon} {distance_icon} {quality_icon} (Priority: {priority_score:.2f})", expanded=i<=3):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**ID:** `{topic['id']}`")
                    st.write(f"**Summary:** {topic['summary']}")
                    st.write(f"**Difficulty:** {topic['difficulty'].title()}")
                with col2:
                    st.write(f"**Weight:** {topic_weight:.3f}")
                    st.write(f"**Min Distance:** {min_distance} steps")
                    st.write(f"**Path Quality:** {max_path_quality:.3f}")
                    st.write(f"**Priority Score:** {priority_score:.3f}")
                
                st.write(f"**Connected to {connection_count} direct matches via highest-quality {min_distance}-step paths**")
                
                # Show enhanced connection paths with quality information
                if topic.get('connection_paths'):
                    st.write("**Connection Paths (ranked by quality):**")
                    for path in topic['connection_paths']:
                        if path.startswith("🏆 BEST:"):
                            st.success(f"  {path}")
                        else:
                            st.write(f"  • {path}")
    
    # Show warning if no results
    if not direct_matches and not graph_related:
        st.warning("No matching topics found above the 0.3 similarity threshold. Try rephrasing your goal with different mathematical terms.")
    
    # Skip problem generation for debugging
    if assessment_data.get('problems'):
        st.subheader("🧮 Mathematical Assessment Problems")
        st.caption("Problem generation temporarily disabled for debugging")
        st.info("🔧 **Debug Mode:** Skipping problem generation to focus on semantic topic matching")
    else:
        st.info("🔧 **Debug Mode:** Problem generation skipped - focusing on semantic similarity analysis")
        st.write("---")
    
    if st.button("📊 Submit Answers & Generate Learning Roadmap"):
        # Check if answers are provided
        unanswered = [topic for topic, answer in answers.items() if not answer.strip()]
        if unanswered:
            st.warning(f"Please provide answers for: {', '.join(unanswered)}")
        else:
            with st.spinner("🤖 AI is assessing your answers and determining your knowledge level..."):
                # Process answers and determine known/unknown topics
                assessment_results = process_assessment_answers(problems, answers)
                
                st.success("✅ Assessment completed!")
                
                # Show assessment results
                st.header("📊 Your Assessment Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("✅ Topics You Know")
                    if assessment_results['known_topics']:
                        for topic in assessment_results['known_topics']:
                            result = assessment_results['assessment_results'][topic]
                            st.write(f"• **{topic}**: {result['assessment']}")
                            st.caption(f"   {result['explanation']}")
                    else:
                        st.write("No topics identified as fully understood")
                
                with col2:
                    st.subheader("📚 Topics to Learn")
                    if assessment_results['unknown_topics']:
                        for topic in assessment_results['unknown_topics']:
                            result = assessment_results['assessment_results'][topic]
                            st.write(f"• **{topic}**: {result['assessment']}")
                            st.caption(f"   {result['explanation']}")
                    else:
                        st.write("All assessed topics understood!")
                
                # Generate personalized learning roadmap using graph pathfinding + LLM
                with st.spinner("🗺️ AI is creating your personalized learning roadmap using graph pathfinding..."):
                    roadmap_result = generate_enhanced_roadmap(
                        G,  # Pass the graph for pathfinding
                        assessment_data['intent'],
                        assessment_results['assessment_results'],
                        assessment_results['known_topics'],
                        assessment_results['unknown_topics']
                    )
                    
                    # Extract roadmap and metadata
                    roadmap = roadmap_result.get('roadmap', roadmap_result) if isinstance(roadmap_result, dict) else roadmap_result
                    roadmap_metadata = roadmap_result.get('metadata', {}) if isinstance(roadmap_result, dict) else {}
                
                # Display the learning roadmap
                st.header("🗺️ Your Personalized Learning Roadmap")
                st.subheader(f"From your current knowledge to: {assessment_data['target_topic']}")
                
                # Show pathfinding metadata if available
                if roadmap_metadata:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Total Steps", roadmap_metadata.get('total_steps', 'N/A'))
                    with col2:
                        st.metric("🎯 Starting From", roadmap_metadata.get('starting_from', 'N/A'))
                    with col3:
                        st.metric("🔄 Alternative Paths", roadmap_metadata.get('alternative_paths', 'N/A'))
                    
                    if roadmap_metadata.get('path_summary'):
                        st.info(f"🛤️ **Optimal Learning Path**: {roadmap_metadata['path_summary']}")
                
                st.caption("🤖 This roadmap uses graph pathfinding to find the shortest learning path combined with AI-generated activities")
                
                if roadmap:
                    for step, activities in roadmap.items():
                        with st.expander(f"**{step}**", expanded=True):
                            for activity in activities:
                                st.write(f"• {activity}")
                            
                            # Add progress tracker
                            if st.checkbox(f"Mark {step} as completed", key=f"step_{step}"):
                                st.success(f"✅ {step} completed!")
                else:
                    st.warning("Could not generate roadmap. Please try with a different learning goal.")
                
                # Show pathfinding details
                if isinstance(roadmap_result, dict) and 'pathfinding_info' in roadmap_result:
                    pathfinding_info = roadmap_result['pathfinding_info']
                    if 'all_paths' in pathfinding_info:
                        with st.expander("🗺️ All Possible Learning Paths", expanded=False):
                            st.caption("Alternative routes from your known topics to the target")
                            for path_info in pathfinding_info['all_paths']:
                                if path_info['length'] is not None:
                                    st.write(f"**{path_info['start']}** → **{path_info['target']}** ({path_info['length']} steps)")
                                    st.code(path_info['path'])
                                else:
                                    st.write(f"**{path_info['start']}** → **{path_info['target']}**: No path available")
                
                # Show detailed assessment breakdown
                with st.expander("🔍 Detailed Assessment Breakdown", expanded=False):
                    st.caption("See how the AI assessed each of your responses")
                    
                    # Add option to show correct solutions
                    show_solutions = st.checkbox("🔓 Show correct solutions (for learning)", key="show_solutions")
                    
                    for i, problem_data in enumerate(problems):
                        topic = problem_data['topic']
                        problem_text = problem_data['problem']
                        user_answer = answers[topic]
                        result = assessment_results['assessment_results'][topic]
                        
                        st.write(f"**Problem {i+1}: {topic}**")
                        st.write(f"*Problem:* {problem_text}")
                        st.write(f"*Your Answer:* {user_answer}")
                        st.write(f"*AI Assessment:* {result['assessment']}")
                        st.write(f"*Explanation:* {result['explanation']}")
                        
                        # Show correct solution if requested and available
                        if show_solutions and 'correct_solution' in result:
                            with st.expander(f"📖 Correct Solution for {topic}", expanded=False):
                                st.write(result['correct_solution'])
                                st.caption("Use this to understand the correct approach and learn from any mistakes.")
                        
                        st.write("---")
                
                # Get learning resources
                st.header("📚 Personalized Learning Resources")
                st.caption("AI-generated resources tailored to your specific assessment results")
                
                # Generate personalized resources based on assessment
                from utils.scraper import generate_personalized_resources
                
                personalized_resources = generate_personalized_resources(
                    target_topic=assessment_data['target_topic'],
                    assessment_results=assessment_results['assessment_results'],
                    known_topics=assessment_results['known_topics'],
                    unknown_topics=assessment_results['unknown_topics'],
                    level=assessment_data['intent'].get('current_level', 'beginner')
                )
                
                # Display personalized resources
                for i, resource in enumerate(personalized_resources, 1):
                    with st.expander(f"📖 Resource {i}: {resource['title']}", expanded=False):
                        st.write(f"**Type:** {resource['type']}")
                        st.write(f"**Description:** {resource['description']}")
                        st.write(f"**Time Needed:** {resource['time']}")
                        if resource.get('personalized', False):
                            st.success("🎯 Personalized for your specific needs")
                        
                        # Add a "completed" checkbox for tracking
                        if st.checkbox(f"Mark as completed", key=f"resource_{i}"):
                            st.success("✅ Resource completed!")
                
                # Add step-specific resources for each roadmap step
                st.subheader("🎯 Step-by-Step Learning Resources")
                st.caption("Specific resources for each step in your learning roadmap")
                
                from utils.scraper import generate_step_specific_resources
                
                for step_name, step_details in roadmap.items():
                    if len(step_details) >= 3:  # Has topic, activity, skill format
                        # Extract topic, activity, skill from formatted strings
                        topic_line = step_details[0].replace("📚 **Topic**: ", "")
                        activity_line = step_details[1].replace("🎯 **Activity**: ", "")
                        skill_line = step_details[2].replace("💡 **Skill**: ", "")
                        
                        with st.expander(f"📚 {step_name} Resources", expanded=False):
                            st.write(f"**Focus:** {topic_line}")
                            st.write(f"**Activity:** {activity_line}")
                            st.write(f"**Skill Goal:** {skill_line}")
                            
                            # Generate step-specific resources
                            step_resources = generate_step_specific_resources(
                                step_topic=topic_line,
                                step_activity=activity_line,
                                step_skill=skill_line,
                                level=assessment_data['intent'].get('current_level', 'beginner')
                            )
                            
                            st.write("**Recommended Resources:**")
                            for j, step_resource in enumerate(step_resources, 1):
                                st.write(f"{j}. **{step_resource['type']}**: {step_resource['title']}")
                                if step_resource.get('supports_activity'):
                                    st.write(f"   - *Supports Activity*: {step_resource['supports_activity']}")
                                if step_resource.get('develops_skill'):
                                    st.write(f"   - *Develops Skill*: {step_resource['develops_skill']}")
                                st.write(f"   - *Time*: {step_resource['time']}")
                                st.write("")

    # Add a reset button
    if st.button("🔄 Start New Assessment"):
        st.session_state.assessment_data = None
        st.session_state.problems_generated = False
        st.session_state.previous_goal = ""
        st.rerun()

# --- Math Knowledge Graph ---
st.header("🗺️ Comprehensive Math Knowledge Graph")
st.write("Explore the relationships between mathematical concepts:")
st.caption(f"📊 **Graph Statistics:** {G.number_of_nodes()} concepts, {G.number_of_edges()} relationships")

# Show legend if highlighting is active
if st.session_state.assessment_data:
    with st.expander("🎨 Graph Legend (Node Highlighting)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🟡 **Target Topic** - Your main learning goal")
            st.markdown("🟢 **Direct Matches** - High semantic similarity (>0.3)")
        with col2:
            st.markdown("🟠 **Graph Related** - Connected via graph traversal")
            st.markdown("⚪ **Other Topics** - Standard graph nodes")
        with col3:
            st.markdown("**Node Size** - Larger = more important")
            st.markdown("**White Edges** - Connect to matched topics")

# Highlight target topic if found
if st.session_state.assessment_data and st.session_state.assessment_data['target_topic']:
    st.info(f"🎯 **Your Target Topic:** {st.session_state.assessment_data['target_topic']}")
    if st.session_state.assessment_data['related_topics']:
        topic_titles = [topic['title'] if isinstance(topic, dict) else str(topic) for topic in st.session_state.assessment_data['related_topics'][:3]]
        st.info(f"🔗 **Related Topics:** {', '.join(topic_titles)}...")

try:
    # Prepare highlighting information if assessment data is available
    highlight_info = None
    if st.session_state.assessment_data:
        direct_match_ids = [topic['id'] for topic in st.session_state.assessment_data.get('direct_matches', [])]
        graph_related_ids = [topic['id'] for topic in st.session_state.assessment_data.get('graph_related', [])]
        target_topic_id = st.session_state.assessment_data.get('target_topic')
        
        highlight_info = {
            'direct_matches': direct_match_ids,
            'graph_related': graph_related_ids,
            'target_topic': target_topic_id
        }
    
    graph_html = draw_graph(G, highlight_info)
    st.components.v1.html(graph_html, height=600)
except Exception as e:
    st.error(f"Error displaying graph: {e}")
    st.write("Graph visualization temporarily unavailable.")

# --- Sidebar info ---
st.sidebar.header("ℹ️ About Math Navigator")
st.sidebar.write("This tool uses **AI-powered graph-based assessment** to create personalized math learning roadmaps.")
st.sidebar.markdown("**🤖 AI Features:**")
st.sidebar.write("• Graph-based topic identification")
st.sidebar.write("• Real-time math problem generation")
st.sidebar.write("• Intelligent answer assessment")
st.sidebar.write("• Personalized learning roadmaps")
st.sidebar.markdown("**📊 Knowledge Graph:**")
st.sidebar.write(f"• {G.number_of_nodes()} mathematical concepts")
st.sidebar.write(f"• {G.number_of_edges()} concept relationships")
st.sidebar.write("• First-year mathematics topics")
st.sidebar.write("• Prerequisite relationships")
st.sidebar.markdown("**How it works:**")
st.sidebar.write("1. 🎯 AI finds your target topic in the graph")
st.sidebar.write("2. 🔗 Identifies related prerequisite topics") 
st.sidebar.write("3. 🧮 Generates actual math problems")
st.sidebar.write("4. 📊 AI assesses your solutions")
st.sidebar.write("5. 🗺️ Creates roadmap from known to unknown")

# Add model info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**🔧 Technical Details:**")
st.sidebar.write("• Model: Qwen2.5-0.5B-Instruct")
st.sidebar.write("• Graph-based problem generation")
st.sidebar.write("• Real-time answer assessment")
st.sidebar.write("• Knowledge-gap identification")

# Add study tips
st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Study Tips:**")
st.sidebar.write("🧮 Show your work clearly")
st.sidebar.write("📝 Explain your reasoning")
st.sidebar.write("❓ Don't be afraid to say 'I don't know'")
st.sidebar.write("🔄 Practice problems regularly")
st.sidebar.write("📈 Build from what you know")