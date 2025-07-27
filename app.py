import streamlit as st
from utils.graph_loader import load_graph, load_excerpts, load_metas
from utils.hf_llm import (
    conduct_semantic_assessment, 
    generate_math_problems, 
    assess_student_answer, 
    generate_learning_roadmap,
    extract_wikipedia_categories,
    conduct_slider_based_assessment
)
from utils.retrieval import build_index, retrieve
from ui_components.graph_viz import draw_graph
from utils.scraper import scrape_resources
import os

st.set_page_config(layout="wide")
st.title("📚 Math Learning Navigator")
st.caption("🤖 AI-Powered Graph-Based Math Assessment & Learning Roadmaps")

# Initialize session state variables
if 'topics_found' not in st.session_state:
    st.session_state.topics_found = False
if 'categories_loaded' not in st.session_state:
    st.session_state.categories_loaded = False
if 'knowledge_assessed' not in st.session_state:
    st.session_state.knowledge_assessed = False
if 'learning_path_generated' not in st.session_state:
    st.session_state.learning_path_generated = False
if 'assessment_data' not in st.session_state:
    st.session_state.assessment_data = None
if 'wikipedia_categories' not in st.session_state:
    st.session_state.wikipedia_categories = []
if 'category_knowledge' not in st.session_state:
    st.session_state.category_knowledge = {}
if 'slider_assessment_data' not in st.session_state:
    st.session_state.slider_assessment_data = None
# Add persistent knowledge profile
if 'persistent_knowledge_profile' not in st.session_state:
    st.session_state.persistent_knowledge_profile = {}
if 'knowledge_profile_loaded' not in st.session_state:
    st.session_state.knowledge_profile_loaded = False

# Load graph + build index once
@st.cache_data
def load_data():
    try:
        G = load_graph("/Users/jiawenyang/Documents/GitHub/-your_fingertips/graph_g.yaml")  # Use the new graph with descriptions
        
        # Load excerpts and metas if they exist
        excerpts, metas = [], []
        try:
            excerpts = load_excerpts("data/excerpts")
            metas = load_metas("data/metas.json")
        except:
            pass  # Files don't exist, use empty lists
        
        return G, excerpts, metas
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error("Failed to load knowledge graph. Please check your data files.")
        return None, [], []

G, texts, metas = load_data()

if G is None:
    st.error("Failed to load math knowledge graph. Please check your data files.")
    st.stop()

# Build index for retrieval (if needed)
embed_model, idx = None, None
try:
    from utils.retrieval import build_index
    embed_model, idx = build_index(texts)
except:
    pass  # Retrieval not needed for core functionality

# Main content
st.header("🧮 Mathematics @your_fingertips")
st.markdown("*AI-powered personalized learning paths through mathematics*")

# Step 1: Goal Input and Topic Discovery
st.subheader("🎯 Step 1: What do you want to learn?")

# Reset assessment if goal changes
if 'previous_goal' not in st.session_state:
    st.session_state.previous_goal = ""

# Use a form to capture Enter key presses
with st.form(key="goal_form", clear_on_submit=False):
    goal = st.text_input(
        "Describe your math learning goal:",
        placeholder="e.g., I want to understand calculus derivatives, learn linear algebra basics, master probability theory...",
        help="Be specific about what mathematical concepts you want to learn. Press Enter or click the button to start!",
        key="goal_input"
    )
    
    # Always show the submit button
    submitted = st.form_submit_button("🔍 Find Related Topics")

# Check if goal changed and reset states
if goal != st.session_state.previous_goal and goal:
    # Reset all states when goal changes
    st.session_state.topics_found = False
    st.session_state.categories_loaded = False
    st.session_state.knowledge_assessed = False
    st.session_state.learning_path_generated = False
    st.session_state.assessment_data = None
    st.session_state.wikipedia_categories = []
    st.session_state.category_knowledge = {}
    st.session_state.slider_assessment_data = None
    st.session_state.previous_goal = goal

# Execute analysis when form is submitted (Enter or button click)
if submitted and goal:
    with st.spinner("🤖 Analyzing with semantic similarity + graph traversal..."):
        st.session_state.assessment_data = conduct_semantic_assessment(goal, G)
        st.session_state.topics_found = True
        st.rerun()

# Display found topics
if st.session_state.topics_found and st.session_state.assessment_data:
    st.success("✅ Found related topics!")
    
    # Show direct matches
    direct_matches = st.session_state.assessment_data.get('direct_matches', [])
    if direct_matches:
        st.subheader("🎯 Direct Matches (Semantic Similarity)")
        for i, topic in enumerate(direct_matches):
            with st.expander(f"{i+1}. **{topic['title']}** (similarity: {topic['similarity_score']:.3f})"):
                st.write(f"**Summary:** {topic['summary']}")
                st.write(f"**Weight:** {topic['weight']:.3f} | **Difficulty:** {topic['difficulty']}")
    
    # Show graph-related topics
    graph_related = st.session_state.assessment_data.get('graph_related', [])
    if graph_related:
        st.subheader("🗺️ Graph-Related Topics (Connected via Knowledge Graph)")
        for i, topic in enumerate(graph_related[:5]):  # Show top 5
            with st.expander(f"{i+1}. **{topic['title']}** (priority: {topic['priority_score']:.2f})"):
                st.write(f"**Summary:** {topic['summary']}")
                st.write(f"**Weight:** {topic['weight']:.3f} | **Distance:** {topic['min_distance']} steps")
                st.write(f"**Path Quality:** {topic['max_path_quality']:.3f}")
                if topic.get('connection_paths'):
                    st.write("**Connection Paths:**")
                    for path in topic['connection_paths'][:2]:  # Show top 2 paths
                        st.write(f"  • {path}")
    
    # Step 2: Load Wikipedia Categories for Self-Assessment
    if not st.session_state.categories_loaded:
        if st.button("📚 Load Foundational Topics for Self-Assessment"):
            with st.spinner("📚 Loading foundational math topics..."):
                st.session_state.wikipedia_categories = extract_wikipedia_categories(G)
                # Initialize with persistent profile if available
                if st.session_state.persistent_knowledge_profile:
                    st.session_state.category_knowledge = st.session_state.persistent_knowledge_profile.copy()
                    st.info("🧠 Using your saved knowledge profile as starting point!")
                st.session_state.categories_loaded = True
                st.rerun()

# Step 3: Self-Assessment with Sliders
if st.session_state.categories_loaded and st.session_state.wikipedia_categories:
    st.subheader("📊 Step 2: Assess Your Current Knowledge")
    st.markdown("Select your knowledge level for each foundational math topic:")
    
    # Show if using persistent profile
    if st.session_state.persistent_knowledge_profile:
        st.success("🧠 **Using your saved knowledge profile!** You can adjust these values below or update your permanent profile at the bottom of the page.")
    
    # Create legend for knowledge levels
    st.markdown("""
    **Knowledge Levels:**
    - **0 - No Knowledge**: I don't know this area at all
    - **1 - Basic**: I know some fundamentals (1 graph layer)
    - **2 - Good**: I understand most concepts (2 graph layers)  
    - **3 - Expert**: I have deep knowledge (3 graph layers)
    """)
    
    st.info(f"📋 **Assessment Topics**: {len(st.session_state.wikipedia_categories)} foundational math topics (those with 0-1 prerequisites)")
    
    # Quick options for temporary assessment
    st.markdown("**Quick Options:**")
    temp_col1, temp_col2, temp_col3 = st.columns([2, 1, 1])
    with temp_col2:
        if st.button("🎲 Random Levels"):
            import random
            for category in st.session_state.wikipedia_categories:
                # Weighted random: more likely to have lower knowledge levels
                random_level = random.choices([0, 1, 2, 3], weights=[40, 30, 20, 10])[0]
                st.session_state.category_knowledge[category['id']] = random_level
            st.success("🎲 Random levels generated!")
            st.rerun()
    with temp_col3:
        if st.button("🗑️ Clear All"):
            for category in st.session_state.wikipedia_categories:
                st.session_state.category_knowledge[category['id']] = 0
            st.success("🗑️ All levels cleared!")
            st.rerun()
    
    # Create sliders for each Wikipedia category
    cols = st.columns(2)  # Two columns for better layout
    
    for i, category in enumerate(st.session_state.wikipedia_categories):
        col = cols[i % 2]  # Alternate between columns
        
        with col:
            # Create discrete level selector with category info
            slider_key = f"knowledge_{category['id']}"
            
            # Options for the select slider
            options = [0, 1, 2, 3]
            option_labels = [
                "0 - No Knowledge",
                "1 - Basic", 
                "2 - Good",
                "3 - Expert"
            ]
            
            # Get current value from category_knowledge (which may be initialized from persistent profile)
            current_value = st.session_state.category_knowledge.get(category['id'], 0)
            current_index = options.index(current_value) if current_value in options else 0
            
            knowledge_level = st.select_slider(
                f"**{category['title']}**",
                options=options,
                format_func=lambda x: option_labels[x],
                value=current_value,
                key=slider_key,
                help=f"Prerequisites: {category['prerequisite_count']} topics\n\n{category.get('summary', 'Mathematical topic')}\n\nEach level includes more graph connections:\n• Level 1: Direct connections\n• Level 2: 2-step connections\n• Level 3: 3-step connections"
            )
            st.session_state.category_knowledge[category['id']] = knowledge_level
    
    # Generate Learning Path Button
    if not st.session_state.knowledge_assessed:
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🛤️ Generate My Learning Path"):
                with st.spinner("🛤️ Generating personalized learning path..."):
                    st.session_state.slider_assessment_data = conduct_slider_based_assessment(
                        goal, G, st.session_state.category_knowledge
                    )
                    st.session_state.knowledge_assessed = True
                    st.session_state.learning_path_generated = True
                    st.rerun()
        with col2:
            if st.button("💾 Save to Profile"):
                st.session_state.persistent_knowledge_profile = st.session_state.category_knowledge.copy()
                st.success("✅ Saved to your knowledge profile!")
                st.rerun()

# Step 4: Display Learning Path
if st.session_state.learning_path_generated and st.session_state.slider_assessment_data:
    st.subheader("🛤️ Your Personalized Learning Path")
    
    learning_path = st.session_state.slider_assessment_data.get('learning_path', {})
    knowledge_mapping = st.session_state.slider_assessment_data.get('knowledge_mapping', {})
    
    # Show knowledge summary
    col1, col2, col3 = st.columns(3)
    with col1:
        known_count = len(knowledge_mapping.get('known_nodes', []))
        st.metric("🟢 Known Topics", known_count)
    with col2:
        partial_count = len(knowledge_mapping.get('partially_known_nodes', []))
        st.metric("🟡 Partially Known", partial_count)
    with col3:
        total_starting = len(st.session_state.slider_assessment_data.get('all_starting_nodes', []))
        st.metric("🚀 Starting Points", total_starting)
    
    # Show explanation of starting points
    if total_starting > 0:
        if known_count > 0 and partial_count > 0:
            st.info(f"🚀 **Path Generation**: Using {known_count} known + {partial_count} partially known = {total_starting} total starting points")
        elif known_count > 0:
            st.info(f"🚀 **Path Generation**: Using {known_count} known topics as starting points")
        elif partial_count > 0:
            st.info(f"🚀 **Path Generation**: Using {partial_count} partially known topics as starting points")
    else:
        st.info("🚀 **Path Generation**: Will use foundational topics (0 prerequisites) as starting points")
    
    # Display path message
    st.info(learning_path.get('message', 'Learning path generated!'))
    
    # Display learning path steps
    path_steps = learning_path.get('path', [])
    if path_steps:
        st.markdown("### 📋 Comprehensive Learning Path:")
        st.caption("This path covers ALL prerequisites needed to reach your target topic")
        
        for i, step in enumerate(path_steps):
            target_indicator = "🎯 " if step.get('is_target') else ""
            prerequisite_level = step.get('prerequisite_level', 0)
            level_badge = f"L{prerequisite_level}" if prerequisite_level > 0 else "Foundation"
            
            with st.expander(f"**Step {step['step_number']}: {target_indicator}{step['title']}** `{level_badge}`"):
                st.write(f"**Summary:** {step['summary']}")
                st.write(f"**Weight:** {step['weight']:.3f} | **Difficulty:** {step['difficulty']}")
                
                if prerequisite_level > 0:
                    st.info(f"📊 **Prerequisite Level:** {prerequisite_level} (requires {prerequisite_level} layers of prior knowledge)")
                else:
                    st.info("🏁 **Foundation Topic** - No prerequisites required")
                
                if step.get('is_target'):
                    st.success("🎯 This is your target topic!")
        
        # Show comprehensive path summary
        path_details = learning_path.get('path_details', {})
        if path_details and path_details.get('path_type') == 'comprehensive_prerequisites':
            st.markdown("### 📊 Learning Path Summary:")
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                st.metric("📚 Total Steps", path_details.get('total_steps', 0))
            with summary_col2:
                st.metric("✅ Prerequisites Covered", path_details.get('prerequisites_covered', 0))
            with summary_col3:
                st.metric("📈 Max Prerequisite Level", path_details.get('max_prerequisite_level', 0))
            
            already_known = path_details.get('already_known', 0)
            if already_known > 0:
                st.success(f"🎉 Great news! You already know {already_known} prerequisite topics, so they're not included in this path.")
        
        # Show path details - REMOVED Path Analysis section

# Graph Visualization (moved to the end)
st.subheader("🌐 Knowledge Graph Visualization")

# Prepare highlighting information
highlight_info = None
if st.session_state.slider_assessment_data:
    # Use slider-based assessment data for highlighting
    direct_match_ids = [topic['id'] for topic in st.session_state.slider_assessment_data.get('direct_matches', [])]
    graph_related_ids = [topic['id'] for topic in st.session_state.slider_assessment_data.get('graph_related', [])]
    target_topic_id = st.session_state.slider_assessment_data.get('target_topic')
    
    # Extract known topics and learning path
    knowledge_mapping = st.session_state.slider_assessment_data.get('knowledge_mapping', {})
    # Use combined starting nodes (known + partially known) for highlighting
    all_starting_nodes = st.session_state.slider_assessment_data.get('all_starting_nodes', [])
    if not all_starting_nodes:
        # Fallback to just known nodes if all_starting_nodes not available
        all_starting_nodes = knowledge_mapping.get('known_nodes', [])
    
    learning_path = st.session_state.slider_assessment_data.get('learning_path', {})
    learning_path_ids = [step['id'] for step in learning_path.get('path', [])]
    
    highlight_info = {
        'direct_matches': direct_match_ids,
        'graph_related': graph_related_ids,
        'target_topic': target_topic_id,
        'known_topics': all_starting_nodes,  # Use combined starting nodes
        'learning_path': learning_path_ids
    }
elif st.session_state.assessment_data:
    # Fallback to semantic assessment data
    direct_match_ids = [topic['id'] for topic in st.session_state.assessment_data.get('direct_matches', [])]
    graph_related_ids = [topic['id'] for topic in st.session_state.assessment_data.get('graph_related', [])]
    target_topic_id = st.session_state.assessment_data.get('target_topic')
    
    highlight_info = {
        'direct_matches': direct_match_ids,
        'graph_related': graph_related_ids,
        'target_topic': target_topic_id,
        'known_topics': [],  # No knowledge mapping in basic semantic assessment
        'learning_path': []  # No learning path in basic semantic assessment
    }

graph_html = draw_graph(G, highlight_info)
st.components.v1.html(graph_html, height=600)

# Legend for graph colors
if highlight_info:
    st.markdown("### 🎨 Graph Legend:")
    legend_col1, legend_col2, legend_col3 = st.columns(3)
    with legend_col1:
        st.markdown("🟡 **Target Topic** - Your main learning goal")
        st.markdown("🟢 **Direct Matches** - Semantically similar topics")
    with legend_col2:
        st.markdown("🔴 **Graph Related** - Connected topics in knowledge graph")
        st.markdown("🔵 **Known Topics** - Topics you already know")
    with legend_col3:
        st.markdown("🟣 **Learning Path** - Recommended learning sequence")
        if highlight_info.get('learning_path'):
            st.markdown(f"*Path has {len(highlight_info['learning_path'])} steps*")

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

# === PERSISTENT KNOWLEDGE ASSESSMENT SECTION (Bottom of Page) ===
st.markdown("---")
st.header("🧠 Your Personal Knowledge Profile")
st.markdown("*Set your knowledge levels once and they'll be remembered for all searches*")

# Load foundational topics for persistent assessment
if not st.session_state.knowledge_profile_loaded:
    with st.spinner("Loading foundational topics for knowledge profile..."):
        foundational_topics = extract_wikipedia_categories(G)
        st.session_state.foundational_topics = foundational_topics
        st.session_state.knowledge_profile_loaded = True

# Display persistent knowledge assessment
if st.session_state.knowledge_profile_loaded:
    st.subheader("📊 Set Your Knowledge Levels")
    st.markdown("These settings will be remembered and used for all your learning path generations.")
    
    # Create legend for knowledge levels
    with st.expander("📖 Knowledge Level Guide", expanded=False):
        st.markdown("""
        **Knowledge Levels:**
        - **0 - No Knowledge**: I don't know this area at all
        - **1 - Basic**: I know some fundamentals (1 graph layer)
        - **2 - Good**: I understand most concepts (2 graph layers)  
        - **3 - Expert**: I have deep knowledge (3 graph layers)
        
        **How it works**: Each level includes topics at different distances in the knowledge graph:
        - Level 1: Direct connections to the topic
        - Level 2: Topics 2 steps away 
        - Level 3: Topics 3 steps away
        """)
    
    st.info(f"📋 **Profile Topics**: {len(st.session_state.foundational_topics)} foundational math topics (those with 0-1 prerequisites)")
    
    # Create form for persistent knowledge assessment
    with st.form(key="knowledge_profile_form"):
        st.markdown("### 🎚️ Adjust Your Knowledge Levels:")
        
        # Add random generation option
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**Quick Options:**")
        with col2:
            if st.form_submit_button("🎲 Generate Random Profile"):
                import random
                # Generate random knowledge levels for demonstration
                for topic in st.session_state.foundational_topics:
                    # Weighted random: more likely to have lower knowledge levels
                    random_level = random.choices([0, 1, 2, 3], weights=[40, 30, 20, 10])[0]
                    st.session_state.persistent_knowledge_profile[topic['id']] = random_level
                st.success("🎲 Random knowledge profile generated!")
                st.rerun()
        with col3:
            if st.form_submit_button("🗑️ Clear All"):
                st.session_state.persistent_knowledge_profile = {}
                st.success("🗑️ All knowledge levels cleared!")
                st.rerun()
        
        # Create columns for better layout
        cols = st.columns(3)  # Three columns for more compact display
        
        for i, topic in enumerate(st.session_state.foundational_topics):
            col = cols[i % 3]  # Cycle through columns
            
            with col:
                # Get current value from persistent profile or default to 0
                current_value = st.session_state.persistent_knowledge_profile.get(topic['id'], 0)
                
                # Options for the select slider
                options = [0, 1, 2, 3]
                option_labels = [
                    "0 - No Knowledge",
                    "1 - Basic", 
                    "2 - Good",
                    "3 - Expert"
                ]
                
                knowledge_level = st.select_slider(
                    f"**{topic['title']}**",
                    options=options,
                    format_func=lambda x: option_labels[x],
                    value=current_value,
                    key=f"persistent_knowledge_{topic['id']}",
                    help=f"Prerequisites: {topic['prerequisite_count']} topics\n\n{topic.get('summary', 'Mathematical topic')}"
                )
                
                # Update persistent profile
                st.session_state.persistent_knowledge_profile[topic['id']] = knowledge_level
        
        # Save button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submitted = st.form_submit_button("💾 Save Knowledge Profile", use_container_width=True)
    
    if submitted:
        # Update the category_knowledge to use persistent profile for current session
        st.session_state.category_knowledge = st.session_state.persistent_knowledge_profile.copy()
        
        # Show confirmation
        total_knowledge = sum(st.session_state.persistent_knowledge_profile.values())
        topics_with_knowledge = len([k for k in st.session_state.persistent_knowledge_profile.values() if k > 0])
        
        st.success(f"✅ Knowledge profile saved! {topics_with_knowledge} topics with knowledge (total score: {total_knowledge})")
        
        # If user has an active learning goal, regenerate the path with new knowledge
        if st.session_state.topics_found and st.session_state.assessment_data:
            with st.spinner("🔄 Updating your learning path with new knowledge profile..."):
                st.session_state.slider_assessment_data = conduct_slider_based_assessment(
                    st.session_state.get('previous_goal', ''), G, st.session_state.category_knowledge
                )
                st.session_state.knowledge_assessed = True
                st.session_state.learning_path_generated = True
                st.rerun()
    
    # Display current profile summary
    if st.session_state.persistent_knowledge_profile:
        st.subheader("📈 Your Current Knowledge Profile")
        
        # Calculate profile statistics
        total_topics = len(st.session_state.foundational_topics)
        knowledge_levels = list(st.session_state.persistent_knowledge_profile.values())
        topics_with_knowledge = len([k for k in knowledge_levels if k > 0])
        avg_knowledge = sum(knowledge_levels) / len(knowledge_levels) if knowledge_levels else 0
        
        # Display stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Topics", total_topics)
        with col2:
            st.metric("✅ Topics Known", topics_with_knowledge)
        with col3:
            st.metric("📈 Average Level", f"{avg_knowledge:.1f}")
        with col4:
            knowledge_score = sum(knowledge_levels)
            st.metric("🎯 Knowledge Score", knowledge_score)
        
        # Show knowledge distribution
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for level in knowledge_levels:
            level_counts[level] += 1
        
        st.markdown("**Knowledge Distribution:**")
        dist_col1, dist_col2, dist_col3, dist_col4 = st.columns(4)
        with dist_col1:
            st.write(f"🔴 No Knowledge: {level_counts[0]}")
        with dist_col2:
            st.write(f"🟡 Basic: {level_counts[1]}")
        with dist_col3:
            st.write(f"🟠 Good: {level_counts[2]}")
        with dist_col4:
            st.write(f"🟢 Expert: {level_counts[3]}")
    
    # Reset profile option
    with st.expander("🔄 Reset Knowledge Profile", expanded=False):
        st.warning("This will reset all your knowledge levels to 0. This action cannot be undone.")
        if st.button("🗑️ Reset All Knowledge Levels"):
            st.session_state.persistent_knowledge_profile = {}
            st.session_state.category_knowledge = {}
            st.success("Knowledge profile reset!")
            st.rerun()