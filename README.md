# 📚 Math Learning Navigator

> **AI-Powered Graph-Based Math Assessment & Optimal Learning Paths**

An intelligent Streamlit application that combines knowledge graph pathfinding with AI assessment to create optimal learning roadmaps. Using NetworkX graph algorithms and Large Language Models, the system identifies your current knowledge level through math problem assessment and generates the shortest learning path to your goal.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![Qwen](https://img.shields.io/badge/model-Qwen2.5--0.5B-green.svg)
![NetworkX](https://img.shields.io/badge/graph-NetworkX-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Key Features

### 🧠 **AI-Powered Knowledge Assessment**
- **Intent Extraction**: LLM analyzes your learning goals and identifies target mathematical concepts
- **Graph-Based Topic Matching**: Finds your target topic within a comprehensive knowledge graph of 369+ math concepts
- **Real-Time Problem Generation**: Creates actual math problems to test prerequisite knowledge
- **Intelligent Answer Assessment**: LLM evaluates your solutions and provides detailed feedback

### 🗺️ **Graph-Based Optimal Pathfinding**
- **Shortest Path Algorithms**: Uses NetworkX to find optimal learning routes from your known topics to your goal
- **Multiple Path Analysis**: Evaluates all possible learning routes and selects the most efficient one
- **Prerequisite Mapping**: Leverages 1,327+ concept relationships to ensure proper learning sequence
- **Alternative Routes**: Shows backup learning paths if the primary route doesn't work

### 📊 **Enhanced Learning Roadmaps**
- **Pathfinding + AI**: Combines graph theory optimization with LLM-generated learning activities
- **Step-by-Step Guidance**: Clear progression from your current knowledge to target mastery
- **Personalized Activities**: Specific exercises and assessments for each learning step
- **Progress Tracking**: Built-in completion tracking with roadmap metadata

### 🎯 **Interactive Knowledge Graph**
- **369 Mathematical Concepts**: Comprehensive first-year mathematics topic coverage
- **1,327 Relationships**: Detailed prerequisite and conceptual connections
- **Visual Exploration**: Interactive PyVis graph with hover details and topic highlighting
- **Real-Time Updates**: Graph highlights your target topic and related concepts

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- ~2GB RAM (for LLM model)
- ~500MB storage (for model caching)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd math-learning-navigator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 📖 How It Works

### Step 1: Define Your Learning Goal 🎯
Enter your math learning objective in natural language:

**Examples:**
- "I want to understand calculus derivatives"
- "I need to learn linear algebra for machine learning"
- "Help me master quadratic equations"
- "I'm struggling with trigonometry identities"

### Step 2: AI Analysis & Graph Matching 🧠
The system will:
- **Extract Intent**: LLM analyzes your goal to identify the target concept, difficulty level, and context
- **Find Target Topic**: Advanced matching algorithm locates your goal within the knowledge graph
- **Identify Prerequisites**: Graph traversal finds related topics you need to know first

### Step 3: Mathematical Assessment 📝
Answer AI-generated math problems that test:
- **Prerequisite Knowledge**: Problems covering topics that lead to your goal
- **Current Understanding**: Assessment of your existing mathematical foundation
- **Problem-Solving Skills**: Real math problems with step-by-step evaluation

### Step 4: Optimal Path Generation 🗺️
Receive:
- **Shortest Learning Path**: Graph algorithms find the most efficient route to your goal
- **Alternative Routes**: Backup paths if you prefer different approaches
- **Step Metadata**: Total steps, starting point, and path summary
- **Enhanced Activities**: LLM-generated specific learning tasks for each step

### Step 5: Personalized Resources & Tracking 📚
- **AI-Generated Resources**: Tailored learning materials for each roadmap step
- **Progress Tracking**: Mark steps and resources as completed
- **Assessment Breakdown**: Review how the AI evaluated your responses
- **Path Visualization**: See your learning journey on the interactive graph

## 🏗️ Project Structure

```
math-learning-navigator/
├── app.py                          # Main Streamlit application
├── math_firstyear.yaml            # Knowledge graph (369 topics, 1,327 relationships)
├── requirements.txt               # Python dependencies
├── README.md                      # This documentation
├── data/
│   └── metas.json                # Metadata for RAG system
├── utils/
│   ├── hf_llm.py                 # LLM integration & pathfinding logic
│   ├── graph_loader.py           # Knowledge graph loading
│   ├── retrieval.py              # RAG system for context
│   └── scraper.py                # Resource generation
└── ui_components/
    └── graph_viz.py              # Interactive graph visualization
```

## 🤖 AI & Graph Technology

### Large Language Model
- **Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Capabilities**: Intent extraction, problem generation, answer assessment, roadmap enhancement
- **Performance**: ~2-5 seconds per generation, ~1-2GB memory usage
- **Offline**: Model cached locally after first download

### Knowledge Graph
- **Nodes**: 369 mathematical topics (linear equations, derivatives, matrices, etc.)
- **Edges**: 1,327 relationships (89.1% "related_to", 10.9% "on_path")
- **Structure**: Directed graph with topic metadata (title, summary, difficulty)
- **Pathfinding**: NetworkX shortest_path algorithm for optimal learning routes

### Assessment System
- **Problem Generation**: LLM creates topic-specific math problems without revealing solutions
- **Answer Evaluation**: Two-step process - LLM solves internally, then compares to student answer
- **Knowledge Mapping**: Maps correct/incorrect responses to known/unknown topic lists
- **Adaptive Flow**: Assessment results directly influence pathfinding starting points

## 🎯 Learning Approaches

The system intelligently combines multiple learning methodologies:

### 🧮 **Problem-Based Learning**
- Real math problems generated for each prerequisite topic
- Step-by-step solution assessment with detailed feedback
- Progressive difficulty based on graph topology
- **Best for**: Skill building, computational fluency, exam preparation

### 🗺️ **Graph-Based Sequencing**
- Optimal learning paths based on mathematical prerequisites
- Shortest routes from known concepts to target mastery
- Alternative pathways for different learning preferences
- **Best for**: Systematic understanding, comprehensive coverage

### 🤖 **AI-Enhanced Activities**
- Personalized learning tasks for each roadmap step
- Context-aware resource recommendations
- Adaptive content based on assessment results
- **Best for**: Personalized learning, targeted skill development

## 🔧 Technical Details

### Core Dependencies
```
streamlit              # Web application framework
networkx              # Graph algorithms & pathfinding
pyvis                 # Interactive graph visualization
transformers          # Hugging Face LLM integration
sentence-transformers # Text embeddings for RAG
faiss-cpu            # Vector similarity search
torch                # PyTorch backend for models
pyyaml               # Knowledge graph parsing
numpy                # Numerical computing
```

### Performance Metrics
- **First Run**: ~30-60 seconds (LLM model download)
- **Subsequent Runs**: ~2-5 seconds per AI operation
- **Memory Usage**: ~1-2GB (includes model and graph)
- **Storage**: ~500MB (cached model files)
- **Graph Operations**: <1 second (pathfinding, traversal)

### Customization Options
- **Expand Knowledge Graph**: Add topics and relationships to `math_firstyear.yaml`
- **Modify LLM Prompts**: Update templates in `utils/hf_llm.py`
- **Adjust Assessment**: Configure problem generation and evaluation logic
- **Custom Visualizations**: Modify graph styling in `ui_components/graph_viz.py`

## 🎯 Example Use Cases

### 📚 **Calculus Preparation**
*Goal*: "I want to understand derivatives for my calculus course"

*Process*:
1. AI identifies "derivatives" as target topic in knowledge graph
2. Generates problems testing: limits, basic functions, algebraic manipulation
3. Pathfinding finds: basic_algebra → functions → limits → derivatives
4. Creates 4-step roadmap with specific activities for each concept

*Result*: Optimal 4-step learning path with personalized math problems and resources

### 💼 **Linear Algebra for ML**
*Goal*: "I need linear algebra for machine learning"

*Process*:
1. AI maps goal to "matrix_operations" and "eigenvalues" topics
2. Tests prerequisite knowledge: vectors, systems of equations, basic algebra
3. Graph pathfinding identifies shortest route through vector spaces
4. Generates enhanced roadmap with ML-specific applications

*Result*: Targeted learning path optimized for machine learning applications

### 🎓 **Comprehensive Review**
*Goal*: "I have a math placement exam and need to review everything"

*Process*:
1. AI identifies multiple target topics across algebra, geometry, trigonometry
2. Assessment covers broad range of foundational concepts
3. Pathfinding generates multiple optimal routes to different topic clusters
4. Creates comprehensive review plan with priority sequencing

*Result*: Systematic review roadmap covering all essential topics efficiently

## 🤝 Contributing

We welcome contributions! Areas for enhancement:

- **Knowledge Graph Expansion**: Add advanced mathematics topics
- **Assessment Refinement**: Improve problem generation and evaluation
- **Visualization Features**: Enhanced graph interactions and learning analytics
- **Performance Optimization**: Faster LLM inference and graph operations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Qwen Team**: For the efficient 0.5B parameter language model
- **NetworkX Community**: For robust graph algorithms
- **Streamlit Team**: For the excellent web application framework
- **Mathematical Community**: For the foundational knowledge represented in our graph

---

*Built with ❤️ for mathematics education and optimal learning paths*
