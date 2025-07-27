# 🧮 Mathematics @your_fingertips

> **AI-Powered Comprehensive Learning Paths Through Mathematics**

An intelligent Streamlit application that creates personalized, mathematically sound learning paths using semantic analysis, knowledge graphs, and comprehensive prerequisite coverage. The system combines sentence transformers for semantic topic matching with graph-based prerequisite analysis to generate complete learning roadmaps that cover all necessary mathematical foundations.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![Qwen](https://img.shields.io/badge/model-Qwen2.5--0.5B-green.svg)
![SentenceTransformers](https://img.shields.io/badge/embeddings-all--MiniLM--L6--v2-purple.svg)
![NetworkX](https://img.shields.io/badge/graph-NetworkX-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Key Features

### 🎯 **Semantic Goal Analysis**
- **Natural Language Understanding**: Enter learning goals in plain English (e.g., "I want to learn machine learning")
- **Sentence Transformer Matching**: Uses `all-MiniLM-L6-v2` for semantic similarity analysis
- **Smart Topic Discovery**: Finds the most relevant mathematical topics from your description
- **Weighted Graph Traversal**: Discovers related topics through intelligent graph exploration

### 🧠 **Persistent Knowledge Profiles**
- **Self-Assessment System**: Intuitive slider-based knowledge evaluation on foundational topics
- **Discrete Knowledge Levels**: 4-level system (0=None, 1=Basic, 2=Good, 3=Expert)
- **Graph Layer Mapping**: Each level corresponds to different depths in the knowledge graph
- **Persistent Storage**: Your knowledge profile is remembered across all searches
- **Quick Demo Features**: Random profile generation for easy demonstrations

### 🛤️ **Comprehensive Prerequisite Coverage**
- **Complete Learning Paths**: Covers ALL prerequisites needed for your target topic
- **Topological Ordering**: Ensures proper learning sequence respecting dependencies
- **Prerequisite Level Analysis**: Shows how many layers of prior knowledge each topic requires
- **No Knowledge Gaps**: Systematic coverage from foundations to advanced concepts
- **Personalized Starting Points**: Begins from your assessed knowledge level

### 📊 **Advanced Graph Visualization**
- **Multi-Color Highlighting**: Different colors for target, known topics, learning path, and related concepts
- **Interactive PyVis Network**: Explore the complete mathematics knowledge graph
- **Real-Time Updates**: Graph highlights change based on your goals and knowledge
- **Comprehensive Legend**: Clear explanation of all highlighting colors and meanings

### 🎲 **Demo-Friendly Features**
- **Random Profile Generation**: One-click realistic knowledge profiles for demonstrations
- **Quick Reset Options**: Easy clearing and regeneration of assessments
- **Streamlined UI**: Clean, intuitive interface with progressive disclosure
- **Enter Key Support**: Natural form interactions with keyboard shortcuts

## 🗺️ Knowledge Graph Structure

### 📚 **Comprehensive Topic Coverage**
- **500+ Mathematical Topics**: From basic arithmetic to advanced machine learning
- **Prerequisite-Based Organization**: Topics organized by their mathematical dependencies
- **Rich Descriptions**: Each topic includes detailed mathematical descriptions
- **Foundational Categories**: 20+ foundational topics (0-1 prerequisites) for assessment

### 🔗 **Intelligent Relationships**
- **Prerequisite Mapping**: Direct prerequisite relationships between mathematical concepts
- **Multi-Level Dependencies**: Topics can have complex, multi-layer prerequisite chains
- **Weight-Based Prioritization**: Topics prioritized by mathematical importance
- **Cycle Detection**: Safe handling of any circular dependencies in the graph

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- ~1GB RAM (for embedding models)
- ~300MB storage (for model caching)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/mathematics-at-your-fingertips.git
   cd mathematics-at-your-fingertips
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
   - Navigate to `http://localhost:8501`
   - The application will automatically load the knowledge graph

## 💡 How to Use

### 🎯 **Step 1: Set Your Learning Goal**
1. Enter your learning objective in natural language
2. Press Enter or click "Find Related Topics"
3. View semantically matched topics and graph-related concepts

### 📊 **Step 2: Assess Your Knowledge**
1. Use the foundational topic sliders to indicate your knowledge levels
2. Choose from 4 levels: None (0), Basic (1), Good (2), Expert (3)
3. Each level includes different depths of graph connections
4. Save your profile for future use

### 🛤️ **Step 3: Generate Learning Path**
1. Click "Generate My Learning Path"
2. View your comprehensive prerequisite-based learning sequence
3. See prerequisite levels and mathematical dependencies
4. Track your progress through the recommended steps

### 🌐 **Step 4: Explore the Graph**
1. Examine the interactive knowledge graph visualization
2. See your known topics (blue), learning path (purple), and target (gold)
3. Hover over nodes for detailed topic information
4. Explore mathematical relationships and connections

## 🎨 Color-Coded Graph Legend

- **🟡 Gold**: Your target learning topic
- **🟣 Purple**: Recommended learning path steps
- **🔵 Blue**: Topics you already know (starting points)
- **🟢 Green**: Direct semantic matches to your goal
- **🔴 Red**: Graph-related topics (connected concepts)

## 🔧 Technical Architecture

### 🧠 **AI & ML Components**
- **Qwen2.5-0.5B-Instruct**: For problem generation and answer assessment
- **SentenceTransformers**: For semantic similarity analysis
- **FAISS**: For efficient similarity search (when needed)
- **Custom Prompt Engineering**: Optimized prompts for mathematical content

### 📊 **Graph Processing**
- **NetworkX**: For graph analysis and pathfinding algorithms
- **PyVis**: For interactive graph visualization
- **Custom Algorithms**: Comprehensive prerequisite discovery and topological ordering
- **YAML Data**: Human-readable knowledge graph format

### 🎨 **User Interface**
- **Streamlit**: Modern, responsive web interface
- **Session State Management**: Persistent user data across interactions
- **Form-Based Interactions**: Natural keyboard and mouse interactions
- **Progressive Disclosure**: Information revealed as needed

## 📁 Project Structure

```
mathematics-at-your-fingertips/
├── app.py                      # Main Streamlit application
├── graph_g.yaml               # Knowledge graph with descriptions
├── requirements.txt           # Python dependencies
├── utils/
│   ├── graph_loader.py        # Graph loading and processing
│   ├── hf_llm.py             # LLM integration and semantic analysis
│   └── retrieval.py          # Embedding and similarity search
├── ui_components/
│   └── graph_viz.py          # Interactive graph visualization
└── README.md                 # This file
```

## 🎯 Key Algorithms

### 📈 **Semantic Topic Matching**
1. **Encode user input** using sentence transformers
2. **Generate embeddings** for all mathematical topics
3. **Compute cosine similarity** between user goal and topics
4. **Filter and rank** results by similarity threshold (>0.3)
5. **Expand via graph traversal** to find related concepts

### 🛤️ **Comprehensive Path Generation**
1. **Discover all prerequisites** using recursive graph traversal
2. **Calculate prerequisite levels** for proper ordering
3. **Apply topological sorting** respecting mathematical dependencies
4. **Prioritize by importance** using topic weights
5. **Generate complete sequence** from foundations to target

### 🧠 **Knowledge Assessment Mapping**
1. **Map slider levels** to graph connection depths
2. **Combine known and partial knowledge** as starting points
3. **Filter prerequisite coverage** based on existing knowledge
4. **Generate personalized paths** from your knowledge base

## 🎲 Demo Features

### **Quick Profile Generation**
- **Random Knowledge Profiles**: Realistic distributions for demonstrations
- **Weighted Randomization**: More likely to generate lower knowledge levels
- **Instant Reset**: Quick clearing for fresh demonstrations

### **Streamlined Workflow**
- **One-Click Operations**: Generate profiles and paths with single clicks
- **Clear Visual Feedback**: Immediate updates and confirmations
- **Persistent Profiles**: Knowledge saved across multiple searches

## 🚀 Advanced Features

### **Multi-Level Knowledge Mapping**
- **Level 1**: Direct connections (1 graph layer)
- **Level 2**: 2-step connections (2 graph layers)  
- **Level 3**: 3-step connections (3 graph layers)
- **Combined Starting Points**: Uses both known and partially known topics

### **Intelligent Fallbacks**
- **Zero Knowledge Handling**: Automatically uses foundational topics as starting points
- **Graph Validation**: Ensures paths are mathematically sound
- **Error Recovery**: Graceful handling of edge cases and missing data

## 📚 Educational Philosophy

This system is built on the principle that **mathematics is hierarchical** - advanced concepts build systematically on foundational knowledge. Rather than providing shortcuts, it ensures comprehensive coverage of all prerequisite knowledge, creating educationally sound learning paths that respect the inherent structure of mathematical knowledge.

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests. Areas where contributions are especially valuable:

- **Knowledge Graph Expansion**: Adding more mathematical topics and relationships
- **Algorithm Improvements**: Better pathfinding and assessment algorithms
- **UI/UX Enhancements**: Improved user interface and experience
- **Educational Content**: Better problem generation and assessment

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For transformer models and sentence embeddings
- **NetworkX**: For powerful graph analysis capabilities
- **Streamlit**: For the excellent web application framework
- **PyVis**: For interactive network visualizations
- **Mathematical Community**: For the foundational knowledge that makes this possible

---

*Transform your mathematical learning journey with AI-powered, comprehensive prerequisite coverage. No gaps, no shortcuts - just complete, personalized learning paths through the beautiful structure of mathematics.* 🧮✨
