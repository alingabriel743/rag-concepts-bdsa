"""
RAG Walkthrough Streamlit Application
=====================================
A comprehensive, multi-page Streamlit app that teaches RAG concepts
through theory and interactive demos.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="RAG Walkthrough",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .concept-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .highlight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Step indicators */
    .step-number {
        display: inline-block;
        width: 30px;
        height: 30px;
        background: #667eea;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 30px;
        font-weight: bold;
        margin-right: 10px;
    }
    
    /* Progress indicator */
    .progress-container {
        margin: 2rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main content
st.markdown('<h1 class="main-header">🔗 RAG Walkthrough</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Master Retrieval-Augmented Generation from Theory to Practice</p>', unsafe_allow_html=True)

# Introduction
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎯 What You'll Learn")
    st.markdown("""
    - What RAG is and why it matters
    - How embeddings capture meaning
    - Vector database fundamentals
    - Document processing strategies
    - Retrieval mechanisms
    - Building complete RAG pipelines
    """)

with col2:
    st.markdown("### 🛠️ Interactive Features")
    st.markdown("""
    - Live embedding visualizations
    - ChromaDB playground
    - Document chunking demos
    - Similarity search explorer
    - End-to-end RAG demo
    - Code examples you can run
    """)

with col3:
    st.markdown("### 📚 Page Overview")
    st.markdown("""
    1. **Introduction to RAG**
    2. **Understanding Embeddings**
    3. **Vector Databases**
    4. **Document Processing**
    5. **Retrieval Mechanisms**
    6. **Generation with Context**
    7. **Complete RAG Pipeline**
    8. **Best Practices**
    """)

st.markdown("---")

# RAG Overview Diagram
st.markdown("## 🔄 The RAG Pipeline at a Glance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 3rem;">📝</div>
        <h4>1. Query</h4>
        <p style="font-size: 0.9rem; color: #666;">User asks a question</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 3rem;">🔍</div>
        <h4>2. Retrieve</h4>
        <p style="font-size: 0.9rem; color: #666;">Find relevant documents</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 3rem;">📄</div>
        <h4>3. Augment</h4>
        <p style="font-size: 0.9rem; color: #666;">Add context to prompt</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 3rem;">🤖</div>
        <h4>4. Generate</h4>
        <p style="font-size: 0.9rem; color: #666;">LLM creates answer</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Getting Started
st.markdown("## 🚀 Getting Started")

st.info("""
👈 **Use the sidebar to navigate through the pages.** 

Each page builds on the previous one, so we recommend going through them in order for the best learning experience.
""")

# Quick stats
st.markdown("### 📊 Quick Stats")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Pages", "8", help="8 comprehensive learning modules")

with col2:
    st.metric("Interactive Demos", "10+", help="Hands-on exercises")

with col3:
    st.metric("Code Examples", "20+", help="Working code snippets")

with col4:
    st.metric("Concepts", "25+", help="Key RAG concepts covered")

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>Built with ❤️ using Streamlit | Vector DB: ChromaDB | Embeddings: Sentence-Transformers</p>
</div>
""", unsafe_allow_html=True)
