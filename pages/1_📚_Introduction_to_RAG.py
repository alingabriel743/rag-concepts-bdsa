"""
Page 1: Introduction to RAG
===========================
What is RAG, why it matters, and how it solves LLM limitations.
"""

import streamlit as st

st.set_page_config(
    page_title="Introduction to RAG",
    page_icon="📚",
    layout="wide"
)

# Title
st.markdown("# 📚 Introduction to RAG")
st.markdown("*Retrieval-Augmented Generation: Giving LLMs Access to Your Data*")
st.markdown("---")

# What is RAG Section
st.markdown("## 🤔 What is RAG?")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **RAG (Retrieval-Augmented Generation)** is an AI framework that combines:
    
    1. **Information Retrieval** — Finding relevant documents from a knowledge base
    2. **Text Generation** — Using an LLM to generate responses based on retrieved context
    
    Instead of relying solely on the LLM's training data, RAG allows models to access 
    **external, up-to-date information** at query time.
    """)

with col2:
    st.info("""
    **RAG in one sentence:**
    
    *"Look up relevant information, then answer the question using what you found."*
    """)

st.markdown("---")

# Why RAG Section
st.markdown("## 🎯 Why Do We Need RAG?")

st.markdown("### The Problem with Pure LLMs")

col1, col2, col3 = st.columns(3)

with col1:
    st.error("""
    **🧠 Knowledge Cutoff**
    
    LLMs only know information up to their training date. They can't answer about recent events.
    
    *"What happened in the news yesterday?"*
    """)

with col2:
    st.error("""
    **👻 Hallucinations**
    
    LLMs sometimes generate plausible-sounding but incorrect information.
    
    *"The Eiffel Tower was built in 1756..."* (wrong!)
    """)

with col3:
    st.error("""
    **🔒 No Private Data**
    
    LLMs don't have access to your company documents, internal wikis, or personal files.
    
    *"What's our Q3 revenue?"*
    """)

st.markdown("### How RAG Solves These Problems")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("""
    **📅 Always Up-to-Date**
    
    RAG retrieves from your current knowledge base, so information is always fresh.
    """)

with col2:
    st.success("""
    **📝 Grounded Answers**
    
    Answers are based on actual documents you provide, reducing hallucinations.
    """)

with col3:
    st.success("""
    **🔓 Access Your Data**
    
    Connect your own documents, databases, and knowledge sources.
    """)

st.markdown("---")

# RAG Pipeline Section
st.markdown("## 🔄 The RAG Pipeline")

st.markdown("""
The RAG process follows these key steps:
""")

# Visual pipeline
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 1.5rem; border-radius: 10px; height: 180px;">
        <h2>1️⃣</h2>
        <h4>User Query</h4>
        <p style="font-size: 0.9rem;">User asks a question</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                color: white; padding: 1.5rem; border-radius: 10px; height: 180px;">
        <h2>2️⃣</h2>
        <h4>Retrieve</h4>
        <p style="font-size: 0.9rem;">Find relevant documents from vector DB</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                color: white; padding: 1.5rem; border-radius: 10px; height: 180px;">
        <h2>3️⃣</h2>
        <h4>Augment</h4>
        <p style="font-size: 0.9rem;">Add context to the prompt</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="text-align: center; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                color: white; padding: 1.5rem; border-radius: 10px; height: 180px;">
        <h2>4️⃣</h2>
        <h4>Generate</h4>
        <p style="font-size: 0.9rem;">LLM generates the answer</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Interactive Demo
st.markdown("## 🎮 Interactive: With vs Without RAG")

st.markdown("See the difference between a pure LLM response and a RAG-augmented response:")

# Example question
question = st.text_input(
    "Ask a question about a hypothetical company:",
    value="What is TechCorp's refund policy?",
    key="demo_question"
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ❌ Without RAG")
    st.markdown("""
    <div style="background: #ffebee; padding: 1rem; border-radius: 10px; border-left: 4px solid #f44336;">
        <strong>LLM Response:</strong><br><br>
        "I don't have specific information about TechCorp's refund policy. 
        Generally, most companies offer 30-day refund policies, but you should 
        check their official website or contact customer support for accurate information."
        <br><br>
        <em style="color: #666;">❌ Vague, generic answer. No actual information.</em>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### ✅ With RAG")
    
    # Simulated context
    context_shown = st.checkbox("Show retrieved context", value=True)
    
    if context_shown:
        st.markdown("""
        <div style="background: #fff3e0; padding: 0.8rem; border-radius: 8px; margin-bottom: 1rem; font-size: 0.9rem;">
            <strong>📄 Retrieved Context:</strong><br>
            "TechCorp offers a 60-day money-back guarantee on all products. 
            Customers can request a full refund within 60 days of purchase by 
            contacting support@techcorp.com. Digital products are refundable 
            within 14 days if not downloaded."
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #e8f5e9; padding: 1rem; border-radius: 10px; border-left: 4px solid #4caf50;">
        <strong>RAG Response:</strong><br><br>
        "According to TechCorp's policy, they offer a <strong>60-day money-back guarantee</strong> 
        on all products. You can request a full refund by contacting support@techcorp.com. 
        Note that digital products have a shorter 14-day refund window if they haven't been downloaded."
        <br><br>
        <em style="color: #666;">✅ Specific, accurate answer based on actual documents!</em>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Real-world use cases
st.markdown("## 🌍 Real-World Use Cases")

use_cases = [
    {
        "icon": "🏢",
        "title": "Enterprise Knowledge Base",
        "desc": "Search internal documents, wikis, and policies"
    },
    {
        "icon": "⚖️",
        "title": "Legal Document Analysis",
        "desc": "Query contracts, regulations, and case law"
    },
    {
        "icon": "🏥",
        "title": "Healthcare Assistant",
        "desc": "Access medical records and clinical guidelines"
    },
    {
        "icon": "📚",
        "title": "Educational Tutor",
        "desc": "Answer questions from textbooks and course materials"
    },
    {
        "icon": "🛒",
        "title": "E-commerce Support",
        "desc": "Product information, FAQs, and order policies"
    },
    {
        "icon": "💻",
        "title": "Code Documentation",
        "desc": "Query codebases, APIs, and technical docs"
    }
]

cols = st.columns(3)
for i, case in enumerate(use_cases):
    with cols[i % 3]:
        st.markdown(f"""
        <div style="background: #f5f5f5; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;">
            <span style="font-size: 2rem;">{case['icon']}</span>
            <h4>{case['title']}</h4>
            <p style="color: #666; font-size: 0.9rem;">{case['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Key takeaways
st.markdown("## 📌 Key Takeaways")

st.success("""
1. **RAG = Retrieval + Generation** — It combines search with language models
2. **Solves LLM limitations** — No more knowledge cutoff or hallucinations about your data
3. **Your data stays yours** — Documents are stored in your vector database, not sent to train models
4. **Simple pipeline** — Query → Retrieve → Augment → Generate
""")

# Navigation
st.markdown("---")
st.markdown("### ➡️ Next: Understanding Embeddings")
st.markdown("Learn how text is converted to vectors that capture semantic meaning.")
