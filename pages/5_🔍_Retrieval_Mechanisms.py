"""
Page 5: Retrieval Mechanisms
============================
Learn about similarity search, distance metrics, and retrieval strategies.
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Retrieval Mechanisms",
    page_icon="🔍",
    layout="wide"
)

# Title
st.markdown("# 🔍 Retrieval Mechanisms")
st.markdown("*Finding the Most Relevant Information*")
st.markdown("---")

# What is retrieval
st.markdown("## 🎯 What is Retrieval in RAG?")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    **Retrieval** is the process of finding the most relevant documents or chunks 
    from your knowledge base given a user query.
    
    It's the "R" in RAG and arguably the most critical step:
    
    - **Good retrieval** → LLM gets relevant context → Great answer
    - **Bad retrieval** → LLM gets irrelevant/wrong context → Poor answer
    
    The quality of your RAG system is largely determined by retrieval quality!
    """)

with col2:
    st.info("""
    **Retrieval Pipeline:**
    
    1. User query → Embed
    2. Search vector DB
    3. Get top-k results
    4. (Optional) Re-rank
    5. Return to LLM
    """)

st.markdown("---")

# Types of retrieval
st.markdown("## 📊 Types of Retrieval")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 1.5rem; border-radius: 10px; height: 280px;">
        <h4>🧠 Semantic Search</h4>
        <p><strong>How:</strong> Embed query, find similar vectors</p>
        <p><strong>Pros:</strong></p>
        <ul>
            <li>Understands meaning</li>
            <li>Handles synonyms</li>
            <li>Works across languages</li>
        </ul>
        <p><strong>Cons:</strong> May miss exact terms</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                color: white; padding: 1.5rem; border-radius: 10px; height: 280px;">
        <h4>🔤 Keyword Search</h4>
        <p><strong>How:</strong> BM25, TF-IDF matching</p>
        <p><strong>Pros:</strong></p>
        <ul>
            <li>Exact term matching</li>
            <li>Fast and efficient</li>
            <li>Well understood</li>
        </ul>
        <p><strong>Cons:</strong> No semantic understanding</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                color: white; padding: 1.5rem; border-radius: 10px; height: 280px;">
        <h4>🔀 Hybrid Search</h4>
        <p><strong>How:</strong> Combine semantic + keyword</p>
        <p><strong>Pros:</strong></p>
        <ul>
            <li>Best of both worlds</li>
            <li>Catches exact terms AND meaning</li>
            <li>More robust</li>
        </ul>
        <p><strong>Cons:</strong> More complex setup</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Distance metrics
st.markdown("## 📏 Distance Metrics Explained")

st.markdown("""
How do we measure "similarity" between vectors? There are several approaches:
""")

tab1, tab2, tab3 = st.tabs(["Cosine Similarity", "Euclidean Distance (L2)", "Dot Product"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Cosine Similarity
        
        Measures the **angle** between two vectors, ignoring magnitude.
        
        **Formula:**
        ```
        cos(θ) = (A · B) / (|A| × |B|)
        ```
        
        **Range:** -1 to 1
        - 1 = identical direction
        - 0 = orthogonal (unrelated)
        - -1 = opposite direction
        
        **Best for:** Text similarity (most common choice!)
        
        **Why?** Document length doesn't affect similarity.
        """)
    
    with col2:
        st.markdown("**Visual Example:**")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Vector-similarity.svg/220px-Vector-similarity.svg.png", width=200)
        st.caption("Vectors A and B have high cosine similarity (small angle)")
        
        st.code("""
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example
vec_a = np.array([1, 2, 3])
vec_b = np.array([1, 2, 2.9])
print(cosine_similarity(vec_a, vec_b))  # ~0.999
        """, language="python")

with tab2:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Euclidean Distance (L2)
        
        Measures the **straight-line distance** between two points.
        
        **Formula:**
        ```
        d = √(Σ(Aᵢ - Bᵢ)²)
        ```
        
        **Range:** 0 to ∞
        - 0 = identical
        - Higher = more different
        
        **Best for:** When magnitude matters
        
        **Note:** Often converted to similarity: `1 / (1 + distance)`
        """)
    
    with col2:
        st.code("""
import numpy as np

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Or equivalently:
def euclidean_distance_v2(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Example
vec_a = np.array([1, 2, 3])
vec_b = np.array([4, 5, 6])
print(euclidean_distance(vec_a, vec_b))  # ~5.196
        """, language="python")

with tab3:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Dot Product (Inner Product)
        
        Simple multiplication and sum of corresponding elements.
        
        **Formula:**
        ```
        A · B = Σ(Aᵢ × Bᵢ)
        ```
        
        **Range:** -∞ to ∞
        - Higher = more similar (usually)
        
        **Best for:** 
        - Normalized vectors (then equals cosine)
        - When you want magnitude to matter
        
        **Note:** Fastest to compute!
        """)
    
    with col2:
        st.code("""
import numpy as np

def dot_product(a, b):
    return np.dot(a, b)

# Example
vec_a = np.array([1, 2, 3])
vec_b = np.array([4, 5, 6])
print(dot_product(vec_a, vec_b))  # 32

# With normalized vectors:
a_norm = vec_a / np.linalg.norm(vec_a)
b_norm = vec_b / np.linalg.norm(vec_b)
print(dot_product(a_norm, b_norm))  # = cosine similarity
        """, language="python")

st.markdown("---")

# Interactive retrieval demo
st.markdown("## 🎮 Interactive: Retrieval Playground")

try:
    from utils.embeddings import get_embeddings, cosine_similarity
    from utils.visualization import plot_retrieval_scores
    import chromadb
    retrieval_available = True
except ImportError:
    retrieval_available = False

if retrieval_available:
    # Sample knowledge base
    if 'retrieval_docs' not in st.session_state:
        st.session_state['retrieval_docs'] = [
            "Python is a high-level programming language known for its simplicity.",
            "Machine learning enables computers to learn from data automatically.",
            "JavaScript is primarily used for web development and browser interactions.",
            "Neural networks are inspired by the structure of the human brain.",
            "SQL is used to manage and query relational databases.",
            "Deep learning uses multiple layers of neural networks.",
            "React is a popular JavaScript library for building user interfaces.",
            "Natural language processing helps computers understand human language.",
            "TensorFlow and PyTorch are popular deep learning frameworks.",
            "Docker containers help package and deploy applications consistently.",
        ]
    
    # Initialize ChromaDB collection
    if 'retrieval_collection' not in st.session_state:
        client = chromadb.Client()
        collection = client.get_or_create_collection("retrieval_demo")
        
        docs = st.session_state['retrieval_docs']
        embeddings = get_embeddings(docs)
        
        collection.add(
            documents=docs,
            embeddings=embeddings.tolist(),
            ids=[f"doc_{i}" for i in range(len(docs))]
        )
        
        st.session_state['retrieval_collection'] = collection
        st.session_state['retrieval_embeddings'] = embeddings
    
    collection = st.session_state['retrieval_collection']
    
    st.markdown("### 📚 Knowledge Base")
    
    with st.expander("View all documents in the knowledge base"):
        for i, doc in enumerate(st.session_state['retrieval_docs']):
            st.markdown(f"{i+1}. {doc}")
    
    st.markdown("### 🔍 Search")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("Enter your query:", value="How do I learn AI?")
    
    with col2:
        k = st.slider("Top-k results:", 1, 10, 5)
    
    if st.button("🔍 Search", type="primary"):
        query_embedding = get_embeddings(query)
        
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            include=["documents", "distances"]
        )
        
        st.markdown("### 📊 Results")
        
        # Show results
        docs = results['documents'][0]
        distances = results['distances'][0]
        
        for i, (doc, dist) in enumerate(zip(docs, distances)):
            similarity = 1 / (1 + dist)
            
            # Color based on similarity
            if similarity > 0.6:
                color = "#4caf50"  # Green
                emoji = "🟢"
            elif similarity > 0.4:
                color = "#ff9800"  # Orange
                emoji = "🟡"
            else:
                color = "#f44336"  # Red
                emoji = "🔴"
            
            st.markdown(f"""
            <div style="background: #f5f5f5; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
                        border-left: 4px solid {color};">
                <strong>{emoji} Rank {i+1}</strong> | Similarity: {similarity:.3f}<br><br>
                {doc}
            </div>
            """, unsafe_allow_html=True)
        
        # Visualization
        fig = plot_retrieval_scores(docs, distances)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Install dependencies to try the interactive demo: `pip install chromadb sentence-transformers`")

st.markdown("---")

# Top-k and thresholding
st.markdown("## 🎚️ Retrieval Parameters")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Top-k Selection
    
    Return the k most similar documents.
    
    **Considerations:**
    - **k too small:** Might miss relevant info
    - **k too large:** Include irrelevant content, waste context
    
    **Typical values:** 3-10 chunks
    
    ```python
    results = collection.query(
        query_texts=["my question"],
        n_results=5  # top-5
    )
    ```
    """)

with col2:
    st.markdown("""
    ### Similarity Threshold
    
    Only return docs above a minimum similarity.
    
    **Considerations:**
    - Prevents low-quality matches
    - May return empty results
    
    **Typical values:** 0.3-0.7 (depends on use case)
    
    ```python
    results = collection.query(...)
    
    # Filter by threshold
    filtered = [
        (doc, score) 
        for doc, score in zip(results['documents'], results['distances'])
        if 1/(1+score) > 0.5  # similarity > 0.5
    ]
    ```
    """)

st.markdown("---")

# Re-ranking
st.markdown("## 🏆 Re-ranking Strategies")

st.markdown("""
Initial retrieval is fast but approximate. **Re-ranking** improves results:
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Cross-Encoder Re-ranking
    
    Use a more powerful model to re-score results.
    
    1. Get top-20 from vector search (fast)
    2. Score each with cross-encoder (accurate)
    3. Return top-5 after re-ranking
    
    **Models:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
    
    ```python
    from sentence_transformers import CrossEncoder
    
    reranker = CrossEncoder('cross-encoder/ms-marco-...')
    
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    
    # Sort by score
    ranked = sorted(zip(candidates, scores), 
                   key=lambda x: x[1], reverse=True)
    ```
    """)

with col2:
    st.markdown("""
    ### Reciprocal Rank Fusion (RRF)
    
    Combine rankings from multiple retrieval methods.
    
    1. Get rankings from semantic search
    2. Get rankings from keyword search (BM25)
    3. Combine using RRF formula
    
    ```python
    def rrf_score(rank, k=60):
        return 1 / (k + rank)
    
    def combine_rankings(semantic_ranks, keyword_ranks):
        combined = {}
        for doc, rank in semantic_ranks.items():
            combined[doc] = rrf_score(rank)
        for doc, rank in keyword_ranks.items():
            combined[doc] = combined.get(doc, 0) + rrf_score(rank)
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)
    ```
    """)

st.markdown("---")

# Key takeaways
st.markdown("## 📌 Key Takeaways")

st.success("""
1. **Retrieval quality = RAG quality** — Invest time in getting this right
2. **Cosine similarity** — Most common metric for text embedding search
3. **Hybrid search** — Combine semantic + keyword for best results
4. **Tune top-k** — Start with 5, adjust based on context window and quality
5. **Re-ranking helps** — Cross-encoders improve accuracy at slight latency cost
6. **Test with real queries** — The best strategy depends on your specific use case
""")

# Navigation
st.markdown("---")
st.markdown("### ➡️ Next: Generation with Context")
st.markdown("Learn how to craft prompts that use retrieved context effectively.")
