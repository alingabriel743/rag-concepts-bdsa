"""
Page 2: Understanding Embeddings
================================
Learn how text becomes vectors and explore semantic similarity.
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Understanding Embeddings",
    page_icon="🔢",
    layout="wide"
)

# Title
st.markdown("# 🔢 Understanding Embeddings")
st.markdown("*How Text Becomes Numbers That Capture Meaning*")
st.markdown("---")

# What are embeddings
st.markdown("## 🧠 What Are Embeddings?")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    **Embeddings** are numerical representations of text (or other data) as vectors 
    in a high-dimensional space.
    
    Think of it like coordinates on a map, but instead of 2D (x, y), embeddings 
    typically have **hundreds of dimensions** (e.g., 384 or 768).
    
    **The key insight:** Similar concepts end up close together in this space!
    
    - "dog" and "puppy" → close together
    - "dog" and "quantum physics" → far apart
    """)

with col2:
    st.info("""
    **In simple terms:**
    
    Embeddings convert text to numbers in a way that preserves meaning.
    
    *Similar text = Similar numbers*
    """)

# Visual explanation
st.markdown("### 📊 From Words to Vectors")

st.markdown("""
```
"The cat sat on the mat"
        ↓
   [Embedding Model]
        ↓
[0.23, -0.15, 0.89, 0.02, ..., -0.34]  ← 384 numbers!
```
""")

st.markdown("---")

# Why embeddings matter
st.markdown("## 🎯 Why Embeddings Matter for RAG")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 1.5rem; border-radius: 10px;">
        <h4>🔍 Semantic Search</h4>
        <p>Find documents by meaning, not just keywords</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">
            "automobile" matches "car" even without exact keyword match
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                color: white; padding: 1.5rem; border-radius: 10px;">
        <h4>📏 Similarity Measurement</h4>
        <p>Compare texts mathematically</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">
            Cosine similarity = 0.95 means very similar
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                color: white; padding: 1.5rem; border-radius: 10px;">
        <h4>⚡ Efficient Retrieval</h4>
        <p>Vector databases enable fast similarity search</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">
            Search millions of vectors in milliseconds
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Interactive embedding explorer
st.markdown("## 🎮 Interactive: Explore Embeddings")

# Try to import the embedding utilities
try:
    from utils.embeddings import get_embeddings, cosine_similarity, reduce_dimensions, EMBEDDING_MODELS
    from utils.visualization import plot_embeddings_2d, plot_similarity_matrix, plot_vector_comparison
    embeddings_available = True
except ImportError as e:
    embeddings_available = False
    st.warning(f"""
    ⚠️ **Embedding libraries not installed yet.**
    
    Run this command to install:
    ```bash
    pip install sentence-transformers scikit-learn
    ```
    
    Error: {e}
    """)

if embeddings_available:
    # Model selection
    st.markdown("### 🔧 Select Embedding Model")
    
    model_col1, model_col2 = st.columns([1, 2])
    
    with model_col1:
        selected_model = st.selectbox(
            "Choose a model:",
            options=list(EMBEDDING_MODELS.keys()),
            format_func=lambda x: f"{EMBEDDING_MODELS[x]['name']} ({EMBEDDING_MODELS[x]['dimension']}d)"
        )
    
    with model_col2:
        model_info = EMBEDDING_MODELS[selected_model]
        st.markdown(f"""
        **{model_info['name']}**
        - Dimension: {model_info['dimension']}
        - Speed: {model_info['speed']}
        - Quality: {model_info['quality']}
        - {model_info['description']}
        """)
    
    st.markdown("---")
    
    # Text to embedding demo
    st.markdown("### 📝 Text to Embedding")
    
    demo_texts = st.text_area(
        "Enter texts to embed (one per line):",
        value="I love machine learning\nArtificial intelligence is fascinating\nThe weather is nice today\nIt's sunny outside\nDeep learning models are powerful",
        height=150
    )
    
    texts = [t.strip() for t in demo_texts.split("\n") if t.strip()]
    
    if st.button("🔢 Generate Embeddings", type="primary"):
        if texts:
            with st.spinner("Generating embeddings..."):
                embeddings = get_embeddings(texts, selected_model)
                
                st.success(f"Generated embeddings for {len(texts)} texts!")
                
                # Show embedding dimension
                st.markdown(f"**Embedding dimension:** {embeddings.shape[1]}")
                
                # Store in session state for other sections
                st.session_state['demo_embeddings'] = embeddings
                st.session_state['demo_texts'] = texts
                
                # Show first embedding preview
                st.markdown("**First embedding preview** (first 10 values):")
                st.code(f"{texts[0]}\n→ [{', '.join([f'{v:.4f}' for v in embeddings[0][:10]])}...]")
        else:
            st.error("Please enter at least one text.")
    
    # Similarity section
    if 'demo_embeddings' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Similarity Analysis")
        
        embeddings = st.session_state['demo_embeddings']
        texts = st.session_state['demo_texts']
        
        # Calculate similarity matrix
        n = len(texts)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sim_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])
        
        # Similarity heatmap
        labels = [t[:30] + "..." if len(t) > 30 else t for t in texts]
        fig = plot_similarity_matrix(sim_matrix, labels)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **How to read this:**
        - Diagonal is always 1.0 (text compared to itself)
        - Higher values (green) = more similar
        - Lower values (red) = less similar
        """)
        
        # 2D visualization
        st.markdown("---")
        st.markdown("### 🗺️ 2D Visualization")
        
        st.markdown("Embeddings projected to 2D using PCA (Principal Component Analysis):")
        
        if len(texts) >= 2:
            reduced = reduce_dimensions(embeddings, n_components=2, method="pca")
            fig = plot_embeddings_2d(reduced, labels)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **What you're seeing:**
            - Each point represents a text
            - Similar texts cluster together
            - Distance ≈ semantic difference
            """)

else:
    st.markdown("### 📝 Demo (Simulation)")
    st.markdown("Install the dependencies above to see live embeddings.")
    
    # Show simulated example
    st.markdown("""
    Here's what the output would look like:
    
    ```
    "I love machine learning"
    → [0.234, -0.156, 0.892, 0.021, -0.445, 0.667, ...]
    
    "Artificial intelligence is fascinating"  
    → [0.256, -0.134, 0.845, 0.056, -0.423, 0.689, ...]
    
    Similarity: 0.92 (very similar topics!)
    ```
    """)

st.markdown("---")

# Distance metrics explanation
st.markdown("## 📏 Distance Metrics")

st.markdown("How do we measure similarity between vectors?")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Cosine Similarity
    
    Measures the angle between vectors.
    
    - Range: -1 to 1
    - 1 = identical direction
    - 0 = orthogonal
    - -1 = opposite
    
    **Best for:** Text similarity
    
    ```python
    similarity = dot(A, B) / (|A| * |B|)
    ```
    """)

with col2:
    st.markdown("""
    ### Euclidean Distance (L2)
    
    Straight-line distance between points.
    
    - Range: 0 to ∞
    - 0 = identical
    - Higher = more different
    
    **Best for:** When magnitude matters
    
    ```python
    distance = sqrt(sum((A - B)²))
    ```
    """)

with col3:
    st.markdown("""
    ### Dot Product
    
    Simple multiplication and sum.
    
    - Range: -∞ to ∞
    - Higher = more similar
    
    **Best for:** Fast computation
    
    ```python
    similarity = sum(A * B)
    ```
    """)

st.markdown("---")

# Popular embedding models
st.markdown("## 🏆 Popular Embedding Models")

models_data = [
    {"Model": "all-MiniLM-L6-v2", "Provider": "Sentence-Transformers", "Dim": 384, "Speed": "⚡⚡⚡", "Quality": "⭐⭐⭐"},
    {"Model": "all-mpnet-base-v2", "Provider": "Sentence-Transformers", "Dim": 768, "Speed": "⚡⚡", "Quality": "⭐⭐⭐⭐⭐"},
    {"Model": "text-embedding-ada-002", "Provider": "OpenAI", "Dim": 1536, "Speed": "⚡⚡", "Quality": "⭐⭐⭐⭐⭐"},
    {"Model": "text-embedding-3-small", "Provider": "OpenAI", "Dim": 1536, "Speed": "⚡⚡⚡", "Quality": "⭐⭐⭐⭐"},
    {"Model": "embed-english-v3.0", "Provider": "Cohere", "Dim": 1024, "Speed": "⚡⚡", "Quality": "⭐⭐⭐⭐⭐"},
]

import pandas as pd
df = pd.DataFrame(models_data)
st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")

# Key takeaways
st.markdown("## 📌 Key Takeaways")

st.success("""
1. **Embeddings = vectors that capture meaning** — Similar text → similar vectors
2. **High-dimensional** — Usually 384 to 1536 dimensions
3. **Enable semantic search** — Find by meaning, not just keywords
4. **Cosine similarity** — Most common metric for text similarity
5. **Model choice matters** — Trade-off between speed and quality
""")

# Navigation
st.markdown("---")
st.markdown("### ➡️ Next: Vector Databases")
st.markdown("Learn how to efficiently store and search embeddings with ChromaDB.")
