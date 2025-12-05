"""
Page 3: Vector Databases
========================
Learn about vector storage, indexing, and hands-on ChromaDB demos.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Vector Databases",
    page_icon="🗄️",
    layout="wide"
)

# Title
st.markdown("# 🗄️ Vector Databases")
st.markdown("*Storing and Searching Embeddings at Scale*")
st.markdown("---")

# What is a Vector DB
st.markdown("## 🤔 What is a Vector Database?")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    A **Vector Database** is a specialized database designed to store, index, and 
    search high-dimensional vectors (embeddings).
    
    Unlike traditional databases that use exact matching:
    - SQL: `WHERE name = 'John'`
    
    Vector databases find **similar** items:
    - Vector: "Find the 5 most similar documents to this query"
    
    This enables **semantic search** — finding information by meaning, 
    not just exact keywords.
    """)

with col2:
    st.info("""
    **Traditional DB vs Vector DB:**
    
    🔎 SQL: *"Find rows where city = 'Paris'"*
    
    🧠 Vector: *"Find items similar to 'vacation in France'"*
    """)

st.markdown("---")

# Why not regular databases
st.markdown("## ❓ Why Not Just Use a Regular Database?")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🐌 The Problem")
    st.markdown("""
    Storing vectors in a regular database (PostgreSQL, MySQL) and doing 
    similarity search would require:
    
    1. **Loading ALL vectors** into memory
    2. **Comparing your query** against EVERY vector
    3. **Sorting** to find the closest ones
    
    With 1 million documents, this means 1 million comparisons per query!
    
    **Time complexity: O(n)** — doesn't scale!
    """)

with col2:
    st.markdown("### 🚀 The Solution")
    st.markdown("""
    Vector databases use special **indexing algorithms**:
    
    - **HNSW** (Hierarchical Navigable Small World)
    - **IVF** (Inverted File Index)
    - **Product Quantization**
    
    These create smart shortcuts to find similar vectors without 
    checking all of them.
    
    **Time complexity: O(log n)** — scales beautifully!
    """)

st.markdown("---")

# ChromaDB Section
st.markdown("## 🎨 Introducing ChromaDB")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **ChromaDB** is an open-source embedding database that's perfect for RAG:
    
    ✅ **Simple API** — Get started in minutes  
    ✅ **In-memory or Persistent** — Choose your storage mode  
    ✅ **Built-in Embedding Functions** — Or bring your own  
    ✅ **Metadata Filtering** — Filter search results by properties  
    ✅ **No External Dependencies** — Runs locally, no servers needed
    """)

with col2:
    st.code("""
pip install chromadb
    """, language="bash")
    
    st.markdown("That's it! You're ready to go.")

st.markdown("---")

# Comparison table
st.markdown("## 📊 Vector Database Comparison")

import pandas as pd

comparison_data = [
    {"Database": "ChromaDB", "Type": "Open Source", "Hosting": "Local/Cloud", "Best For": "Prototyping, Small-Medium", "Complexity": "⭐"},
    {"Database": "Pinecone", "Type": "Managed", "Hosting": "Cloud Only", "Best For": "Production, Enterprise", "Complexity": "⭐⭐"},
    {"Database": "Weaviate", "Type": "Open Source", "Hosting": "Local/Cloud", "Best For": "ML Features, GraphQL", "Complexity": "⭐⭐⭐"},
    {"Database": "FAISS", "Type": "Library", "Hosting": "Local Only", "Best For": "Research, Max Speed", "Complexity": "⭐⭐⭐"},
    {"Database": "Qdrant", "Type": "Open Source", "Hosting": "Local/Cloud", "Best For": "Production, Filtering", "Complexity": "⭐⭐"},
    {"Database": "Milvus", "Type": "Open Source", "Hosting": "Local/Cloud", "Best For": "Large Scale, Enterprise", "Complexity": "⭐⭐⭐⭐"},
]

df = pd.DataFrame(comparison_data)
st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")

# Interactive ChromaDB Demo
st.markdown("## 🎮 Interactive: ChromaDB Playground")

# Try to import ChromaDB
try:
    import chromadb
    from utils.vector_store import (
        get_chroma_client, create_collection, add_documents, 
        query_collection, get_collection_stats, delete_collection
    )
    from utils.embeddings import get_embeddings
    chroma_available = True
except ImportError as e:
    chroma_available = False
    st.warning(f"""
    ⚠️ **ChromaDB not installed yet.**
    
    Run this command to install:
    ```bash
    pip install -r requirements.txt
    ```
    
    Error: {e}
    """)

if chroma_available:
    # Initialize client
    if 'chroma_client' not in st.session_state:
        st.session_state['chroma_client'] = chromadb.Client()
    
    client = st.session_state['chroma_client']
    
    # Tabs for different operations
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Create Collection", "➕ Add Documents", "🔍 Search", "📊 View Stats"])
    
    with tab1:
        st.markdown("### Create a Collection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            collection_name = st.text_input("Collection name:", value="my_documents", key="create_collection_name")
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🆕 Create Collection", type="primary"):
                try:
                    collection = client.get_or_create_collection(name=collection_name)
                    st.session_state['active_collection'] = collection_name
                    st.success(f"✅ Collection '{collection_name}' created/loaded!")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Show code
        with st.expander("📝 See the code"):
            st.code("""
import chromadb

# Create a client
client = chromadb.Client()

# Create a collection
collection = client.get_or_create_collection(
    name="my_documents"
)
            """, language="python")
    
    with tab2:
        st.markdown("### Add Documents")
        
        if 'active_collection' not in st.session_state:
            st.warning("⚠️ Please create a collection first!")
        else:
            collection = client.get_collection(st.session_state['active_collection'])
            
            documents_input = st.text_area(
                "Enter documents (one per line):",
                value="Python is a great programming language.\nJavaScript is used for web development.\nMachine learning requires lots of data.\nDeep learning uses neural networks.\nThe weather in Paris is lovely in spring.",
                height=150
            )
            
            if st.button("➕ Add Documents", type="primary"):
                docs = [d.strip() for d in documents_input.split("\n") if d.strip()]
                
                if docs:
                    with st.spinner("Generating embeddings and adding documents..."):
                        embeddings = get_embeddings(docs)
                        ids = [f"doc_{i}" for i in range(len(docs))]
                        
                        collection.add(
                            documents=docs,
                            embeddings=embeddings.tolist(),
                            ids=ids
                        )
                        
                        st.success(f"✅ Added {len(docs)} documents!")
                        st.session_state['docs_added'] = True
        
        # Show code
        with st.expander("📝 See the code"):
            st.code("""
# Add documents to collection
collection.add(
    documents=["doc1", "doc2", "doc3"],
    embeddings=[[0.1, 0.2, ...], ...],  # Optional if using default embedder
    metadatas=[{"source": "web"}, ...],  # Optional metadata
    ids=["id1", "id2", "id3"]
)
            """, language="python")
    
    with tab3:
        st.markdown("### Search Documents")
        
        if 'active_collection' not in st.session_state:
            st.warning("⚠️ Please create a collection first!")
        elif 'docs_added' not in st.session_state:
            st.warning("⚠️ Please add some documents first!")
        else:
            collection = client.get_collection(st.session_state['active_collection'])
            
            query = st.text_input("Enter your search query:", value="programming languages")
            n_results = st.slider("Number of results:", min_value=1, max_value=10, value=3)
            
            if st.button("🔍 Search", type="primary"):
                with st.spinner("Searching..."):
                    query_embedding = get_embeddings(query)
                    
                    results = collection.query(
                        query_embeddings=query_embedding.tolist(),
                        n_results=n_results,
                        include=["documents", "distances"]
                    )
                    
                    st.markdown("### Results:")
                    
                    for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
                        similarity = 1 / (1 + dist)  # Convert distance to similarity
                        st.markdown(f"""
                        <div style="background: #f5f5f5; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
                                    border-left: 4px solid {'#4caf50' if similarity > 0.5 else '#ff9800'};">
                            <strong>#{i+1}</strong> (Similarity: {similarity:.3f})<br>
                            {doc}
                        </div>
                        """, unsafe_allow_html=True)
        
        # Show code
        with st.expander("📝 See the code"):
            st.code("""
# Query the collection
results = collection.query(
    query_texts=["programming languages"],  # or query_embeddings
    n_results=3,
    include=["documents", "distances", "metadatas"]
)

# Results structure:
# {
#     'ids': [['id1', 'id2', 'id3']],
#     'documents': [['doc1', 'doc2', 'doc3']],
#     'distances': [[0.1, 0.3, 0.6]]
# }
            """, language="python")
    
    with tab4:
        st.markdown("### Collection Statistics")
        
        # List all collections
        collections = client.list_collections()
        
        if not collections:
            st.info("No collections yet. Create one in the first tab!")
        else:
            st.markdown(f"**Total collections:** {len(collections)}")
            
            for coll in collections:
                count = coll.count()
                st.markdown(f"""
                <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <strong>📁 {coll.name}</strong><br>
                    Documents: {count}
                </div>
                """, unsafe_allow_html=True)
            
            # Delete option
            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            with col1:
                coll_to_delete = st.selectbox(
                    "Select collection to delete:",
                    options=[c.name for c in collections]
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️ Delete", type="secondary"):
                    client.delete_collection(coll_to_delete)
                    if 'active_collection' in st.session_state and st.session_state['active_collection'] == coll_to_delete:
                        del st.session_state['active_collection']
                    if 'docs_added' in st.session_state:
                        del st.session_state['docs_added']
                    st.success(f"Deleted collection '{coll_to_delete}'")
                    st.rerun()

else:
    st.markdown("### 📝 Code Preview")
    st.markdown("Install ChromaDB to try the interactive demo! Here's what the code looks like:")
    
    st.code("""
import chromadb

# Create client (in-memory)
client = chromadb.Client()

# Or persistent
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.get_or_create_collection("my_docs")

# Add documents
collection.add(
    documents=["Hello world", "Goodbye world"],
    ids=["id1", "id2"]
)

# Query
results = collection.query(
    query_texts=["greeting"],
    n_results=1
)
print(results['documents'])  # [['Hello world']]
    """, language="python")

st.markdown("---")

# Indexing algorithms
st.markdown("## 🧮 How Vector Indexing Works")

st.markdown("""
Vector databases use clever algorithms to avoid comparing against every vector:
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### HNSW (Hierarchical Navigable Small World)
    
    Creates a multi-layer graph structure:
    
    1. **Top layer:** Sparse, long-distance connections
    2. **Middle layers:** More nodes, medium connections
    3. **Bottom layer:** All nodes, local connections
    
    Search starts at the top and "zooms in" quickly!
    
    **Used by:** ChromaDB, Pinecone, Qdrant
    """)

with col2:
    st.markdown("""
    ### IVF (Inverted File Index)
    
    Divides vector space into regions:
    
    1. **Cluster** vectors into groups (centroids)
    2. **Store** each vector with its cluster
    3. **Search** only in nearby clusters
    
    Like looking only in relevant sections of a library!
    
    **Used by:** FAISS, Milvus
    """)

st.markdown("---")

# Key takeaways
st.markdown("## 📌 Key Takeaways")

st.success("""
1. **Vector DBs are specialized** — Optimized for similarity search, not exact match
2. **Indexing enables scale** — HNSW, IVF algorithms make search fast
3. **ChromaDB is beginner-friendly** — Simple API, no server setup needed
4. **Metadata filtering** — Combine semantic search with traditional filters
5. **Storage options** — In-memory for dev, persistent for production
""")

# Navigation
st.markdown("---")
st.markdown("### ➡️ Next: Document Processing")
st.markdown("Learn how to chunk and prepare documents for RAG.")
