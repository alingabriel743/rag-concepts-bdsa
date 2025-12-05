"""
Page 7: Complete RAG Pipeline
=============================
End-to-end interactive RAG demonstration.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Complete RAG Pipeline",
    page_icon="🔗",
    layout="wide"
)

# Title
st.markdown("# 🔗 Complete RAG Pipeline")
st.markdown("*Putting It All Together: End-to-End Demo*")
st.markdown("---")

# Pipeline overview
st.markdown("## 🗺️ The Complete Pipeline")

st.markdown("""
You've learned all the pieces. Now let's see them work together!
""")

# Visual pipeline
cols = st.columns(5)
steps = [
    ("📄", "1. Load", "Documents"),
    ("✂️", "2. Chunk", "Split text"),
    ("🔢", "3. Embed", "Vectorize"),
    ("🗄️", "4. Store", "ChromaDB"),
    ("🔍", "5. Query", "Retrieve & Generate")
]

for col, (icon, title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 1rem; border-radius: 10px; height: 120px;">
            <span style="font-size: 2rem;">{icon}</span>
            <h4 style="margin: 0.5rem 0 0 0;">{title}</h4>
            <p style="margin: 0; font-size: 0.85rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Check dependencies
try:
    from utils.embeddings import get_embeddings
    from utils.chunking import chunk_by_sentences, get_chunk_stats
    from utils.visualization import plot_chunk_distribution, plot_retrieval_scores
    import chromadb
    import numpy as np
    deps_available = True
except ImportError as e:
    deps_available = False
    st.error(f"""
    **Dependencies not installed.** Run:
    ```bash
    pip install chromadb sentence-transformers scikit-learn plotly
    ```
    Error: {e}
    """)

if deps_available:
    # Initialize session state
    if 'rag_pipeline_state' not in st.session_state:
        st.session_state['rag_pipeline_state'] = {
            'step': 1,
            'documents': None,
            'chunks': None,
            'collection': None,
            'client': None
        }
    
    state = st.session_state['rag_pipeline_state']
    
    # Progress indicator
    progress = (state['step'] - 1) / 5
    st.progress(progress)
    st.markdown(f"**Current Step: {state['step']}/5**")
    
    st.markdown("---")
    
    # Step 1: Load Documents
    st.markdown("## 📄 Step 1: Load Documents")
    
    with st.expander("📄 Document Input", expanded=(state['step'] == 1)):
        input_method = st.radio(
            "Choose input method:",
            ["Use sample documents", "Enter custom text"],
            horizontal=True
        )
        
        if input_method == "Use sample documents":
            sample_docs = """Artificial Intelligence (AI) is transforming industries worldwide. Machine learning, a subset of AI, enables systems to learn from data and improve over time without explicit programming.

Deep learning uses neural networks with multiple layers to process complex patterns. These networks are inspired by the human brain's structure and can handle tasks like image recognition and natural language processing.

Natural Language Processing (NLP) focuses on the interaction between computers and human language. Applications include chatbots, language translation, sentiment analysis, and text summarization.

Large Language Models (LLMs) like GPT-4 and Claude are trained on vast amounts of text data. They can generate human-like text, answer questions, write code, and assist with various tasks.

Retrieval-Augmented Generation (RAG) combines the power of LLMs with external knowledge retrieval. This approach reduces hallucinations and allows models to access up-to-date information.

Vector databases store embeddings and enable semantic search. Popular options include ChromaDB, Pinecone, and Weaviate. They use algorithms like HNSW for fast similarity search.

Prompt engineering is the art of crafting effective prompts for LLMs. Good prompts include clear instructions, context, and examples. This skill is essential for getting the best results from AI models.

Fine-tuning allows you to adapt pre-trained models to specific tasks or domains. This is more efficient than training from scratch and can significantly improve performance on specialized tasks."""
            
            st.text_area("Sample Documents:", value=sample_docs, height=200, disabled=True)
            doc_text = sample_docs
        else:
            doc_text = st.text_area(
                "Enter your documents:",
                placeholder="Paste your text here...",
                height=200
            )
        
        if st.button("✅ Load Documents", type="primary", disabled=(not doc_text)):
            state['documents'] = doc_text
            state['step'] = max(state['step'], 2)
            st.success(f"✅ Loaded {len(doc_text)} characters!")
            st.rerun()
    
    # Step 2: Chunk Documents
    if state['step'] >= 2:
        st.markdown("## ✂️ Step 2: Chunk Documents")
        
        with st.expander("✂️ Chunking Settings", expanded=(state['step'] == 2)):
            col1, col2 = st.columns(2)
            
            with col1:
                sentences_per_chunk = st.slider("Sentences per chunk:", 2, 8, 3)
                overlap = st.slider("Overlap (sentences):", 0, 3, 1)
            
            with col2:
                st.markdown("**Preview:**")
                if state['documents']:
                    preview_chunks = chunk_by_sentences(
                        state['documents'], 
                        sentences_per_chunk, 
                        overlap
                    )
                    st.markdown(f"Will create ~{len(preview_chunks)} chunks")
            
            if st.button("✂️ Chunk Documents", type="primary"):
                chunks = chunk_by_sentences(state['documents'], sentences_per_chunk, overlap)
                state['chunks'] = chunks
                state['step'] = max(state['step'], 3)
                st.success(f"✅ Created {len(chunks)} chunks!")
                
                # Show stats
                stats = get_chunk_stats(chunks)
                col1, col2, col3 = st.columns(3)
                col1.metric("Chunks", stats['count'])
                col2.metric("Avg Length", f"{stats['avg_length']:.0f}")
                col3.metric("Total Chars", stats['total_length'])
                
                # Show chunks
                with st.expander("View chunks"):
                    for i, chunk in enumerate(chunks):
                        st.markdown(f"**Chunk {i+1}:** {chunk[:100]}...")
                
                st.rerun()
    
    # Step 3: Generate Embeddings & Store
    if state['step'] >= 3:
        st.markdown("## 🔢🗄️ Step 3 & 4: Embed & Store in Vector DB")
        
        with st.expander("🔢 Embedding & Storage", expanded=(state['step'] == 3)):
            st.markdown(f"**Chunks to embed:** {len(state['chunks']) if state['chunks'] else 0}")
            
            if st.button("🚀 Embed & Store in ChromaDB", type="primary"):
                with st.spinner("Generating embeddings..."):
                    embeddings = get_embeddings(state['chunks'])
                    st.success(f"✅ Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
                
                with st.spinner("Storing in ChromaDB..."):
                    # Create new client and collection for this demo
                    client = chromadb.Client()
                    collection = client.get_or_create_collection("rag_demo")
                    
                    # Clear existing data
                    try:
                        existing = collection.get()
                        if existing['ids']:
                            collection.delete(ids=existing['ids'])
                    except:
                        pass
                    
                    # Add documents
                    collection.add(
                        documents=state['chunks'],
                        embeddings=embeddings.tolist(),
                        ids=[f"chunk_{i}" for i in range(len(state['chunks']))],
                        metadatas=[{"chunk_index": i} for i in range(len(state['chunks']))]
                    )
                    
                    state['client'] = client
                    state['collection'] = collection
                    state['step'] = 5
                    
                    st.success(f"✅ Stored {collection.count()} chunks in ChromaDB!")
                    st.rerun()
    
    # Step 5: Query
    if state['step'] >= 5 and state['collection'] is not None:
        st.markdown("## 🔍 Step 5: Query the RAG System")
        
        st.success("🎉 **Pipeline ready!** Ask questions about the documents.")
        
        # Groq API Key input
        st.markdown("### 🔑 LLM Configuration")
        
        groq_api_key = st.text_input(
            "Enter your Groq API Key:",
            type="password",
            help="Get your free API key at https://console.groq.com/keys"
        )
        
        if not groq_api_key:
            st.info("👆 Enter your Groq API key to enable LLM generation. Get a free key at [console.groq.com](https://console.groq.com/keys)")
        
        # Query interface
        st.markdown("### 💬 Ask a Question")
        
        query = st.text_input(
            "Ask a question:",
            value="What is RAG and how does it work?",
            key="rag_query"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            n_results = st.slider("Top-k results:", 1, 5, 3)
        with col2:
            temperature = st.slider("Temperature:", 0.0, 1.0, 0.3, 0.1)
        
        if st.button("🔍 Search & Generate", type="primary"):
            with st.spinner("Searching..."):
                # Generate query embedding
                query_embedding = get_embeddings(query)
                
                # Search
                results = state['collection'].query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=n_results,
                    include=["documents", "distances", "metadatas"]
                )
                
                st.markdown("---")
                
                # Show retrieval results
                st.markdown("### 📚 Retrieved Context")
                
                docs = results['documents'][0]
                distances = results['distances'][0]
                
                context_parts = []
                for i, (doc, dist) in enumerate(zip(docs, distances)):
                    similarity = 1 / (1 + dist)
                    context_parts.append(f"[Source {i+1}] {doc}")
                    
                    st.markdown(f"""
                    <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
                                border-left: 4px solid #2196f3;">
                        <strong>Source {i+1}</strong> (Similarity: {similarity:.3f})<br>
                        {doc}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Build RAG prompt
                context = "\n\n".join(context_parts)
                
                rag_prompt = f"""Answer the question based on the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:"""
                
                st.markdown("---")
                st.markdown("### 📝 Generated RAG Prompt")
                
                with st.expander("View full prompt"):
                    st.code(rag_prompt, language="text")
                
                # Generate response with Groq
                st.markdown("### 🤖 Response")
                
                if groq_api_key:
                    try:
                        from utils.llm import get_groq_client, generate_rag_response
                        
                        with st.spinner("Generating response with Llama 3.3..."):
                            client = get_groq_client(groq_api_key)
                            response = generate_rag_response(
                                client=client,
                                question=query,
                                context_chunks=docs,
                                model="llama-3.3-70b-versatile",
                                temperature=temperature
                            )
                        
                        st.markdown(f"""
                        <div style="background: #e8f5e9; padding: 1.5rem; border-radius: 10px; 
                                    border-left: 4px solid #4caf50;">
                            <strong>🦙 Llama 3.3 70B Response:</strong><br><br>
                            {response}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.caption("Powered by Groq + Llama 3.3 70B Versatile")
                        
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
                else:
                    st.warning("""
                    **No API key provided.** Enter your Groq API key above to generate actual responses.
                    
                    Without an API key, here's what the response would look like based on the context:
                    
                    > Based on the retrieved documents, RAG (Retrieval-Augmented Generation) combines the power 
                    > of Large Language Models with external knowledge retrieval [Source 1]. This approach reduces 
                    > hallucinations and allows models to access up-to-date information [Source 2].
                    """)
        
        # Reset button
        st.markdown("---")
        if st.button("🔄 Reset Pipeline"):
            st.session_state['rag_pipeline_state'] = {
                'step': 1,
                'documents': None,
                'chunks': None,
                'collection': None,
                'client': None
            }
            st.rerun()

else:
    st.markdown("## 📝 Pipeline Overview (Demo Mode)")
    
    st.markdown("""
    Without the dependencies, here's a code walkthrough of the complete pipeline:
    """)
    
    st.code("""
# Complete RAG Pipeline Example

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 1. Initialize components
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("my_docs")
openai_client = OpenAI()

# 2. Load and chunk documents
documents = load_documents("./docs/")
chunks = chunk_documents(documents, chunk_size=500, overlap=50)

# 3. Generate embeddings
embeddings = embedder.encode(chunks)

# 4. Store in vector database
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

# 5. Query function
def query_rag(question):
    # Embed question
    q_embedding = embedder.encode([question])
    
    # Search
    results = collection.query(query_embeddings=q_embedding, n_results=5)
    context = "\\n\\n".join(results['documents'][0])
    
    # Generate response
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Answer based only on the provided context."},
            {"role": "user", "content": f"Context:\\n{context}\\n\\nQuestion: {question}"}
        ]
    )
    
    return response.choices[0].message.content

# Use it!
answer = query_rag("What is machine learning?")
print(answer)
    """, language="python")

st.markdown("---")

# Key takeaways
st.markdown("## 📌 Key Takeaways")

st.success("""
1. **5 Steps:** Load → Chunk → Embed → Store → Query
2. **Each step matters** — Quality at each stage affects final results
3. **Vector DB is the backbone** — Enables semantic search at scale
4. **Prompt formatting** — Clearly separate context from question
5. **Iterate and improve** — Tune chunking, retrieval, and prompts
""")

# Navigation
st.markdown("---")
st.markdown("### ➡️ Next: Best Practices")
st.markdown("Learn tips, tricks, and common pitfalls to avoid!")
