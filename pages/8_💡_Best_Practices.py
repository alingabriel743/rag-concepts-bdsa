"""
Page 8: Best Practices & Resources
==================================
Tips, common pitfalls, and learning resources for RAG.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Best Practices",
    page_icon="💡",
    layout="wide"
)

# Title
st.markdown("# 💡 Best Practices & Resources")
st.markdown("*Tips, Pitfalls, and Where to Go Next*")
st.markdown("---")

# Congratulations
st.balloons()

st.success("""
🎉 **Congratulations!** You've completed the RAG Walkthrough!

You now understand:
- What RAG is and why it matters
- How embeddings capture semantic meaning
- Vector database operations with ChromaDB
- Document chunking strategies
- Retrieval mechanisms and metrics
- Prompt engineering for RAG
- Building complete pipelines
""")

st.markdown("---")

# Best practices
st.markdown("## ✅ Best Practices")

tabs = st.tabs(["Chunking", "Embeddings", "Retrieval", "Generation", "Production"])

with tabs[0]:
    st.markdown("""
    ### 📄 Chunking Best Practices
    
    | Practice | Why |
    |----------|-----|
    | **Start with 200-500 tokens** | Good balance of context and precision |
    | **Use 10-20% overlap** | Prevents information loss at boundaries |
    | **Match strategy to content** | Code → line-based; Docs → paragraph-based |
    | **Preserve metadata** | Source, page, date helps with filtering |
    | **Consider semantic chunking** | For high-quality, production systems |
    
    ```python
    # Good chunking config
    config = {
        "chunk_size": 400,      # tokens
        "chunk_overlap": 50,    # tokens
        "strategy": "recursive",
        "metadata": True
    }
    ```
    """)

with tabs[1]:
    st.markdown("""
    ### 🔢 Embedding Best Practices
    
    | Practice | Why |
    |----------|-----|
    | **Use the same model** | Query and docs must use identical embeddings |
    | **Choose dimension wisely** | 384 for speed, 768+ for quality |
    | **Cache embeddings** | Don't re-embed unchanged documents |
    | **Batch process** | Much faster than one-by-one |
    | **Normalize if using dot product** | Ensures consistent similarity |
    
    **Model Selection Guide:**
    
    - **Prototyping:** `all-MiniLM-L6-v2` (fast, 384d)
    - **Production:** `all-mpnet-base-v2` (quality, 768d)
    - **API:** `text-embedding-ada-002` (OpenAI)
    - **Multilingual:** `paraphrase-multilingual-MiniLM-L12-v2`
    """)

with tabs[2]:
    st.markdown("""
    ### 🔍 Retrieval Best Practices
    
    | Practice | Why |
    |----------|-----|
    | **Start with top-5** | Usually sufficient, adjust based on testing |
    | **Use hybrid search** | Combines semantic + keyword for robustness |
    | **Set similarity threshold** | Filter out low-confidence results |
    | **Consider re-ranking** | Cross-encoders improve accuracy |
    | **Test with real queries** | Synthetic tests miss edge cases |
    
    **Decision Tree:**
    ```
    Need exact matching? → Add keyword search (BM25)
    Results seem random? → Lower top-k, increase threshold
    Missing relevant docs? → Increase top-k, try hybrid
    Latency too high? → Remove re-ranking, reduce k
    ```
    """)

with tabs[3]:
    st.markdown("""
    ### 🤖 Generation Best Practices
    
    | Practice | Why |
    |----------|-----|
    | **Low temperature (0-0.3)** | More factual, less creative |
    | **Explicit instructions** | Tell model to use only context |
    | **Include source citations** | Builds trust, enables verification |
    | **Handle "I don't know"** | Better than hallucinated answers |
    | **Limit response length** | Prevents rambling |
    
    **Prompt Template:**
    ```python
    prompt = f'''You are a helpful assistant. 
    Answer ONLY using the information provided below.
    If the answer isn't in the context, say "I don't have that information."
    
    Context:
    {context}
    
    Question: {question}
    
    Answer (cite sources using [1], [2], etc.):'''
    ```
    """)

with tabs[4]:
    st.markdown("""
    ### 🏭 Production Best Practices
    
    | Practice | Why |
    |----------|-----|
    | **Monitor retrieval quality** | Track hit rate, MRR |
    | **Version your embeddings** | Model changes break old embeddings |
    | **Use persistent storage** | Don't lose your vector DB |
    | **Implement caching** | Reduce API calls and latency |
    | **Set up evaluation** | Continuous quality monitoring |
    | **Plan for scale** | Consider managed vector DBs |
    
    **Architecture Considerations:**
    - Separate ingestion and query services
    - Use async processing for large uploads
    - Implement rate limiting for LLM calls
    - Set up proper logging and observability
    """)

st.markdown("---")

# Common pitfalls
st.markdown("## ⚠️ Common Pitfalls to Avoid")

col1, col2 = st.columns(2)

with col1:
    st.error("""
    **❌ Different embedding models**
    
    Query and documents must use the SAME model!
    
    ```python
    # WRONG!
    docs = embed(chunks, model="all-MiniLM-L6-v2")
    query = embed(question, model="text-ada-002")
    
    # RIGHT!
    docs = embed(chunks, model="all-MiniLM-L6-v2")
    query = embed(question, model="all-MiniLM-L6-v2")
    ```
    """)
    
    st.error("""
    **❌ Chunks too large or too small**
    
    - Too large: Irrelevant content drowns signal
    - Too small: Lost context, fragmented info
    
    ✅ Start with 200-500 tokens, adjust based on testing
    """)
    
    st.error("""
    **❌ Ignoring metadata**
    
    Without metadata, you can't:
    - Filter by source/date/type
    - Track document provenance
    - Debug retrieval issues
    
    ✅ Always attach meaningful metadata
    """)

with col2:
    st.error("""
    **❌ Not testing with real queries**
    
    Development queries ≠ User queries!
    
    ✅ Collect actual user questions
    ✅ Build a test dataset
    ✅ Measure retrieval quality metrics
    """)
    
    st.error("""
    **❌ Trusting LLM output blindly**
    
    Even with RAG, models can:
    - Misinterpret context
    - Combine information incorrectly
    - Still hallucinate occasionally
    
    ✅ Request citations
    ✅ Implement human review for critical use cases
    """)
    
    st.error("""
    **❌ Ignoring latency**
    
    RAG adds multiple steps:
    - Embedding query
    - Vector search
    - LLM generation
    
    ✅ Cache where possible
    ✅ Use smaller models for speed-critical apps
    ✅ Consider async processing
    """)

st.markdown("---")

# Evaluation metrics
st.markdown("## 📊 Evaluation Metrics")

st.markdown("""
How do you know if your RAG system is working well?
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🎯 Retrieval Metrics
    
    **Hit Rate @ K**
    
    *"Is the answer in top-k results?"*
    
    ```
    correct_retrievals / total_queries
    ```
    
    **MRR (Mean Reciprocal Rank)**
    
    *"On average, where does the correct doc appear?"*
    
    ```
    1/rank averaged across queries
    ```
    """)

with col2:
    st.markdown("""
    ### 📝 Generation Metrics
    
    **Faithfulness**
    
    *"Does the answer use only context?"*
    
    Rate 1-5 or binary
    
    **Answer Relevance**
    
    *"Does it actually answer the question?"*
    
    Human eval or LLM-as-judge
    """)

with col3:
    st.markdown("""
    ### 🔄 End-to-End
    
    **RAGAS Score**
    
    Combines:
    - Faithfulness
    - Answer relevance
    - Context relevance
    
    **Human Preference**
    
    A/B testing, user feedback
    """)

st.markdown("---")

# Tools and frameworks
st.markdown("## 🛠️ Tools & Frameworks")

st.markdown("""
### Popular RAG Frameworks

| Framework | Best For | Key Features |
|-----------|----------|--------------|
| **LangChain** | General purpose | Extensive integrations, chains |
| **LlamaIndex** | Document-heavy apps | Focus on indexing, querying |
| **Haystack** | Production NLP | End-to-end pipelines |
| **Semantic Kernel** | .NET ecosystem | Microsoft's framework |

### Vector Databases

| Database | Type | Best For |
|----------|------|----------|
| **ChromaDB** | Open source | Prototyping, small-medium scale |
| **Pinecone** | Managed | Production, scalability |
| **Weaviate** | Open source | ML features, hybrid search |
| **Qdrant** | Open source | Performance, filtering |
| **Milvus** | Open source | Large scale, enterprise |
| **pgvector** | PostgreSQL extension | Existing Postgres users |

### Evaluation Tools

| Tool | Purpose |
|------|---------|
| **RAGAS** | RAG evaluation metrics |
| **TruLens** | LLM app evaluation |
| **Arize** | ML observability |
| **LangSmith** | LangChain debugging |
""")

st.markdown("---")

# Resources
st.markdown("## 📚 Learning Resources")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📖 Articles & Tutorials
    
    - [RAG Explained (Anthropic)](https://www.anthropic.com/research/rag)
    - [Building RAG (LangChain Docs)](https://python.langchain.com/docs/tutorials/rag/)
    - [Chunking Strategies (Pinecone)](https://www.pinecone.io/learn/chunking-strategies/)
    - [Embedding Models Comparison](https://huggingface.co/spaces/mteb/leaderboard)
    
    ### 📺 Video Courses
    
    - DeepLearning.AI: Building Applications with Vector Databases
    - LangChain RAG Course
    - Pinecone Learning Center
    """)

with col2:
    st.markdown("""
    ### 📦 GitHub Repositories
    
    - [LangChain](https://github.com/langchain-ai/langchain)
    - [LlamaIndex](https://github.com/run-llama/llama_index)
    - [ChromaDB](https://github.com/chroma-core/chroma)
    - [RAGAS](https://github.com/explodinggradients/ragas)
    
    ### 🔬 Research Papers
    
    - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    - "Dense Passage Retrieval for Open-Domain Question Answering"
    - "REPLUG: Retrieval-Augmented Black-Box Language Models"
    """)

st.markdown("---")

# What's next
st.markdown("## 🚀 What's Next?")

st.markdown("""
### Suggested Learning Path

1. **Build a simple RAG app** with your own documents
2. **Experiment with chunking** strategies on your content
3. **Try different embedding models** and compare quality
4. **Add evaluation** metrics to measure improvement
5. **Explore advanced topics:**
   - Hybrid search (semantic + keyword)
   - Re-ranking with cross-encoders
   - Multi-modal RAG (images, tables)
   - Agentic RAG (multiple retrieval steps)
   - Fine-tuning embeddings for your domain
""")

st.markdown("---")

# Final message
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; border-radius: 15px; margin: 2rem 0;">
    <h2>🎓 You've Completed the RAG Walkthrough!</h2>
    <p style="font-size: 1.1rem;">
        You now have the knowledge to build RAG systems that can:
    </p>
    <p>
        ✅ Process and chunk documents effectively<br>
        ✅ Generate and store embeddings<br>
        ✅ Retrieve relevant context<br>
        ✅ Generate accurate, grounded responses
    </p>
    <p style="margin-top: 1rem;">
        <strong>Remember:</strong> The best RAG system is one that's been tested, 
        evaluated, and iteratively improved based on real usage!
    </p>
</div>
""", unsafe_allow_html=True)

# Navigation back to start
st.markdown("---")
st.markdown("### 🏠 Return to Start")
st.markdown("Go back to the home page to review any concepts!")
