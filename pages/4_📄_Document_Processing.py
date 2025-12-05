"""
Page 4: Document Processing
===========================
Learn chunking strategies and document preparation for RAG.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Document Processing",
    page_icon="📄",
    layout="wide"
)

# Title
st.markdown("# 📄 Document Processing")
st.markdown("*Preparing Documents for Effective Retrieval*")
st.markdown("---")

# Why chunking matters
st.markdown("## 🤔 Why Does Chunking Matter?")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    Before storing documents in a vector database, we need to split them into 
    smaller pieces called **chunks**. Why?
    
    1. **LLM Context Limits** — Models have token limits (4k-128k). Large docs won't fit.
    2. **Precision** — Smaller chunks = more precise retrieval
    3. **Relevance** — A 10-page doc might have 1 relevant paragraph
    4. **Embedding Quality** — Embeddings work best on focused content
    
    The challenge: chunks must be **small enough** to be precise, but 
    **large enough** to maintain context!
    """)

with col2:
    st.warning("""
    **The Goldilocks Problem:**
    
    🐻 Too big → Irrelevant info included
    
    🐻 Too small → Context lost
    
    🐻 Just right → Focused but complete
    """)

st.markdown("---")

# Chunking strategies
st.markdown("## 📏 Chunking Strategies")

tabs = st.tabs(["Fixed Size", "Sentence-Based", "Paragraph-Based", "Recursive", "Semantic"])

with tabs[0]:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Fixed Character/Token Size
        
        Split text every N characters or tokens.
        
        **Pros:**
        - Simple and predictable
        - Consistent chunk sizes
        - Easy to implement
        
        **Cons:**
        - May cut words mid-sentence
        - Ignores document structure
        - Can break semantic units
        """)
    
    with col2:
        st.code("""
def chunk_fixed(text, size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
        """, language="python")

with tabs[1]:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Sentence-Based
        
        Split at sentence boundaries, group N sentences per chunk.
        
        **Pros:**
        - Preserves complete thoughts
        - Natural reading units
        - Respects grammar
        
        **Cons:**
        - Variable chunk sizes
        - Long sentences = large chunks
        """)
    
    with col2:
        st.code("""
import re

def chunk_sentences(text, n=3, overlap=1):
    sentences = re.split(r'(?<=[.!?])\\s+', text)
    chunks = []
    for i in range(0, len(sentences), n - overlap):
        chunk = ' '.join(sentences[i:i+n])
        chunks.append(chunk)
    return chunks
        """, language="python")

with tabs[2]:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Paragraph-Based
        
        Split at paragraph boundaries (double newlines).
        
        **Pros:**
        - Preserves logical sections
        - Author-intended groupings
        - Great for structured docs
        
        **Cons:**
        - Very variable sizes
        - Some paragraphs too large
        """)
    
    with col2:
        st.code("""
def chunk_paragraphs(text):
    paragraphs = text.split('\\n\\n')
    return [p.strip() for p in paragraphs if p.strip()]
        """, language="python")

with tabs[3]:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Recursive Splitting
        
        Try separators in order: paragraphs → sentences → words.
        
        **Pros:**
        - Best of all worlds
        - Respects structure when possible
        - Handles edge cases
        
        **Cons:**
        - More complex logic
        - Harder to tune
        """)
    
    with col2:
        st.code("""
def chunk_recursive(text, max_size=500, seps=None):
    if seps is None:
        seps = ["\\n\\n", "\\n", ". ", " "]
    
    if len(text) <= max_size:
        return [text]
    
    for sep in seps:
        if sep in text:
            parts = text.split(sep)
            # Recursively process...
            # Combine small chunks...
    return chunks
        """, language="python")

with tabs[4]:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Semantic Chunking
        
        Use embeddings to find natural break points.
        
        **Pros:**
        - Content-aware splitting
        - Groups related content
        - Smart topic boundaries
        
        **Cons:**
        - Computationally expensive
        - Requires embeddings twice
        - More complex setup
        """)
    
    with col2:
        st.code("""
def chunk_semantic(text, threshold=0.5):
    sentences = split_sentences(text)
    embeddings = embed(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i-1], embeddings[i])
        if sim < threshold:  # Topic change
            chunks.append(' '.join(current_chunk))
            current_chunk = []
        current_chunk.append(sentences[i])
    
    return chunks
        """, language="python")

st.markdown("---")

# Interactive chunking demo
st.markdown("## 🎮 Interactive: Chunking Playground")

# Try to import chunking utilities
try:
    from utils.chunking import (
        chunk_by_characters, chunk_by_sentences, chunk_by_paragraphs,
        chunk_by_tokens, recursive_chunk, get_chunk_stats, CHUNKING_STRATEGIES
    )
    from utils.visualization import plot_chunk_distribution
    chunking_available = True
except ImportError as e:
    chunking_available = False
    st.warning(f"Import warning: {e}")
    chunking_available = True  # The utilities should work

if chunking_available:
    # Sample text
    sample_text = """Artificial Intelligence (AI) has transformed the way we live and work. From virtual assistants like Siri and Alexa to recommendation systems on Netflix and Spotify, AI is everywhere.

Machine learning, a subset of AI, enables computers to learn from data without being explicitly programmed. Deep learning takes this further by using neural networks with many layers.

Natural Language Processing (NLP) is another exciting field. It allows machines to understand and generate human language. Applications include chatbots, translation services, and sentiment analysis.

The future of AI holds immense promise. Researchers are working on general AI that could match human intelligence across all domains. However, ethical considerations and safety measures must keep pace with technological advances.

In healthcare, AI is revolutionizing diagnosis and treatment. Machine learning models can detect diseases from medical images with remarkable accuracy. Personalized medicine is becoming a reality.

The educational sector is also being transformed. AI tutors can provide personalized learning experiences, adapting to each student's pace and style. Assessment and feedback can be automated, freeing teachers to focus on mentoring."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter or paste your text:",
            value=sample_text,
            height=300
        )
    
    with col2:
        strategy = st.selectbox(
            "Chunking strategy:",
            options=["characters", "sentences", "paragraphs", "tokens", "recursive"],
            format_func=lambda x: CHUNKING_STRATEGIES[x]["name"]
        )
        
        st.markdown(f"**{CHUNKING_STRATEGIES[strategy]['description']}**")
        
        # Strategy-specific parameters
        if strategy == "characters":
            chunk_size = st.slider("Chunk size (chars):", 100, 1000, 300)
            overlap = st.slider("Overlap (chars):", 0, 200, 50)
        elif strategy == "sentences":
            sentences_per_chunk = st.slider("Sentences per chunk:", 1, 10, 3)
            overlap_sentences = st.slider("Overlap (sentences):", 0, 3, 1)
        elif strategy == "tokens":
            max_tokens = st.slider("Max tokens:", 50, 500, 150)
            overlap_tokens = st.slider("Overlap (tokens):", 0, 50, 20)
        elif strategy == "recursive":
            max_chunk_size = st.slider("Max chunk size:", 200, 1000, 400)
    
    if st.button("✂️ Chunk Text", type="primary"):
        # Apply chunking
        if strategy == "characters":
            chunks = chunk_by_characters(text_input, chunk_size, overlap)
        elif strategy == "sentences":
            chunks = chunk_by_sentences(text_input, sentences_per_chunk, overlap_sentences)
        elif strategy == "paragraphs":
            chunks = chunk_by_paragraphs(text_input)
        elif strategy == "tokens":
            chunks = chunk_by_tokens(text_input, max_tokens, overlap_tokens)
        else:  # recursive
            chunks = recursive_chunk(text_input, max_chunk_size)
        
        # Store results
        st.session_state['chunks'] = chunks
        st.session_state['chunk_stats'] = get_chunk_stats(chunks)
    
    # Display results
    if 'chunks' in st.session_state:
        chunks = st.session_state['chunks']
        stats = st.session_state['chunk_stats']
        
        st.markdown("---")
        st.markdown("### 📊 Results")
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Chunks", stats['count'])
        col2.metric("Avg Length", f"{stats['avg_length']:.0f}")
        col3.metric("Min Length", stats['min_length'])
        col4.metric("Max Length", stats['max_length'])
        
        # Distribution chart
        if len(chunks) > 1:
            fig = plot_chunk_distribution([len(c) for c in chunks])
            st.plotly_chart(fig, use_container_width=True)
        
        # Show chunks
        st.markdown("### 📝 Chunks Preview")
        
        for i, chunk in enumerate(chunks):
            with st.expander(f"Chunk {i+1} ({len(chunk)} chars)"):
                st.markdown(f"""
                <div style="background: #f5f5f5; padding: 1rem; border-radius: 8px; 
                            border-left: 4px solid #667eea; white-space: pre-wrap;">
{chunk}
                </div>
                """, unsafe_allow_html=True)

st.markdown("---")

# Overlap explained
st.markdown("## 🔗 The Importance of Overlap")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    **Overlap** means including some text from the previous chunk in the next one.
    
    **Why it matters:**
    
    1. **Context preservation** — Maintains continuity across chunks
    2. **Sentence completion** — Ensures thoughts aren't cut off
    3. **Better retrieval** — Relevant info might span chunk boundaries
    
    **Rule of thumb:** Use 10-20% overlap
    
    Example with 100-char chunks, 20-char overlap:
    """)

with col2:
    st.code("""
Original text: "The quick brown fox jumps over the lazy dog."

Without overlap:
Chunk 1: "The quick brown fox j"
Chunk 2: "umps over the lazy do"
Chunk 3: "g."

With overlap (20 chars):
Chunk 1: "The quick brown fox j"
Chunk 2: "fox jumps over the lazy do"
Chunk 3: "the lazy dog."

Notice how "jumps" and "dog" are preserved!
    """, language="text")

st.markdown("---")

# Metadata
st.markdown("## 🏷️ Adding Metadata to Chunks")

st.markdown("""
Metadata helps you filter and organize chunks during retrieval.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Common Metadata Fields:**
    
    - `source` — File name or URL
    - `page` — Page number in document
    - `chapter` — Section or chapter
    - `date` — Document date
    - `author` — Content creator
    - `chunk_index` — Position in document
    """)

with col2:
    st.code("""
chunks_with_metadata = []

for i, chunk in enumerate(chunks):
    chunks_with_metadata.append({
        "content": chunk,
        "metadata": {
            "source": "company_policy.pdf",
            "page": 5,
            "chunk_index": i,
            "category": "HR"
        }
    })

# In ChromaDB:
collection.add(
    documents=[c["content"] for c in chunks_with_metadata],
    metadatas=[c["metadata"] for c in chunks_with_metadata],
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

# Query with filter:
results = collection.query(
    query_texts=["vacation policy"],
    where={"category": "HR"}
)
    """, language="python")

st.markdown("---")

# Key takeaways
st.markdown("## 📌 Key Takeaways")

st.success("""
1. **Chunking is crucial** — Bad chunks = bad retrieval = bad answers
2. **Match strategy to content** — Structured docs → paragraphs; unstructured → recursive
3. **Size matters** — 200-500 tokens is a good starting point
4. **Use overlap** — 10-20% overlap preserves context
5. **Add metadata** — Enables filtering and improves organization
6. **Experiment** — The best chunking strategy depends on your specific use case
""")

# Navigation
st.markdown("---")
st.markdown("### ➡️ Next: Retrieval Mechanisms")
st.markdown("Learn how to find the most relevant chunks for a query.")
