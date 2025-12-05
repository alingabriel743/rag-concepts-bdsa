"""
Page 6: Generation with Context
===============================
Learn prompt engineering for RAG and context formatting.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Generation with Context",
    page_icon="🤖",
    layout="wide"
)

# Title
st.markdown("# 🤖 Generation with Context")
st.markdown("*Crafting Prompts That Leverage Retrieved Information*")
st.markdown("---")

# The generation step
st.markdown("## 🎯 The Generation Step in RAG")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    The **Generation** step is where the LLM creates an answer using:
    
    1. **The user's question**
    2. **Retrieved context** from the vector database
    3. **A carefully crafted prompt** that guides the model
    
    This is the "G" in RAG — where all the preparation pays off!
    
    The key challenge: How do we format the prompt so the LLM:
    - Uses the provided context effectively
    - Doesn't hallucinate beyond the context
    - Produces clear, helpful answers
    """)

with col2:
    st.info("""
    **Generation Input:**
    ```
    System: You are a helpful assistant...
    
    Context: [Retrieved documents]
    
    Question: User's query
    
    Answer: [LLM generates this]
    ```
    """)

st.markdown("---")

# Prompt anatomy
st.markdown("## 📝 Anatomy of a RAG Prompt")

st.markdown("""
A well-structured RAG prompt has several components:
""")

# Visual prompt breakdown
st.code("""
┌─────────────────────────────────────────────────────────────────┐
│  SYSTEM PROMPT                                                  │
│  "You are a helpful assistant. Answer questions based only     │
│   on the provided context. If you don't know, say so."         │
├─────────────────────────────────────────────────────────────────┤
│  CONTEXT SECTION                                                │
│  "Here is the relevant information:                            │
│                                                                 │
│   Document 1: [Retrieved chunk 1]                               │
│   Document 2: [Retrieved chunk 2]                               │
│   Document 3: [Retrieved chunk 3]"                              │
├─────────────────────────────────────────────────────────────────┤
│  USER QUESTION                                                  │
│  "Question: What is the refund policy?"                        │
├─────────────────────────────────────────────────────────────────┤
│  INSTRUCTIONS (Optional)                                        │
│  "Provide a clear, concise answer. Cite the source if          │
│   possible."                                                    │
└─────────────────────────────────────────────────────────────────┘
""", language="text")

st.markdown("---")

# Prompt templates
st.markdown("## 📋 Prompt Template Examples")

tab1, tab2, tab3, tab4 = st.tabs(["Basic", "With Citations", "Conversational", "Strict"])

with tab1:
    st.markdown("### Basic RAG Prompt")
    st.code("""
prompt = f'''Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:'''
    """, language="python")
    
    st.markdown("**Best for:** Simple Q&A, getting started")

with tab2:
    st.markdown("### With Source Citations")
    st.code("""
prompt = f'''You are a research assistant. Answer the question using ONLY 
the information provided in the sources below. Cite your sources using 
[Source N] notation.

Sources:
[Source 1] {doc1}
[Source 2] {doc2}
[Source 3] {doc3}

Question: {question}

Provide a detailed answer with citations:'''
    """, language="python")
    
    st.markdown("**Best for:** Research, fact-checking, transparency")

with tab3:
    st.markdown("### Conversational Style")
    st.code("""
prompt = f'''You are a friendly customer support agent for TechCorp.
Use the following information to help the customer:

<knowledge_base>
{context}
</knowledge_base>

Customer message: {question}

Respond in a warm, helpful manner. If you don't have the information 
to answer fully, acknowledge this and offer to connect them with a 
human agent.'''
    """, language="python")
    
    st.markdown("**Best for:** Customer support, chatbots")

with tab4:
    st.markdown("### Strict (Prevent Hallucination)")
    st.code("""
prompt = f'''You are a precise assistant that ONLY uses information 
from the provided context.

RULES:
1. ONLY use information explicitly stated in the context
2. If the context doesn't contain the answer, say "I don't have 
   information about that in the provided documents"
3. Do not make assumptions or use external knowledge
4. Quote relevant parts of the context when possible

Context:
---
{context}
---

Question: {question}

Answer (following the rules strictly):'''
    """, language="python")
    
    st.markdown("**Best for:** Legal, medical, high-stakes applications")

st.markdown("---")

# Interactive prompt builder
st.markdown("## 🎮 Interactive: Prompt Builder")

st.markdown("Build and test your RAG prompt template:")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🔧 Configure Prompt")
    
    system_prompt = st.text_area(
        "System Prompt:",
        value="You are a helpful assistant. Answer questions based only on the provided context. If the answer isn't in the context, say you don't know.",
        height=100
    )
    
    context_format = st.selectbox(
        "Context Format:",
        options=[
            "Simple list",
            "Numbered sources",
            "XML tags",
            "Markdown sections"
        ]
    )
    
    include_instructions = st.checkbox("Include output instructions", value=True)
    
    if include_instructions:
        output_instructions = st.text_area(
            "Output Instructions:",
            value="Provide a clear, concise answer. If citing sources, use [Source N] format.",
            height=80
        )

with col2:
    st.markdown("### 📄 Sample Data")
    
    sample_context = [
        "TechCorp was founded in 2015 by Jane Smith in San Francisco.",
        "TechCorp offers a 60-day money-back guarantee on all products.",
        "TechCorp's customer support is available 24/7 via chat or email."
    ]
    
    st.markdown("**Sample Context Documents:**")
    for i, doc in enumerate(sample_context):
        st.markdown(f"{i+1}. {doc}")
    
    sample_question = st.text_input("Sample Question:", value="What is TechCorp's refund policy?")

# Build the prompt
st.markdown("---")
st.markdown("### 📝 Generated Prompt")

# Format context based on selection
if context_format == "Simple list":
    formatted_context = "\n".join(sample_context)
elif context_format == "Numbered sources":
    formatted_context = "\n".join([f"[Source {i+1}] {doc}" for i, doc in enumerate(sample_context)])
elif context_format == "XML tags":
    formatted_context = "\n".join([f"<document id='{i+1}'>{doc}</document>" for i, doc in enumerate(sample_context)])
else:  # Markdown sections
    formatted_context = "\n\n".join([f"### Document {i+1}\n{doc}" for i, doc in enumerate(sample_context)])

# Build prompt
prompt_parts = [f"System: {system_prompt}\n"]
prompt_parts.append(f"Context:\n{formatted_context}\n")
prompt_parts.append(f"Question: {sample_question}\n")
if include_instructions:
    prompt_parts.append(f"Instructions: {output_instructions}\n")
prompt_parts.append("Answer:")

final_prompt = "\n".join(prompt_parts)

st.code(final_prompt, language="text")

# Copy button
st.markdown(f"**Total characters:** {len(final_prompt)} | **Estimated tokens:** ~{len(final_prompt) // 4}")

st.markdown("---")

# Handling edge cases
st.markdown("## ⚠️ Handling Edge Cases")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### No Relevant Results
    
    What if the vector search returns nothing useful?
    
    **Options:**
    1. **Graceful fallback:** "I don't have specific information about that"
    2. **Suggest alternatives:** "I couldn't find X, but here's info about Y"
    3. **Ask for clarification:** "Could you rephrase your question?"
    
    ```python
    if not results or max_similarity < 0.3:
        return "I don't have information about that in my knowledge base."
    ```
    """)

with col2:
    st.markdown("""
    ### Context Too Long
    
    What if retrieved content exceeds the context window?
    
    **Options:**
    1. **Truncate:** Use only top-k chunks
    2. **Summarize:** Compress chunks first
    3. **Prioritize:** Use metadata to select most relevant
    
    ```python
    def fit_context(chunks, max_tokens=3000):
        context = ""
        for chunk in chunks:
            if len(context) + len(chunk) < max_tokens * 4:
                context += chunk + "\\n\\n"
            else:
                break
        return context
    ```
    """)

st.markdown("---")

# LLM integration example
st.markdown("## 🔌 LLM Integration")

st.markdown("Example using OpenAI's API:")

st.code("""
from openai import OpenAI

def generate_rag_response(question, context_chunks):
    client = OpenAI()
    
    # Format context
    context = "\\n\\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)])
    
    # Build messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer based only on the provided context."
        },
        {
            "role": "user", 
            "content": f\"\"\"Context:
{context}

Question: {question}

Answer based on the context above:\"\"\"
        }
    ]
    
    # Generate response
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content
""", language="python")

st.markdown("---")

# Temperature and parameters
st.markdown("## 🌡️ Generation Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Temperature
    
    Controls randomness/creativity.
    
    - **0.0:** Deterministic, factual
    - **0.7:** Balanced (default)
    - **1.0+:** Creative, varied
    
    **For RAG:** Use 0.0-0.5 for factual answers
    """)

with col2:
    st.markdown("""
    ### Max Tokens
    
    Limits response length.
    
    - Short answers: 100-200
    - Detailed: 500-1000
    - Long-form: 2000+
    
    **Tip:** Set based on expected answer length
    """)

with col3:
    st.markdown("""
    ### Top-p (Nucleus)
    
    Alternative to temperature.
    
    - 0.1: Very focused
    - 0.9: More diverse
    
    **Tip:** Usually keep at 1.0 and use temperature instead
    """)

st.markdown("---")

# Key takeaways
st.markdown("## 📌 Key Takeaways")

st.success("""
1. **Prompt structure matters** — Clear separation of system prompt, context, and question
2. **Be explicit about rules** — Tell the model to use only provided context
3. **Format context clearly** — Numbered sources, XML tags, or markdown
4. **Handle edge cases** — No results, context overflow, ambiguous queries
5. **Use low temperature** — 0.0-0.5 for factual RAG applications
6. **Request citations** — Helps verify accuracy and build trust
""")

# Navigation
st.markdown("---")
st.markdown("### ➡️ Next: Complete RAG Pipeline")
st.markdown("Put it all together in a working end-to-end demo!")
