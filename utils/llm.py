"""
LLM utilities for the RAG walkthrough app.
Provides Groq integration for text generation.
"""

import streamlit as st
from groq import Groq
from typing import Optional, List

# Available Groq models
GROQ_MODELS = {
    "llama-3.3-70b-versatile": {
        "name": "Llama 3.3 70B Versatile",
        "description": "Most capable Llama model, great for complex tasks",
        "context_window": 128000
    },
    "llama-3.1-8b-instant": {
        "name": "Llama 3.1 8B Instant",
        "description": "Fast and efficient for simple tasks",
        "context_window": 128000
    },
    "mixtral-8x7b-32768": {
        "name": "Mixtral 8x7B",
        "description": "Strong reasoning capabilities",
        "context_window": 32768
    }
}

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def get_groq_client(api_key: str) -> Groq:
    """
    Create a Groq client with the given API key.
    
    Args:
        api_key: Groq API key
        
    Returns:
        Groq client instance
    """
    return Groq(api_key=api_key)


def generate_response(
    client: Groq,
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None
) -> str:
    """
    Generate a response using Groq.
    
    Args:
        client: Groq client
        prompt: User prompt
        model: Model to use
        temperature: Generation temperature (0-1)
        max_tokens: Maximum tokens in response
        system_prompt: Optional system prompt
        
    Returns:
        Generated text response
    """
    messages = []
    
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content


def generate_rag_response(
    client: Groq,
    question: str,
    context_chunks: List[str],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3
) -> str:
    """
    Generate a RAG response using retrieved context.
    
    Args:
        client: Groq client
        question: User question
        context_chunks: Retrieved context chunks
        model: Model to use
        temperature: Generation temperature
        
    Returns:
        Generated answer
    """
    # Format context with source numbers
    formatted_context = "\n\n".join([
        f"[Source {i+1}] {chunk}" 
        for i, chunk in enumerate(context_chunks)
    ])
    
    system_prompt = """You are a helpful assistant. Answer questions based ONLY on the provided context. 
If the answer isn't in the context, say "I don't have enough information to answer that question."
When possible, cite your sources using [Source N] notation."""

    user_prompt = f"""Answer the question based on the context below.

Context:
{formatted_context}

Question: {question}

Answer:"""

    return generate_response(
        client=client,
        prompt=user_prompt,
        model=model,
        temperature=temperature,
        system_prompt=system_prompt
    )
