"""
Document chunking utilities for the RAG walkthrough app.
Provides various chunking strategies.
"""

from typing import List, Tuple
import re


def chunk_by_characters(
    text: str, 
    chunk_size: int = 500, 
    overlap: int = 50
) -> List[str]:
    """
    Split text into chunks of fixed character size with overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum number of characters per chunk
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return [c for c in chunks if c]


def chunk_by_sentences(
    text: str, 
    sentences_per_chunk: int = 3, 
    overlap_sentences: int = 1
) -> List[str]:
    """
    Split text into chunks based on sentence boundaries.
    
    Args:
        text: Text to chunk
        sentences_per_chunk: Number of sentences per chunk
        overlap_sentences: Number of overlapping sentences
        
    Returns:
        List of text chunks
    """
    # Simple sentence splitting
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    i = 0
    
    while i < len(sentences):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunk = ' '.join(chunk_sentences)
        chunks.append(chunk)
        i += sentences_per_chunk - overlap_sentences
        
        if i >= len(sentences):
            break
    
    return chunks


def chunk_by_paragraphs(text: str) -> List[str]:
    """
    Split text into chunks based on paragraph boundaries.
    
    Args:
        text: Text to chunk
        
    Returns:
        List of text chunks (paragraphs)
    """
    paragraphs = text.split('\n\n')
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_by_tokens(
    text: str, 
    max_tokens: int = 200, 
    overlap_tokens: int = 20
) -> List[str]:
    """
    Split text into chunks based on token count (approximate using words).
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of overlapping tokens
        
    Returns:
        List of text chunks
    """
    # Simple tokenization by whitespace
    words = text.split()
    
    if not words:
        return []
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + max_tokens
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap_tokens
        
        if start >= len(words):
            break
    
    return chunks


def recursive_chunk(
    text: str, 
    max_chunk_size: int = 500,
    separators: List[str] = None
) -> List[str]:
    """
    Recursively split text using a hierarchy of separators.
    
    Args:
        text: Text to chunk
        max_chunk_size: Maximum chunk size
        separators: List of separators to try in order
        
    Returns:
        List of text chunks
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    
    if not text or len(text) <= max_chunk_size:
        return [text] if text else []
    
    # Try each separator
    for sep in separators:
        if sep == "":
            # Last resort: split by character
            return chunk_by_characters(text, max_chunk_size, overlap=0)
        
        if sep in text:
            parts = text.split(sep)
            
            chunks = []
            current_chunk = ""
            
            for part in parts:
                if len(current_chunk) + len(sep) + len(part) <= max_chunk_size:
                    if current_chunk:
                        current_chunk += sep + part
                    else:
                        current_chunk = part
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    if len(part) > max_chunk_size:
                        # Recursively split with next separator
                        sub_chunks = recursive_chunk(
                            part, 
                            max_chunk_size, 
                            separators[separators.index(sep) + 1:]
                        )
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = part
            
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks
    
    return [text]


def get_chunk_stats(chunks: List[str]) -> dict:
    """
    Get statistics about the chunks.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        Dictionary with chunk statistics
    """
    if not chunks:
        return {
            "count": 0,
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0,
            "total_length": 0
        }
    
    lengths = [len(c) for c in chunks]
    
    return {
        "count": len(chunks),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "total_length": sum(lengths)
    }


# Chunking strategies info for UI
CHUNKING_STRATEGIES = {
    "characters": {
        "name": "Fixed Characters",
        "description": "Split by character count with overlap",
        "pros": ["Simple", "Predictable size"],
        "cons": ["May cut words/sentences"]
    },
    "sentences": {
        "name": "Sentence-based",
        "description": "Split at sentence boundaries",
        "pros": ["Preserves meaning", "Complete sentences"],
        "cons": ["Variable chunk sizes"]
    },
    "paragraphs": {
        "name": "Paragraph-based",
        "description": "Split at paragraph boundaries",
        "pros": ["Preserves context", "Logical units"],
        "cons": ["May be too large or small"]
    },
    "tokens": {
        "name": "Token-based",
        "description": "Split by token/word count",
        "pros": ["Good for LLM context limits"],
        "cons": ["May split sentences"]
    },
    "recursive": {
        "name": "Recursive",
        "description": "Hierarchical splitting (paragraphs → sentences → words)",
        "pros": ["Best of all worlds", "Smart splitting"],
        "cons": ["More complex"]
    }
}
