"""
Embedding utilities for the RAG walkthrough app.
Provides embedding generation using Sentence-Transformers.
"""

import numpy as np
from typing import List, Union
import streamlit as st


@st.cache_resource
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load and cache the embedding model.
    
    Args:
        model_name: Name of the sentence-transformers model to use
        
    Returns:
        SentenceTransformer model instance
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def get_embeddings(
    texts: Union[str, List[str]], 
    model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Generate embeddings for the given text(s).
    
    Args:
        texts: Single text or list of texts to embed
        model_name: Name of the embedding model to use
        
    Returns:
        Numpy array of embeddings
    """
    model = get_embedding_model(model_name)
    
    if isinstance(texts, str):
        texts = [texts]
    
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance (L2 norm)
    """
    return np.linalg.norm(vec1 - vec2)


def reduce_dimensions(
    embeddings: np.ndarray, 
    n_components: int = 2, 
    method: str = "pca"
) -> np.ndarray:
    """
    Reduce embedding dimensions for visualization.
    
    Args:
        embeddings: High-dimensional embeddings
        n_components: Target number of dimensions (2 or 3)
        method: Dimensionality reduction method ('pca' or 'tsne')
        
    Returns:
        Reduced dimensionality embeddings
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    if embeddings.shape[0] < n_components:
        # Not enough samples
        return embeddings[:, :n_components]
    
    if method.lower() == "pca":
        reducer = PCA(n_components=n_components)
    elif method.lower() == "tsne":
        # t-SNE needs perplexity < n_samples
        perplexity = min(30, embeddings.shape[0] - 1)
        reducer = TSNE(n_components=n_components, perplexity=max(1, perplexity), random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return reducer.fit_transform(embeddings)


def get_embedding_dimension(model_name: str = "all-MiniLM-L6-v2") -> int:
    """
    Get the embedding dimension for a given model.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        Embedding dimension
    """
    model = get_embedding_model(model_name)
    return model.get_sentence_embedding_dimension()


# Popular embedding models info
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "description": "Fast and efficient, great for demos",
        "speed": "⚡ Fast",
        "quality": "⭐⭐⭐"
    },
    "all-mpnet-base-v2": {
        "name": "all-mpnet-base-v2", 
        "dimension": 768,
        "description": "Higher quality, slower",
        "speed": "🐢 Slower",
        "quality": "⭐⭐⭐⭐⭐"
    },
    "paraphrase-MiniLM-L6-v2": {
        "name": "paraphrase-MiniLM-L6-v2",
        "dimension": 384,
        "description": "Good for paraphrase detection",
        "speed": "⚡ Fast",
        "quality": "⭐⭐⭐"
    }
}
