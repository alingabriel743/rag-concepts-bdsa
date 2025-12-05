"""
ChromaDB vector store utilities for the RAG walkthrough app.
Provides a clean wrapper around ChromaDB operations.
"""

import chromadb
from typing import List, Dict, Any, Optional
import streamlit as st
import uuid


@st.cache_resource
def get_chroma_client(persist: bool = False, path: str = "./chroma_db"):
    """
    Get a ChromaDB client instance.
    
    Args:
        persist: Whether to persist data to disk
        path: Path for persistent storage
        
    Returns:
        ChromaDB client
    """
    if persist:
        return chromadb.PersistentClient(path=path)
    else:
        return chromadb.Client()


def create_collection(
    client: chromadb.Client,
    name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> chromadb.Collection:
    """
    Create or get a collection.
    
    Args:
        client: ChromaDB client
        name: Collection name
        metadata: Optional collection metadata
        
    Returns:
        ChromaDB collection
    """
    return client.get_or_create_collection(
        name=name,
        metadata=metadata or {"description": "RAG demo collection"}
    )


def add_documents(
    collection: chromadb.Collection,
    documents: List[str],
    embeddings: List[List[float]] = None,
    metadatas: List[Dict[str, Any]] = None,
    ids: List[str] = None
) -> List[str]:
    """
    Add documents to a collection.
    
    Args:
        collection: ChromaDB collection
        documents: List of document texts
        embeddings: Pre-computed embeddings (optional)
        metadatas: Document metadata (optional)
        ids: Document IDs (auto-generated if not provided)
        
    Returns:
        List of document IDs
    """
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in documents]
    
    add_kwargs = {
        "documents": documents,
        "ids": ids
    }
    
    if embeddings is not None:
        add_kwargs["embeddings"] = embeddings
    
    if metadatas is not None:
        add_kwargs["metadatas"] = metadatas
    
    collection.add(**add_kwargs)
    
    return ids


def query_collection(
    collection: chromadb.Collection,
    query_texts: List[str] = None,
    query_embeddings: List[List[float]] = None,
    n_results: int = 5,
    where: Dict[str, Any] = None,
    include: List[str] = None
) -> Dict[str, Any]:
    """
    Query a collection for similar documents.
    
    Args:
        collection: ChromaDB collection
        query_texts: Query text(s)
        query_embeddings: Query embedding(s)
        n_results: Number of results to return
        where: Metadata filter
        include: What to include in results
        
    Returns:
        Query results
    """
    if include is None:
        include = ["documents", "metadatas", "distances"]
    
    query_kwargs = {
        "n_results": n_results,
        "include": include
    }
    
    if query_texts is not None:
        query_kwargs["query_texts"] = query_texts
    elif query_embeddings is not None:
        query_kwargs["query_embeddings"] = query_embeddings
    
    if where is not None:
        query_kwargs["where"] = where
    
    return collection.query(**query_kwargs)


def get_collection_stats(collection: chromadb.Collection) -> Dict[str, Any]:
    """
    Get statistics about a collection.
    
    Args:
        collection: ChromaDB collection
        
    Returns:
        Dictionary with collection statistics
    """
    count = collection.count()
    
    return {
        "name": collection.name,
        "count": count,
        "metadata": collection.metadata
    }


def delete_collection(client: chromadb.Client, name: str) -> None:
    """
    Delete a collection.
    
    Args:
        client: ChromaDB client
        name: Collection name
    """
    try:
        client.delete_collection(name)
    except ValueError:
        pass  # Collection doesn't exist


def list_collections(client: chromadb.Client) -> List[str]:
    """
    List all collections.
    
    Args:
        client: ChromaDB client
        
    Returns:
        List of collection names
    """
    collections = client.list_collections()
    return [c.name for c in collections]


# ChromaDB info for UI
CHROMADB_INFO = {
    "name": "ChromaDB",
    "description": "Open-source embedding database",
    "features": [
        "Simple API",
        "In-memory or persistent",
        "Built-in embedding functions",
        "Metadata filtering",
        "Multiple distance metrics"
    ],
    "use_cases": [
        "Semantic search",
        "RAG applications",
        "Recommendation systems",
        "Duplicate detection"
    ],
    "distance_metrics": {
        "l2": "Euclidean distance (default)",
        "ip": "Inner product",
        "cosine": "Cosine similarity"
    }
}
