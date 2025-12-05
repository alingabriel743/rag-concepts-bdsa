"""
Visualization utilities for the RAG walkthrough app.
Provides Plotly charts for embeddings and diagrams.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Optional


def plot_embeddings_2d(
    embeddings: np.ndarray,
    labels: List[str],
    title: str = "Embedding Space (2D)",
    color_values: Optional[List[float]] = None
) -> go.Figure:
    """
    Create a 2D scatter plot of embeddings.
    
    Args:
        embeddings: 2D array of embeddings (n_samples, 2)
        labels: List of labels for each point
        title: Plot title
        color_values: Optional values for coloring points
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if color_values is not None:
        fig.add_trace(go.Scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            mode='markers+text',
            text=labels,
            textposition='top center',
            marker=dict(
                size=12,
                color=color_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Similarity")
            ),
            hovertemplate='<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            mode='markers+text',
            text=labels,
            textposition='top center',
            marker=dict(
                size=12,
                color='#667eea',
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        template="plotly_white",
        showlegend=False,
        height=500
    )
    
    return fig


def plot_embeddings_3d(
    embeddings: np.ndarray,
    labels: List[str],
    title: str = "Embedding Space (3D)"
) -> go.Figure:
    """
    Create a 3D scatter plot of embeddings.
    
    Args:
        embeddings: 3D array of embeddings (n_samples, 3)
        labels: List of labels for each point
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[go.Scatter3d(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        z=embeddings[:, 2],
        mode='markers+text',
        text=labels,
        marker=dict(
            size=8,
            color=embeddings[:, 2],
            colorscale='Viridis',
            opacity=0.8
        ),
        hovertemplate='<b>%{text}</b><extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3"
        ),
        template="plotly_white",
        height=600
    )
    
    return fig


def plot_similarity_matrix(
    similarity_matrix: np.ndarray,
    labels: List[str],
    title: str = "Similarity Matrix"
) -> go.Figure:
    """
    Create a heatmap of similarity scores.
    
    Args:
        similarity_matrix: 2D array of similarity scores
        labels: Labels for rows/columns
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=labels,
        y=labels,
        colorscale='RdYlGn',
        text=np.round(similarity_matrix, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{y} vs %{x}<br>Similarity: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Documents",
        yaxis_title="Documents",
        template="plotly_white",
        height=500
    )
    
    return fig


def plot_chunk_distribution(
    chunk_lengths: List[int],
    title: str = "Chunk Length Distribution"
) -> go.Figure:
    """
    Create a histogram of chunk lengths.
    
    Args:
        chunk_lengths: List of chunk lengths
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[go.Histogram(
        x=chunk_lengths,
        nbinsx=20,
        marker_color='#667eea',
        opacity=0.75
    )])
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Chunk Length (characters)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400
    )
    
    # Add mean line
    mean_length = np.mean(chunk_lengths)
    fig.add_vline(
        x=mean_length,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_length:.0f}",
        annotation_position="top"
    )
    
    return fig


def plot_retrieval_scores(
    documents: List[str],
    scores: List[float],
    title: str = "Retrieval Scores"
) -> go.Figure:
    """
    Create a bar chart of retrieval scores.
    
    Args:
        documents: Document snippets
        scores: Similarity/distance scores
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Truncate long documents for display
    labels = [d[:50] + "..." if len(d) > 50 else d for d in documents]
    
    # Convert distances to similarity (if needed)
    # Lower distance = higher similarity
    similarities = [1 / (1 + s) for s in scores]
    
    fig = go.Figure(data=[go.Bar(
        x=similarities,
        y=labels,
        orientation='h',
        marker_color='#667eea',
        text=[f"{s:.3f}" for s in similarities],
        textposition='outside'
    )])
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Similarity Score",
        yaxis_title="Documents",
        template="plotly_white",
        height=max(300, len(documents) * 50),
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def plot_rag_pipeline() -> go.Figure:
    """
    Create a visual representation of the RAG pipeline.
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Pipeline steps
    steps = [
        ("📝 Query", 0, "#667eea"),
        ("🔢 Embed", 1, "#764ba2"),
        ("🔍 Retrieve", 2, "#f093fb"),
        ("📄 Context", 3, "#f5576c"),
        ("🤖 Generate", 4, "#4facfe")
    ]
    
    # Add boxes
    for label, x, color in steps:
        fig.add_shape(
            type="rect",
            x0=x-0.4, y0=-0.3, x1=x+0.4, y1=0.3,
            fillcolor=color,
            line=dict(color=color, width=2),
            layer="below"
        )
        fig.add_annotation(
            x=x, y=0,
            text=label,
            showarrow=False,
            font=dict(size=14, color="white")
        )
    
    # Add arrows
    for i in range(len(steps) - 1):
        fig.add_annotation(
            x=i+0.5, y=0,
            ax=i+0.45, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowcolor="#333"
        )
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.5, 4.5]),
        yaxis=dict(visible=False, range=[-0.5, 0.5]),
        height=150,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig


def plot_vector_comparison(
    vector1: np.ndarray,
    vector2: np.ndarray,
    labels: tuple = ("Vector 1", "Vector 2"),
    n_dims: int = 20
) -> go.Figure:
    """
    Create a comparison of two embedding vectors.
    
    Args:
        vector1: First vector
        vector2: Second vector
        labels: Labels for the vectors
        n_dims: Number of dimensions to display
        
    Returns:
        Plotly figure
    """
    # Take first n dimensions
    v1 = vector1[:n_dims]
    v2 = vector2[:n_dims]
    dims = list(range(n_dims))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=dims,
        y=v1,
        name=labels[0],
        marker_color='#667eea',
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        x=dims,
        y=v2,
        name=labels[1],
        marker_color='#f5576c',
        opacity=0.7
    ))
    
    fig.update_layout(
        title=dict(text=f"Vector Comparison (first {n_dims} dimensions)", x=0.5),
        xaxis_title="Dimension Index",
        yaxis_title="Value",
        barmode='group',
        template="plotly_white",
        height=400
    )
    
    return fig
