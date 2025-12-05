# RAG Walkthrough Application

A comprehensive, interactive Streamlit application for understanding Retrieval-Augmented Generation (RAG) concepts through theoretical explanations and hands-on demonstrations.

## Overview

This educational application provides a complete walkthrough of RAG systems, covering the fundamental concepts, practical implementations, and best practices. It includes interactive demos for each component of the RAG pipeline, from text embeddings to LLM-based generation.

## Features

- **Multi-page Structure**: Eight dedicated pages covering all aspects of RAG
- **Interactive Demonstrations**: Hands-on experiments with embeddings, chunking, and retrieval
- **Vector Database Integration**: ChromaDB playground for practical experience
- **LLM Generation**: Groq integration with Llama 3.3 70B for real-time responses
- **Visualization Tools**: Embedding plots, similarity matrices, and retrieval score charts

## Requirements

- Python 3.10 - 3.12 (recommended: 3.12)
- Groq API key (free tier available at https://console.groq.com)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "BDSA 2025"
```

2. Create and activate a virtual environment:
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Project Structure

```
.
├── app.py                    # Main entry point
├── requirements.txt          # Python dependencies
├── pages/
│   ├── 1_Introduction_to_RAG.py
│   ├── 2_Understanding_Embeddings.py
│   ├── 3_Vector_Databases.py
│   ├── 4_Document_Processing.py
│   ├── 5_Retrieval_Mechanisms.py
│   ├── 6_Generation_with_Context.py
│   ├── 7_Complete_RAG_Pipeline.py
│   └── 8_Best_Practices.py
├── utils/
│   ├── embeddings.py         # Sentence transformer utilities
│   ├── chunking.py           # Text chunking strategies
│   ├── vector_store.py       # ChromaDB wrapper
│   ├── visualization.py      # Plotly chart functions
│   └── llm.py                # Groq LLM integration
└── data/
    └── sample_documents.txt  # Sample content for demos
```

## Application Pages

| Page | Topic | Description |
|------|-------|-------------|
| 1 | Introduction to RAG | Overview of RAG architecture and use cases |
| 2 | Understanding Embeddings | Text vectorization and semantic similarity |
| 3 | Vector Databases | ChromaDB operations and indexing algorithms |
| 4 | Document Processing | Chunking strategies and preprocessing |
| 5 | Retrieval Mechanisms | Similarity search and ranking methods |
| 6 | Generation with Context | Prompt engineering and LLM integration |
| 7 | Complete RAG Pipeline | End-to-end interactive demonstration |
| 8 | Best Practices | Optimization tips and common pitfalls |

## Technologies

- **Streamlit**: Web application framework
- **ChromaDB**: Vector database for embedding storage and retrieval
- **Sentence Transformers**: Text embedding generation (all-MiniLM-L6-v2)
- **Groq**: LLM API for text generation (Llama 3.3 70B Versatile)
- **Plotly**: Interactive data visualizations
- **scikit-learn**: Dimensionality reduction for embedding visualization

## Usage

1. Navigate through the pages in order for a structured learning experience
2. Each page contains theoretical explanations followed by interactive demos
3. On the Complete RAG Pipeline page, enter your Groq API key to enable LLM generation
4. Experiment with different parameters to understand their effects

## API Keys

The application requires a Groq API key for LLM generation features. The key is entered directly in the application interface and is not stored.

To obtain a free API key:
1. Visit https://console.groq.com
2. Create an account
3. Generate an API key from the dashboard

## License

This project is provided for educational purposes.

## Acknowledgments

- Hugging Face for the Sentence Transformers library
- Chroma for the open-source vector database
- Groq for the fast LLM inference API
