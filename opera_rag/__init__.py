"""OpERA RAG — ask questions about PhilSA OpERA repositories and get cited answers."""
from .rag import OperaRAG, DEFAULT_REPOS, DEFAULT_EMBED, DEFAULT_LLM
from .index import HybridIndex
from .loaders import load_repo, Chunk

__all__ = ["OperaRAG", "HybridIndex", "load_repo", "Chunk",
           "DEFAULT_REPOS", "DEFAULT_EMBED", "DEFAULT_LLM"]
