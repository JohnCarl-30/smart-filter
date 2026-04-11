from __future__ import annotations
import logging
import os
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

_model: TextEmbedding | None = None

def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        logger.info("Initializing FastEmbed (BAAI/bge-small-en-v1.5)...")
        _model = TextEmbedding()
    return _model

def embed(text: str) -> list[float]:
    """Generate dense vector using FastEmbed (BAAI/bge-small-en-v1.5) - 384 dimensions."""
    model = _get_model()
    embeddings = list(model.embed([text.replace("\n", " ")]))
    return embeddings[0].tolist()

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    model = _get_model()
    cleaned = [t.replace("\n", " ") for t in texts]
    embeddings = list(model.embed(cleaned))
    return [e.tolist() for e in embeddings]
