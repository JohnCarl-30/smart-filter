from __future__ import annotations
from qdrant_client import QdrantClient
from config import PROJECT_ROOT, QDRANT_ENDPOINT, QDRANT_API_KEY

_qdrant: QdrantClient | None = None


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        if QDRANT_ENDPOINT:
            _qdrant = QdrantClient(
                url=QDRANT_ENDPOINT,
                api_key=QDRANT_API_KEY,
            )
        else:
            storage_path = str(PROJECT_ROOT / "qdrant_storage")
            _qdrant = QdrantClient(path=storage_path)
    return _qdrant
