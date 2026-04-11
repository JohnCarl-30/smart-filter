import os
from pathlib import Path
from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent

# Support both the documented repo-root `.env` and the currently used
# `backend/.env` without overriding already exported shell variables.
for env_file in (PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"):
    load_dotenv(env_file, override=False)

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM: int = 384
INTENT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")

QDRANT_ENDPOINT: str = os.getenv("QDRANT_ENDPOINT", "")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME: str = "properties"
PROPERTY_QA_COLLECTION_NAME: str = "property_qa"
PROPERTY_QA_DENSE_VECTOR_NAME: str = "dense"
PROPERTY_QA_SPARSE_VECTOR_NAME: str = "sparse"
PROPERTY_QA_SPARSE_MODEL: str = "Qdrant/bm25"
PROPERTY_QA_DENSE_DIM: int = EMBEDDING_DIM

PROPERTIES_FILE: Path = PROJECT_ROOT / "data" / "properties.json"

WEIGHTS = {
    "semantic": 0.40,
    "filter":   0.30,
    "quality":  0.15,
    "memory":   0.15,
}

QDRANT_TOP_K: int = 30
FINAL_TOP_K: int = 10
