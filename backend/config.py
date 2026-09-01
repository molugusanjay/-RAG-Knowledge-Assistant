import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploaded_documents"
DATA_DIR = BASE_DIR / "vector_data"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CHUNK_SIZE = 500  # Characters per chunk
DEFAULT_CHUNK_OVERLAP = 100  # Overlap characters
DEFAULT_TOP_K = 4  # Top-K matching chunks
DEFAULT_SIMILARITY_THRESHOLD = 0.1  # Min similarity score filter

# LLM Config
DEFAULT_MODEL = "gemini-3.6-flash"
