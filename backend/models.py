from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class DocumentMetadata(BaseModel):
    id: str
    filename: str
    file_size: int
    file_type: str
    chunk_count: int
    uploaded_at: str

class ChunkInfo(BaseModel):
    id: str
    doc_id: str
    doc_name: str
    page_number: Optional[int] = 1
    content: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchResultChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: Optional[int] = 1
    content: str
    similarity_score: float

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4
    similarity_threshold: Optional[float] = 0.05
    api_key: Optional[str] = None
    model_name: Optional[str] = "gemini-3.6-flash"
    custom_system_prompt: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResultChunk]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    model_used: str
    has_api_key: bool

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.0

class SettingsUpdateRequest(BaseModel):
    chunk_size: Optional[int] = 500
    chunk_overlap: Optional[int] = 100
