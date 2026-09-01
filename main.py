import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.config import BASE_DIR, UPLOAD_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from backend.models import (
    DocumentMetadata,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResultChunk,
)
from backend.document_loader import DocumentLoader
from backend.vector_store import VectorStore
from backend.rag_engine import RAGEngine

app = FastAPI(
    title="RAG-Based Knowledge Assistant API",
    description="Full-stack Internship Project API for Document Ingestion, Chunking, Hybrid Vector Search, and RAG Query Answering.",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Services
vector_store = VectorStore()
rag_engine = RAGEngine(vector_store=vector_store)

# Global settings state
app_settings = {
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
}


@app.get("/api/stats")
async def get_stats():
    """Returns vector store and knowledge base system metrics."""
    stats = vector_store.get_stats()
    stats["current_chunk_size"] = app_settings["chunk_size"]
    stats["current_chunk_overlap"] = app_settings["chunk_overlap"]
    return stats


@app.get("/api/documents")
async def list_documents():
    """Returns list of all uploaded documents."""
    return vector_store.get_all_documents()


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
):
    """
    Uploads document (PDF, DOCX, TXT, MD), parses content, extracts text,
    splits into recursive chunks, and indexes into the Vector Store.
    """
    allowed_extensions = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".json"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {', '.join(allowed_extensions)}",
        )

    doc_id = str(uuid.uuid4())[:8]
    save_filename = f"{doc_id}_{file.filename}"
    file_path = UPLOAD_DIR / save_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(file_path)

        # Parse pages
        pages = DocumentLoader.load_document(str(file_path), file.filename)

        c_size = chunk_size or app_settings["chunk_size"]
        c_overlap = chunk_overlap or app_settings["chunk_overlap"]

        # Generate chunks
        chunks = DocumentLoader.create_chunks(
            pages=pages,
            doc_id=doc_id,
            doc_name=file.filename,
            chunk_size=c_size,
            chunk_overlap=c_overlap,
        )

        # Fallback if no text extracted (e.g. empty or non-text document)
        if not chunks:
            chunks = [{
                "id": f"{doc_id}_chunk_0",
                "doc_id": doc_id,
                "doc_name": file.filename,
                "page_number": 1,
                "content": f"[Document {file.filename} uploaded - File contains empty or custom non-standard text formatting]",
                "chunk_index": 0,
                "metadata": {"char_len": 0, "word_count": 0}
            }]

        doc_metadata = {
            "id": doc_id,
            "filename": file.filename,
            "saved_name": save_filename,
            "file_size": file_size,
            "file_type": file_ext.replace(".", "").upper(),
            "chunk_count": len(chunks),
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Index in Vector Store
        vector_store.add_document(doc_metadata, chunks)

        return {
            "message": "Document uploaded and indexed successfully",
            "document": doc_metadata,
            "chunks_created": len(chunks),
        }

    except Exception as e:
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Deletes document and purges its vector embeddings."""
    docs = vector_store.get_all_documents()
    target_doc = next((d for d in docs if d["id"] == doc_id), None)

    if not target_doc:
        raise HTTPException(status_code=404, detail="Document not found")

    saved_name = target_doc.get("saved_name")
    if saved_name:
        file_path = UPLOAD_DIR / saved_name
        if file_path.exists():
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Failed to delete raw file {file_path}: {e}")

    vector_store.delete_document(doc_id)
    return {"message": "Document deleted successfully", "doc_id": doc_id}


@app.post("/api/documents/clear")
async def clear_documents():
    """Clears all uploaded documents and resets vector store."""
    vector_store.clear_all()
    # Remove files in upload dir
    for f in UPLOAD_DIR.glob("*"):
        if f.is_file():
            try:
                os.remove(f)
            except Exception as e:
                print(f"Error removing file {f}: {e}")

    return {"message": "Knowledge base reset successfully"}


@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Executes full RAG workflow: Vector Retrieval + Context Injection + LLM Response."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty")

    response = rag_engine.query(
        user_query=request.query,
        top_k=request.top_k or 4,
        similarity_threshold=request.similarity_threshold or 0.05,
        api_key=request.api_key,
        model_name=request.model_name or "gemini-3.6-flash",
        custom_system_prompt=request.custom_system_prompt,
    )
    return response


@app.post("/api/search")
async def search_vectors(request: SearchRequest):
    """Executes standalone vector similarity search for vector store visual inspection."""
    matching_chunks = vector_store.search(
        query=request.query,
        top_k=request.top_k or 5,
        similarity_threshold=request.similarity_threshold or 0.0,
    )
    return {
        "query": request.query,
        "count": len(matching_chunks),
        "results": matching_chunks,
    }


@app.get("/api/chunks")
async def list_chunks(doc_id: Optional[str] = None):
    """Returns vector chunks, optionally filtered by document ID."""
    all_chunks = vector_store.get_all_chunks()
    if doc_id:
        all_chunks = [c for c in all_chunks if c["doc_id"] == doc_id]
    return {"total": len(all_chunks), "chunks": all_chunks}


@app.post("/api/settings")
async def update_settings(chunk_size: int = Form(500), chunk_overlap: int = Form(100)):
    """Updates default chunking hyperparameters."""
    app_settings["chunk_size"] = chunk_size
    app_settings["chunk_overlap"] = chunk_overlap
    return {"message": "Settings updated", "settings": app_settings}


# Mount static web frontend files
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(frontend_dir / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
