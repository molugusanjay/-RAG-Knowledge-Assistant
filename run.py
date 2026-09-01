import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def print_banner():
    print("=" * 70)
    print(" 🚀 NexusRAG - RAG-Based Knowledge Assistant (Internship Project)")
    print("=" * 70)
    print(" Starting up background service & checking dependencies...")

def check_and_install_dependencies():
    req_file = BASE_DIR / "requirements.txt"
    if not req_file.exists():
        return

    print(" 📦 Verifying Python package dependencies...")
    try:
        import fastapi
        import uvicorn
        import sklearn
        import pypdf
        import docx
        print(" ✅ All required dependencies are present!")
    except ImportError:
        print(" ⏳ Missing dependencies detected. Installing via pip...")
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
        subprocess.check_call(cmd)
        print(" ✅ Dependencies installed successfully!")

def load_sample_docs_if_empty():
    """Auto-populates sample documents if none exist in knowledge base."""
    from backend.vector_store import VectorStore
    from backend.document_loader import DocumentLoader
    import uuid

    store = VectorStore()
    if len(store.get_all_documents()) == 0:
        print(" 📄 Knowledge Base is currently empty. Auto-ingesting sample demo documents...")
        samples_dir = BASE_DIR / "sample_documents"
        if samples_dir.exists():
            for sample_file in samples_dir.glob("*.txt"):
                doc_id = str(uuid.uuid4())[:8]
                pages = DocumentLoader.load_document(str(sample_file), sample_file.name)
                chunks = DocumentLoader.create_chunks(
                    pages=pages, doc_id=doc_id, doc_name=sample_file.name, chunk_size=500, chunk_overlap=100
                )
                metadata = {
                    "id": doc_id,
                    "filename": sample_file.name,
                    "saved_name": sample_file.name,
                    "file_size": sample_file.stat().st_size,
                    "file_type": "TXT",
                    "chunk_count": len(chunks),
                    "uploaded_at": "Demo Sample Data",
                }
                store.add_document(metadata, chunks)
                print(f"    - Ingested '{sample_file.name}' ({len(chunks)} chunks)")
            print(" ✅ Demo documents loaded into Vector Store!")

def open_browser():
    time.sleep(1.5)
    url = "http://127.0.0.1:8000"
    print(f" 🌐 Opening Web Dashboard at {url}...")
    webbrowser.open(url)

if __name__ == "__main__":
    print_banner()
    check_and_install_dependencies()
    load_sample_docs_if_empty()

    print("\n 🔥 Booting Uvicorn FastAPI Server on http://127.0.0.1:8000...")
    
    # Launch browser thread
    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
