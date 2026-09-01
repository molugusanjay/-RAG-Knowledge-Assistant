import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.vector_store import VectorStore
from backend.document_loader import DocumentLoader
from backend.rag_engine import RAGEngine

def test_rag():
    print("--- 1. Testing Document Loader ---")
    sample_file = Path("sample_documents/Generative_AI_Overview.txt")
    pages = DocumentLoader.load_document(str(sample_file), sample_file.name)
    chunks = DocumentLoader.create_chunks(pages, "doc_test_1", sample_file.name, chunk_size=500, chunk_overlap=100)
    print(f"Loaded {len(pages)} pages and generated {len(chunks)} chunks.")
    assert len(chunks) > 0, "Chunks creation failed!"

    print("\n--- 2. Testing Vector Store & Ingest ---")
    vstore = VectorStore()
    metadata = {
        "id": "doc_test_1",
        "filename": sample_file.name,
        "saved_name": sample_file.name,
        "file_size": 1000,
        "file_type": "TXT",
        "chunk_count": len(chunks),
        "uploaded_at": "Test",
    }
    vstore.add_document(metadata, chunks)
    stats = vstore.get_stats()
    print("Vector Store Stats:", stats)
    assert stats["total_chunks"] > 0, "Vector store indexing failed!"

    print("\n--- 3. Testing Hybrid Vector Search ---")
    results = vstore.search("What is Top-K Context Retrieval?", top_k=3)
    print(f"Search returned {len(results)} matches.")
    for idx, r in enumerate(results, 1):
        print(f" Match [{idx}] (Score: {r['similarity_score']}): {r['content'][:80]}...")
    assert len(results) > 0, "Vector search returned no results!"

    print("\n--- 4. Testing RAG QA Engine (Extractive Fallback) ---")
    engine = RAGEngine(vstore)
    response = engine.query(user_query="What is document chunking?", top_k=2)
    print(f"Model used: {response.model_used}")
    print(f"Retrieval time: {response.retrieval_time_ms} ms | Total time: {response.total_time_ms} ms")
    assert len(response.sources) > 0, "RAG sources missing!"

    print("\n--- 5. Testing Document Summary Query ('summarise my document') ---")
    summary_resp = engine.query(user_query="summarise my document", top_k=3)
    print(f"Summary Sources Count: {len(summary_resp.sources)}")
    print(f"Summary Answer Snippet:\n{summary_resp.answer[:300]}...")
    assert len(summary_resp.sources) > 0, "Summary query failed to retrieve document context!"

    print("\n[SUCCESS] ALL BACKEND RAG TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_rag()
