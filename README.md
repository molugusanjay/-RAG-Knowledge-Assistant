# 🚀 NexusRAG - RAG-Based Knowledge Assistant

> **Internship Showcase Project**  
> **Difficulty Level**: Medium / Advanced Full-Stack AI Application  
> **Domain**: Artificial Intelligence / Natural Language Processing / Full-Stack Web Development  

---

## 📌 Project Overview
**NexusRAG** is a full-stack Retrieval-Augmented Generation (RAG) Knowledge Assistant that enables users to upload custom documents (PDF, DOCX, TXT, MD), convert unstructured text into high-dimensional vector embeddings, store them in a vector index with hybrid search, and ask questions through an interactive conversational web dashboard.

The system combines **Hybrid Retrieval** (TF-IDF Cosine Similarity + BM25 Keyword Boosting) with **Google Gemini 2.5 Flash** for LLM generation, along with a zero-dependency **Local Extractive QA Engine** for offline usage without API keys.

---

## ✨ Key Features

1. **Multi-Format Document Vault**:
   - Parses `.pdf`, `.docx`, `.txt`, and `.md` files.
   - Live drag-and-drop uploader with document stats (file size, chunk counts).

2. **Recursive Character Chunking Engine**:
   - Dynamically splits text into overlapping character windows (e.g. 500 characters with 100 character overlap).
   - Preserves sentence and paragraph boundaries to retain semantic context.

3. **Hybrid Vector Store & Search**:
   - Evaluates TF-IDF vector similarity combined with keyword matching.
   - Computes distance metrics and returns Top-K relevant document passages.

4. **Conversational RAG & Source Citation Accordion**:
   - Generates answers anchored strictly in retrieved ground-truth context.
   - Provides expandable **Source Citations** showing exact document name, page number, and similarity match percentage.

5. **Vector DB Explorer & Hyperparameter Tuning**:
   - Visual inspection tool to browse all indexed text chunks across the knowledge base.
   - Real-time sliders to adjust chunk size, overlap, Top-K, and similarity cutoffs.

6. **Zero-Dependency Offline Mode & Gemini LLM Integration**:
   - Works 100% offline out-of-the-box using the local extractive QA engine.
   - Connects to Google Gemini API (`google-genai`) when an API key is entered in the UI settings.

---

## 🛠️ Technology Stack

- **Backend Framework**: Python 3.14 + FastAPI + Uvicorn
- **Document Extractors**: `pypdf`, `python-docx`
- **Vector Search Engine**: `scikit-learn` (TF-IDF Vectorizer + Cosine Similarity) + NumPy
- **Generative AI SDK**: `google-genai` (Google Gemini 2.5 Flash integration)
- **Frontend UI**: Single Page Web Dashboard built with HTML5, Vanilla CSS3 (Glassmorphism, Dark Mode), and Vanilla JavaScript
- **API Protocol**: RESTful JSON over HTTP

---

## 📂 Project Structure

```
rag-knowledge-assistant/
├── backend/
│   ├── __init__.py
│   ├── config.py              # Application settings & hyperparameter defaults
│   ├── document_loader.py     # PDF/DOCX/TXT loader & recursive text chunker
│   ├── vector_store.py        # Vector index, hybrid search & disk state persistence
│   ├── rag_engine.py          # Prompt builder, Gemini integration & local QA fallback
│   └── models.py              # Pydantic schemas for REST API endpoints
├── frontend/
│   ├── index.html             # Dashboard UI (Chat, Vector Explorer, Analytics, Settings)
│   ├── styles.css             # Glassmorphism aesthetic, dark mode theme & animations
│   └── app.js                 # Frontend state, API bindings & chat stream handler
├── sample_documents/
│   ├── Generative_AI_Overview.txt
│   └── Company_Policies_Handbook.txt
├── main.py                    # FastAPI app declaration & REST endpoints
├── run.py                     # One-click startup script (auto pip install + browser launcher)
├── requirements.txt           # Python dependency manifest
└── README.md                  # Comprehensive internship documentation
```

---

## ⚡ Quick Start Guide

### Step 1: Clone / Navigate to Project Directory
```bash
cd C:\Users\molug\.gemini\antigravity\scratch\rag-knowledge-assistant
```

### Step 2: Run the One-Click Launcher
```bash
python run.py
```

*The launcher script will automatically:*
1. Verify Python dependencies and install missing packages from `requirements.txt`.
2. Auto-ingest sample demo documents into the Vector Store.
3. Start the FastAPI backend server at `http://127.0.0.1:8000`.
4. Open your default web browser to `http://127.0.0.1:8000`.

---

## 📡 REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/stats` | `GET` | Returns total documents, chunks, and index state metrics |
| `/api/documents` | `GET` | Lists all uploaded documents in the knowledge base |
| `/api/documents/upload` | `POST` | Uploads file (PDF/DOCX/TXT), splits into chunks, and updates vector index |
| `/api/documents/{doc_id}` | `DELETE` | Removes document and purges vector chunks |
| `/api/query` | `POST` | Executes full RAG pipeline (vector search + context synthesis + answer generation) |
| `/api/search` | `POST` | Executes standalone vector search for testing similarity matches |
| `/api/chunks` | `GET` | Lists raw chunks stored in the vector index |
| `/api/documents/clear` | `POST` | Resets and clears the entire knowledge base |

---

## 🎓 Internship Presentation Guide

When presenting this project to your internship evaluator:

1. **Explain RAG Architecture**: Point out how RAG prevents hallucination by retrieving relevant document passages *before* feeding context to the language model.
2. **Demonstrate Source Citations**: Show how every answer rendered in the Chat UI includes expandable source tags showing document title, page number, and similarity score.
3. **Showcase Vector DB Explorer**: Navigate to the *Vector DB Explorer* tab to show how document chunks are split, indexed, and retrieved.
4. **Highlight Offline & LLM Modes**: Demonstrate that the project works without API key dependencies, but seamlessly connects to Google Gemini for full AI generation.
