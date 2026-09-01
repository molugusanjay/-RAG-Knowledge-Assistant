import os
import re
import math
import json
from collections import Counter
from typing import List, Dict, Any, Tuple
from pathlib import Path
from backend.config import DATA_DIR


class PurePythonTFIDF:
    """
    Zero-dependency, high-speed Pure Python TF-IDF Vectorizer and Cosine Similarity engine.
    Ensures 100% cross-platform compatibility across all Python versions (3.8 - 3.14+).
    """

    def __init__(self):
        self.doc_count = 0
        self.df = Counter()
        self.idf = {}
        self.doc_vectors = []
        self.vocabulary = set()

    def _tokenize(self, text: str) -> List[str]:
        # Extract word tokens (lower case, alphanumeric)
        return re.findall(r'\b\w+\b', text.lower())

    def fit_transform(self, docs: List[str]):
        self.doc_count = len(docs)
        self.df = Counter()
        tokenized_docs = []

        for doc in docs:
            tokens = self._tokenize(doc)
            tokenized_docs.append(tokens)
            for token in set(tokens):
                self.df[token] += 1

        self.vocabulary = set(self.df.keys())

        # IDF with standard log smoothing: ln((1 + N) / (1 + df)) + 1
        self.idf = {
            term: math.log((1 + self.doc_count) / (1 + count)) + 1.0
            for term, count in self.df.items()
        }

        # Build sparse TF-IDF vectors (dict representation)
        self.doc_vectors = []
        for tokens in tokenized_docs:
            tf = Counter(tokens)
            total_terms = max(1, len(tokens))
            vec = {
                term: (count / total_terms) * self.idf.get(term, 0.0)
                for term, count in tf.items()
            }
            self.doc_vectors.append(vec)

    def transform_query(self, query: str) -> Dict[str, float]:
        tokens = self._tokenize(query)
        if not tokens:
            return {}
        tf = Counter(tokens)
        total_terms = len(tokens)
        return {
            term: (count / total_terms) * self.idf.get(term, 0.0)
            for term, count in tf.items()
            if term in self.idf
        }

    @staticmethod
    def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        if not vec1 or not vec2:
            return 0.0

        common_terms = set(vec1.keys()).intersection(set(vec2.keys()))
        if not common_terms:
            return 0.0

        dot_product = sum(vec1[t] * vec2[t] for t in common_terms)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot_product / (norm1 * norm2)


class VectorStore:
    """
    In-memory hybrid Vector Store utilizing Pure Python TF-IDF Cosine Similarity
    and BM25/Keyword overlap boosting for accurate RAG retrieval.
    """

    def __init__(self, persistence_dir: Path = DATA_DIR):
        self.persistence_dir = persistence_dir
        self.chunks: List[Dict[str, Any]] = []
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.vectorizer = PurePythonTFIDF()
        self.is_fitted = False
        self._load_state()

    def add_document(self, doc_metadata: Dict[str, Any], chunks: List[Dict[str, Any]]):
        """Adds document metadata and text chunks to vector store and updates embeddings."""
        doc_id = doc_metadata["id"]
        # Remove existing if overwriting
        self.chunks = [c for c in self.chunks if c["doc_id"] != doc_id]

        self.documents[doc_id] = doc_metadata
        self.chunks.extend(chunks)

        self._rebuild_index()
        self._save_state()

    def delete_document(self, doc_id: str) -> bool:
        """Deletes document and its chunks from store."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.chunks = [c for c in self.chunks if c["doc_id"] != doc_id]
            self._rebuild_index()
            self._save_state()
            return True
        return False

    def clear_all(self):
        """Clears all stored documents and chunks."""
        self.documents = {}
        self.chunks = []
        self.is_fitted = False
        self._save_state()

    def _rebuild_index(self):
        """Re-fits TF-IDF index over all chunk contents."""
        if not self.chunks:
            self.is_fitted = False
            return

        corpus = [c["content"] for c in self.chunks]
        try:
            self.vectorizer.fit_transform(corpus)
            self.is_fitted = True
        except Exception as e:
            print(f"Error rebuilding vector index: {e}")
            self.is_fitted = False

    def search(
        self, query: str, top_k: int = 4, similarity_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Performs hybrid similarity search:
        1. Cosine similarity on TF-IDF vectors
        2. Keyword overlap boosting (BM25 heuristic)
        Returns top_k chunks matching query with similarity_score.
        """
        if not self.chunks or not self.is_fitted or not query.strip():
            return []

        try:
            query_vec = self.vectorizer.transform_query(query)
            query_words = set(re.findall(r'\b\w+\b', query.lower()))

            results = []

            for idx, chunk in enumerate(self.chunks):
                doc_vec = self.vectorizer.doc_vectors[idx]
                raw_cosine = self.vectorizer.cosine_similarity(query_vec, doc_vec)

                # Keyword overlap bonus
                content_words = set(re.findall(r'\b\w+\b', chunk["content"].lower()))
                overlap = len(query_words.intersection(content_words))
                keyword_boost = (overlap / max(1, len(query_words))) * 0.20

                total_score = min(1.0, raw_cosine + keyword_boost)

                if total_score >= similarity_threshold:
                    result_chunk = dict(chunk)
                    result_chunk["similarity_score"] = round(total_score, 4)
                    results.append(result_chunk)

            # Sort descending by score
            results.sort(key=lambda x: x["similarity_score"], reverse=True)

            # Meta-query detection (asking for summary, overview, or general info)
            meta_query_keywords = {
                "summarise", "summarize", "summary", "overview", "about",
                "document", "documents", "file", "files", "explain", "detail",
                "points", "main", "key", "content", "contents", "tl;dr", "tldr"
            }
            is_meta_query = any(w in meta_query_keywords for w in query_words)

            if (not results or is_meta_query) and self.chunks:
                # If no direct vector match found or asking general summary, include top initial chunks of documents
                fallback_results = []
                seen_chunk_ids = {r["id"] for r in results}

                # Prioritize first chunks of each document in knowledge base
                for doc_id in self.documents:
                    doc_chunks = [c for c in self.chunks if c["doc_id"] == doc_id]
                    for c in doc_chunks[:2]:
                        if c["id"] not in seen_chunk_ids:
                            res = dict(c)
                            res["similarity_score"] = round(res.get("similarity_score", 0.50), 4)
                            if res["similarity_score"] == 0:
                                res["similarity_score"] = 0.50
                            fallback_results.append(res)
                            seen_chunk_ids.add(c["id"])

                # Combine results
                combined = results + fallback_results
                return combined[:top_k]

            return results[:top_k]

        except Exception as e:
            print(f"Error executing vector search: {e}")
            return []

    def get_all_documents(self) -> List[Dict[str, Any]]:
        return list(self.documents.values())

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        return self.chunks

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "vocabulary_size": len(self.vectorizer.vocabulary) if self.is_fitted else 0,
            "is_indexed": self.is_fitted,
        }

    def _save_state(self):
        """Persists store metadata to disk."""
        try:
            state_file = self.persistence_dir / "store_state.json"
            data = {
                "documents": self.documents,
                "chunks": self.chunks,
            }
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving vector store state: {e}")

    def _load_state(self):
        """Loads state from disk if exists."""
        state_file = self.persistence_dir / "store_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = data.get("documents", {})
                    self.chunks = data.get("chunks", [])
                    self._rebuild_index()
            except Exception as e:
                print(f"Error loading vector store state: {e}")
