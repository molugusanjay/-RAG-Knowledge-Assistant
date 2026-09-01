import time
import os
from typing import List, Dict, Any, Tuple
from backend.vector_store import VectorStore
from backend.models import SearchResultChunk, QueryResponse

# Import google-genai safely
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class RAGEngine:
    """RAG Orchestrator handling retrieval, prompt synthesis, and LLM generation."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def query(
        self,
        user_query: str,
        top_k: int = 4,
        similarity_threshold: float = 0.05,
        api_key: str = None,
        model_name: str = "gemini-2.5-flash",
        custom_system_prompt: str = None,
    ) -> QueryResponse:
        start_total = time.time()

        # Step 1: Retrieval
        t_retrieval_start = time.time()
        matching_chunks = self.vector_store.search(
            query=user_query, top_k=top_k, similarity_threshold=similarity_threshold
        )
        retrieval_time_ms = round((time.time() - t_retrieval_start) * 1000, 2)

        # Convert to Pydantic objects
        source_chunks = [
            SearchResultChunk(
                chunk_id=c["id"],
                doc_id=c["doc_id"],
                doc_name=c["doc_name"],
                page_number=c.get("page_number", 1),
                content=c["content"],
                similarity_score=c["similarity_score"],
            )
            for c in matching_chunks
        ]

        # Step 2: Generation
        t_gen_start = time.time()
        has_api_key = bool(api_key and api_key.strip())

        if not matching_chunks:
            answer = (
                "I searched the uploaded document knowledge base, but I couldn't find any "
                "relevant context matching your query. Please upload relevant documents or adjust your query terms."
            )
            model_used = "Knowledge Base Search (No Matches)"
        elif has_api_key and GENAI_AVAILABLE:
            answer, model_used = self._generate_with_gemini(
                query=user_query,
                matching_chunks=matching_chunks,
                api_key=api_key.strip(),
                model_name=model_name,
                custom_prompt=custom_system_prompt,
            )
        else:
            answer, model_used = self._generate_extractive_fallback(
                query=user_query, matching_chunks=matching_chunks
            )

        gen_time_ms = round((time.time() - t_gen_start) * 1000, 2)
        total_time_ms = round((time.time() - start_total) * 1000, 2)

        return QueryResponse(
            query=user_query,
            answer=answer,
            sources=source_chunks,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=gen_time_ms,
            total_time_ms=total_time_ms,
            model_used=model_used,
            has_api_key=has_api_key,
        )

    def _generate_with_gemini(
        self,
        query: str,
        matching_chunks: List[Dict[str, Any]],
        api_key: str,
        model_name: str,
        custom_prompt: str = None,
    ) -> Tuple[str, str]:
        """Calls Google Gemini API using google-genai SDK with automatic model version fallbacks."""
        candidate_models = []
        if model_name:
            candidate_models.append(model_name)
        for fallback in ["gemini-3.6-flash", "gemini-1.5-flash", "gemini-2.0-flash", "gemini-flash"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None

        # Build context string
        context_blocks = []
        for idx, c in enumerate(matching_chunks, 1):
            context_blocks.append(
                f"--- Source [{idx}] (Document: {c['doc_name']}, Page: {c.get('page_number', 1)}) ---\n"
                f"{c['content']}"
            )
        formatted_context = "\n\n".join(context_blocks)

        sys_instruction = (
            custom_prompt
            or "You are an intelligent RAG (Retrieval-Augmented Generation) Assistant. "
            "Answer the user's question accurately and concisely based ONLY on the provided context sources. "
            "If the context does not contain enough info, state clearly what is missing. "
            "Cite sources using [Source X] inline tags when making factual statements."
        )

        prompt = (
            f"### CONTEXT DOCUMENTS:\n{formatted_context}\n\n"
            f"### USER QUESTION:\n{query}\n\n"
            f"### ANSWER:"
        )

        for target_model in candidate_models:
            try:
                client = genai.Client(api_key=api_key)

                config = types.GenerateContentConfig(
                    system_instruction=sys_instruction,
                    temperature=0.2,
                )

                response = client.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=config,
                )

                if response and response.text:
                    return response.text.strip(), f"Google Gemini ({target_model})"
            except Exception as e:
                last_error = str(e)
                print(f"Model '{target_model}' error: {e}. Trying next candidate...")

        # If all API candidates failed, fallback to local extractive QA
        error_msg = f"Gemini API Error: {last_error}. Falling back to Local Extractive QA Engine."
        fallback_ans, _ = self._generate_extractive_fallback(query, matching_chunks)
        return f"[Warning] {error_msg}\n\n{fallback_ans}", "Gemini API (Error Fallback)"

    def _generate_extractive_fallback(
        self, query: str, matching_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Smart offline extractive synthesizer when running without an API key.
        Extracts key sentences matching the query from retrieved context chunks.
        """
        lines = [
            f"Here is the summary extracted from **{len(matching_chunks)} relevant document passages** in your knowledge base:\n"
        ]

        for idx, chunk in enumerate(matching_chunks, 1):
            doc_name = chunk["doc_name"]
            page_num = chunk.get("page_number", 1)
            score = chunk.get("similarity_score", 0.0)
            content = chunk["content"].strip()

            lines.append(
                f"**Key Findings from {doc_name} (Page {page_num})** - *{int(score * 100)}% match*"
            )

            # Extract bullet sentences
            sentences = [s.strip() for s in content.split("\n") if s.strip()]
            for s in sentences[:4]:
                if s.startswith("-") or s.startswith("*") or s.startswith("1.") or s.startswith("2."):
                    lines.append(f"• {s}")
                else:
                    lines.append(f"• {s[:160]}..." if len(s) > 160 else f"• {s}")
            lines.append("")

        return "\n".join(lines), "Local RAG Extractive Engine (Offline Ready)"
