import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Try importing pypdf and docx safely
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None


class DocumentLoader:
    """Parses PDF, DOCX, TXT, and MD files and performs recursive character chunking."""

    @staticmethod
    def load_document(file_path: str, filename: str) -> List[Dict[str, Any]]:
        """
        Reads document and returns list of page/section dicts:
        [{ "content": str, "page_number": int }]
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            return DocumentLoader._load_pdf(file_path)
        elif ext in [".docx", ".doc"]:
            return DocumentLoader._load_docx(file_path)
        elif ext in [".txt", ".md", ".csv", ".json"]:
            return DocumentLoader._load_text(file_path)
        else:
            # Fallback text loader
            return DocumentLoader._load_text(file_path)

    @staticmethod
    def _load_pdf(file_path: str) -> List[Dict[str, Any]]:
        pages = []
        if pypdf:
            try:
                reader = pypdf.PdfReader(file_path)
                for idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append({"content": text.strip(), "page_number": idx + 1})
            except Exception as e:
                print(f"Error parsing PDF with pypdf: {e}")

        # Fallback if pypdf extracts nothing or is not installed
        if not pages:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                pages.append({"content": content, "page_number": 1})

        return pages

    @staticmethod
    def _load_docx(file_path: str) -> List[Dict[str, Any]]:
        sections = []
        if docx:
            try:
                doc = docx.Document(file_path)
                full_text = []
                for p in doc.paragraphs:
                    if p.text.strip():
                        full_text.append(p.text.strip())
                content = "\n".join(full_text)
                sections.append({"content": content, "page_number": 1})
                return sections
            except Exception as e:
                print(f"Error parsing DOCX: {e}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            sections.append({"content": content, "page_number": 1})
        return sections

    @staticmethod
    def _load_text(file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [{"content": content, "page_number": 1}]

    @staticmethod
    def create_chunks(
        pages: List[Dict[str, Any]],
        doc_id: str,
        doc_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Splits pages into overlapping text chunks with semantic boundaries (paragraphs/sentences).
        """
        chunks = []
        global_chunk_idx = 0

        for page in pages:
            page_num = page.get("page_number", 1)
            raw_text = page.get("content", "")

            if not raw_text.strip():
                continue

            # Split text by paragraphs / sentences recursively
            page_chunks = DocumentLoader._recursive_split(
                raw_text, chunk_size, chunk_overlap
            )

            for chunk_text in page_chunks:
                if len(chunk_text.strip()) < 10:
                    continue

                chunk_id = f"{doc_id}_chunk_{global_chunk_idx}"
                chunks.append(
                    {
                        "id": chunk_id,
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "page_number": page_num,
                        "content": chunk_text.strip(),
                        "chunk_index": global_chunk_idx,
                        "metadata": {
                            "char_len": len(chunk_text),
                            "word_count": len(chunk_text.split()),
                        },
                    }
                )
                global_chunk_idx += 1

        return chunks

    @staticmethod
    def _recursive_split(
        text: str, chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        """Splits text into chunks using recursive separators (paragraphs, sentences, spaces)."""
        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        return DocumentLoader._split_text(text, separators, chunk_size, chunk_overlap)

    @staticmethod
    def _split_text(
        text: str, separators: List[str], chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        final_chunks = []
        if len(text) <= chunk_size or not separators:
            return [text]

        sep = separators[0]
        splits = text.split(sep) if sep else list(text)

        current_chunk = ""
        for s in splits:
            item = s + sep if sep else s
            if len(current_chunk) + len(item) <= chunk_size:
                current_chunk += item
            else:
                if current_chunk.strip():
                    final_chunks.append(current_chunk.strip())

                # If single split item itself exceeds chunk size, recurse down separators
                if len(item) > chunk_size and len(separators) > 1:
                    sub_chunks = DocumentLoader._split_text(
                        item, separators[1:], chunk_size, chunk_overlap
                    )
                    final_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    # Apply overlap from trailing part of previous chunk
                    overlap_start = max(0, len(current_chunk) - chunk_overlap)
                    overlap_str = current_chunk[overlap_start:]
                    current_chunk = overlap_str + item

        if current_chunk.strip():
            final_chunks.append(current_chunk.strip())

        return final_chunks
