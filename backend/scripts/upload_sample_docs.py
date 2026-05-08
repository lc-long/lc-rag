"""Upload sample documents to the RAG system for testing.

Uses simple text splitting - no need to import the complex chunker.
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from dotenv import load_dotenv
load_dotenv(".env")

from src.core.storage.vector_store import init_pgvector, VectorStore, DocumentChunk
from src.core.embedder.aliyun_embedder import AliyunEmbedder
from src.api.database import SessionLocal, engine, Base
from src.api.models import Document
import uuid
from datetime import datetime, timezone

DOCUMENTS_DIR = Path(__file__).parent.parent.parent / "document"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def simple_chunk(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Simple fixed-size chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append({
            "content": chunk_text,
            "start": start,
            "end": end,
        })
        start = end - overlap
    return chunks


def upload_sample_docs():
    """Upload all markdown documents from the document directory."""
    print("Initializing pgvector...")
    init_pgvector()

    vector_store = VectorStore()
    embedder = AliyunEmbedder()

    session = SessionLocal()

    # Check existing
    existing = session.query(Document).count()
    print(f"Existing documents: {existing}")

    md_files = list(DOCUMENTS_DIR.glob("*.md"))
    print(f"Found {len(md_files)} markdown files to upload")

    for md_file in md_files:
        print(f"\nProcessing: {md_file.name}")

        # Read content
        content = md_file.read_text(encoding="utf-8")

        # Create document record
        doc_id = str(uuid.uuid4())
        doc = Document(
            id=doc_id,
            name=md_file.name,
            size=len(content),
            status="completed",
            created_at=datetime.now(timezone.utc),
        )
        session.add(doc)
        session.flush()  # Get doc_id

        # Split into chunks
        chunks = simple_chunk(content)
        print(f"  Split into {len(chunks)} chunks")

        # Prepare for embedding
        texts = [c["content"] for c in chunks]
        print(f"  Embedding {len(texts)} texts...")

        # Embed
        try:
            embeddings = embedder.embed(texts)
            print(f"  Embedded {len(embeddings)} texts")
        except Exception as e:
            print(f"  Embedding failed: {e}")
            session.rollback()
            continue

        # Store chunks with embeddings
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            embedding_vec = embeddings[i] if i < len(embeddings) else [0.0] * 1024

            db_chunk = DocumentChunk(
                id=chunk_id,
                doc_id=doc_id,
                chunk_type="parent",
                content=chunk["content"],
                embedding=embedding_vec,
                metadata_={
                    "source": md_file.name,
                    "page_number": 0,
                }
            )
            session.add(db_chunk)

        session.commit()
        print(f"  Stored {len(chunks)} chunks for doc {doc_id[:8]}...")

    total_docs = session.query(Document).count()
    total_chunks = session.query(DocumentChunk).count()
    print(f"\nTotal documents: {total_docs}")
    print(f"Total chunks: {total_chunks}")

    session.close()
    return total_docs, total_chunks


if __name__ == "__main__":
    upload_sample_docs()