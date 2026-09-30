import hashlib
from pathlib import Path

import pymupdf
import chromadb
from sentence_transformers import SentenceTransformer

DOCUMENTS_PATH = Path("documents")
DB_PATH = "chroma_db"
COLLECTION_NAME = "rag_documents"

pdf_paths = sorted(DOCUMENTS_PATH.glob("*.pdf"))
if not pdf_paths:
    raise SystemExit(f"No PDF files found in {DOCUMENTS_PATH}")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=DB_PATH)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
existing = collection.get(include=["metadatas"])
legacy_ids = [
    chunk_id
    for chunk_id, metadata in zip(existing["ids"], existing["metadatas"])
    if chunk_id.startswith("chunk_") and not metadata
]
if legacy_ids:
    collection.delete(ids=legacy_ids)

for pdf_path in pdf_paths:
    source = pdf_path.relative_to(DOCUMENTS_PATH).as_posix()
    print(f"Reading {source}...", flush=True)
    with pymupdf.open(pdf_path) as doc:
        page_count = len(doc)
        full_text = "\n".join(page.get_text() for page in doc)

    if not full_text.strip():
        print(
            f"Skipped {source}: no searchable text found across {page_count} pages. "
            "This appears to be a scanned PDF and needs OCR before it can be indexed."
        )
        continue

    chunks = []
    start = 0
    while start < len(full_text):
        chunk = full_text[start : start + 500].strip()
        if chunk:
            chunks.append(chunk)
        start += 400

    source_id = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    print(f"Extracted text from {page_count} pages; embedding {len(chunks)} chunks...", flush=True)
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()
    collection.delete(where={"source": source})
    collection.upsert(
        ids=[f"{source_id}_{index}" for index in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": source} for _ in chunks],
    )
    print(f"Indexed {len(chunks)} chunks from {source}")

print("Vector database updated successfully!")
