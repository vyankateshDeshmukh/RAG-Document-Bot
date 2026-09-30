import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "chroma_db"
COLLECTION_NAME = "rag_documents"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

question = input("Ask a question about your PDFs: ").strip()
if not question:
    raise SystemExit("Please enter a question.")

result_count = min(3, collection.count())
if result_count == 0:
    raise SystemExit("No PDF chunks are indexed. Run build_vector_db.py first.")

# Convert question to embedding
question_embedding = model.encode([question]).tolist()

# Search ChromaDB
results = collection.query(
    query_embeddings=question_embedding,
    n_results=result_count
)

print("\nQUESTION:")
print(question)

print("\nRELEVANT PASSAGES (use these excerpts to answer):")
for i, (document, metadata) in enumerate(
    zip(results["documents"][0], results["metadatas"][0]), start=1
):
    print(f"\n--- Result {i} ---")
    source = metadata.get("source", "unknown (previously indexed data)") if metadata else "unknown (previously indexed data)"
    print(f"Source: {source}")
    print(document)
