import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------
# 🔹 Configuration
# ---------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

KNOWLEDGE_DIR = "data/knowledge_docs"
INDEX_PATH = "models/faiss_index.index"
DOC_STORE_PATH = "models/doc_store.npy"

TOP_K_DEFAULT = 3


# ---------------------------------------------------
# 🔹 Load Embedding Model Once
# ---------------------------------------------------

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---------------------------------------------------
# 🔹 Build FAISS Index (L2 Distance)
# ---------------------------------------------------

def build_index():
    """
    Build FAISS index using L2 distance.
    Run once before starting API.
    """

    all_chunks = []

    # Load all .txt knowledge documents
    for filename in os.listdir(KNOWLEDGE_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(KNOWLEDGE_DIR, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = [
                chunk.strip()
                for chunk in text.split("\n\n")
                if chunk.strip()
            ]

            all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No knowledge documents found.")

    embeddings = embedding_model.encode(all_chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    np.save(DOC_STORE_PATH, np.array(all_chunks, dtype=object))

    print("✅ FAISS L2 index built and saved.")


# ---------------------------------------------------
# 🔹 Load Index Once at Startup
# ---------------------------------------------------

index = None
doc_store = None

def _load_index():
    global index, doc_store

    if not os.path.exists(INDEX_PATH):
        raise ValueError("FAISS index not found. Run build_index() first.")

    index = faiss.read_index(INDEX_PATH)
    doc_store = np.load(DOC_STORE_PATH, allow_pickle=True)


# Auto-load if exists
if os.path.exists(INDEX_PATH):
    _load_index()


# ---------------------------------------------------
# 🔹 Retrieve Function (L2 + Custom Similarity)
# ---------------------------------------------------

def retrieve(query: str, top_k: int = TOP_K_DEFAULT, return_scores: bool = False):
    """
    Retrieve top-k chunks using L2 distance.
    Converts L2 distance to similarity using:
        similarity = 1 / (1 + distance)
    """

    global index, doc_store

    if index is None or doc_store is None:
        raise ValueError("Index not loaded. Run build_index().")

    query_embedding = embedding_model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    similarity_scores = []

    for i, idx in enumerate(indices[0]):
        if idx < len(doc_store):
            results.append(doc_store[idx])

            # Convert L2 distance → similarity score
            sim_score = 1 / (1 + distances[0][i])
            similarity_scores.append(float(sim_score))

    if return_scores:
        return list(zip(results, similarity_scores))

    return results

