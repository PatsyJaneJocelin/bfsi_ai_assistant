import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load dataset
with open("data/alpaca_dataset.json", "r") as f:
    dataset = json.load(f)

# Extract instructions
instructions = [item["instruction"] for item in dataset]

# Generate embeddings
instruction_embeddings = embedding_model.encode(instructions)

# Create FAISS index
dimension = instruction_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(instruction_embeddings))


def search_similar_query(query, threshold=0.75):
    """
    Returns stored response if similarity above threshold.
    """

    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), 1)

    """
    We can use FAISS with L2 distance. For simplicity in the MVP,
    we convert distance into a bounded similarity score using 1/(1+d).
    In future iterations, we plan to switch to cosine similarity with normalized embeddings
    for better semantic alignment.”
    """

    similarity_score = 1 / (1 + distances[0][0])  # Convert distance to similarity score

    if similarity_score >= threshold:
        matched_response = dataset[indices[0][0]]["output"]
        return matched_response, float(similarity_score)

    return None, float(similarity_score)
