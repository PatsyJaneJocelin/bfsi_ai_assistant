from fastapi import FastAPI
from pydantic import BaseModel

from services.similarity import search_similar_query
from services.guardrails import apply_guardrails
from services.model import generate_slm_response, generate_rag_response

app = FastAPI(title="BFSI Call Center AI Assistant")


# ---------------------------------------------------
# 🔹 Request Schema
# ---------------------------------------------------

class QueryRequest(BaseModel):
    query: str


# ---------------------------------------------------
# 🔹 Health Endpoint
# ---------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "running"}


# ---------------------------------------------------
# 🔹 Helper: Detect complex financial queries
# ---------------------------------------------------

def is_complex_query(query: str) -> bool:
    complex_keywords = [
        "calculate",
        "breakdown",
        "formula",
        "penalty",
        "clause",
        "amortization",
        "foreclosure",
        "interest calculation"
    ]
    return any(word in query.lower() for word in complex_keywords)


# ---------------------------------------------------
# 🔹 Main Query Endpoint
# ---------------------------------------------------

@app.post("/query")
def handle_query(payload: QueryRequest):

    query = payload.query.strip()

    if not query:
        return {"response": "Query cannot be empty.", "tier": "error"}

    # Tier 0: Guardrails
    blocked, message = apply_guardrails(query)
    if blocked:
        return {"response": message, "tier": "guardrail"}

    # Tier 1: Dataset Similarity
    result, score = search_similar_query(query)
    if result:
        return {
            "response": result,
            "tier": "dataset",
            "similarity_score": round(score, 3)
        }

    # Tier 3: RAG
    if is_complex_query(query):
        response = generate_rag_response(query)
        return {"response": response, "tier": "rag"}

    # Tier 2: SLM fallback
    response = generate_slm_response(query)
    return {"response": response, "tier": "slm"}
