import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from services.rag import retrieve

MODEL_PATH = "./models/slm"

# Load tokenizer and model once
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

model.eval()


# ---------------------------------------------------
# 🔹 Common generation helper
# ---------------------------------------------------

def _generate_text(prompt: str, max_tokens: int = 120) -> str:
    """
    Internal helper for deterministic text generation.
    """

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.0,           # deterministic
            do_sample=False,
            repetition_penalty=1.2
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only generated portion
    generated = full_output[len(prompt):].strip()

    # Stop if structure repeats
    stop_tokens = ["### Policy Context:", "### Customer Query:", "### Response:"]
    for token in stop_tokens:
        if token in generated:
            generated = generated.split(token)[0]

    return generated.strip()


# ---------------------------------------------------
# 🔹 Tier 2: Pure SLM (No Retrieval)
# ---------------------------------------------------

def generate_slm_response(query: str) -> str:
    """
    Controlled fallback generation without retrieval.
    """

    prompt = f"""
You are a compliant BFSI call center AI assistant.
Provide a professional and policy-safe response.
Do NOT generate specific financial numbers, links, or assumptions.
If unsure, advise contacting official support.

### Customer Query:
{query}

### Response:
"""

    response = _generate_text(prompt, max_tokens=100)

    return _post_process(response)


# ---------------------------------------------------
# 🔹 Tier 3: RAG-Based Generation
# ---------------------------------------------------

def generate_rag_response(query: str) -> str:
    """
    Retrieval-Augmented Generation.
    Uses policy context strictly.
    """

    retrieved_docs = retrieve(query, top_k=3)
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are a compliant BFSI policy assistant.
Answer strictly using the policy context below.
Do NOT generate assumptions or external information.
If answer is not found in context, say:
"I'm unable to find that information in the policy."

### Policy Context:
{context}

### Customer Query:
{query}

### Response:
"""

    response = _generate_text(prompt, max_tokens=120)

    return _post_process(response)


# ---------------------------------------------------
# 🔹 Safety Post-Processing
# ---------------------------------------------------

def _post_process(response: str) -> str:
    """
    Final compliance cleaning layer.
    """

    # Block URLs
    if re.search(r"http|www", response):
        return "For accurate and policy-aligned information, please contact official customer support."

    # Block large numeric hallucinations (like fake rates)
    if re.search(r"\d{4,}", response):
        return "For accurate financial details, please refer to official banking channels."

    # Remove strange artifacts
    response = re.sub(r"[*\[\]]+", "", response)

    # Keep only first paragraph
    response = response.split("\n")[0].strip()

    return response

