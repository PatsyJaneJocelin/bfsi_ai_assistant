# ***BFSI Call Center AI Assistant***



## Overview



This project is a Tiered AI Assistant for BFSI (Banking, Financial Services, Insurance) queries built using:



* FastAPI



* Sentence Transformers



* FAISS



* DistilGPT2 (Fine-tuned SLM)



* Retrieval-Augmented Generation (RAG)



The system is designed to provide **safe, compliant, and structured financial responses** using a multi-tier architecture.



## 

## Architecture

#### Tiered Flow

User Query

&nbsp;    │

&nbsp;    ▼

Guardrails (Sensitive Filtering)

&nbsp;    │

&nbsp;    ▼

Dataset Similarity (FAISS Search)

&nbsp;    │

&nbsp;    ├── Match Found → Return Dataset Response

&nbsp;    │

&nbsp;    ▼

Complex Query?

&nbsp;    │

&nbsp;    ├── YES → RAG (Policy Retrieval + SLM)

&nbsp;    │

&nbsp;    └── NO → SLM Fallback



## 

## Key Features



* Dataset-first compliance



* Sensitive information guardrails



* Semantic similarity search



* Retrieval-Augmented Generation (RAG)



* Fine-tuned Small Language Model (SLM)



* Tiered fallback architecture





## Project Structure



bfsi\_ai\_assistant/

│

├── main.py

├── services/

│   ├── guardrails.py

│   ├── similarity.py

│   ├── rag.py

│   └── model.py

│

├── data/

│   ├── alpaca\_dataset.json

│   └── knowledge\_docs/

│       └── policies.txt

│

├── train\_slm.py

├── generate\_dataset.py

├── requirements.txt

└── README.md





## Setup Instructions



1. Clone Repository

git clone https://github.com/your-username/bfsi\_ai\_assistant.git

cd bfsi\_ai\_assistant



2\. Create Virtual Environment

python -m venv venv

venv\\Scripts\\activate   # Windows



3\. Install Dependencies

pip install -r requirements.txt



4\. Train SLM (Optional)

python train\_slm.py



5\. Run Application

uvicorn main:app --reload





Open:



http://127.0.0.1:8000/docs





## Guardrails



The system blocks:



* Account numbers



* OTPs



* Passwords



* Credit card details



* Crypto queries



* Sensitive matters and information





## Future Improvements



* LoRA fine-tuning



* SME-reviewed dataset expansion



* Policy auto-update pipeline



* Multilingual support
