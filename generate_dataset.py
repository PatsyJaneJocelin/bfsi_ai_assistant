import json

# Professional prefixes for variation
prefixes = [
    "",
    "Please explain ",
    "Kindly clarify ",
    "Help me understand ",
    "Could you provide details on "
]

categories = {
    "Loan Eligibility": {
        "questions": [
            "the eligibility criteria for a personal loan",
            "home loan eligibility requirements",
            "factors that determine loan approval",
            "how loan eligibility is assessed",
            "requirements to apply for a business loan"
        ],
        "response": "Loan eligibility is evaluated based on income stability, credit profile, employment type, existing obligations, and repayment capacity as per current bank policy."
    },

    "Loan Application Status": {
        "questions": [
            "my loan application status",
            "how to track a loan request",
            "loan approval timeline",
            "current status of my submitted loan",
            "steps to check loan processing stage"
        ],
        "response": "Loan application status can be verified using the registered mobile number or application reference ID through official banking channels."
    },

    "EMI Details": {
        "questions": [
            "how EMI is calculated",
            "the EMI formula",
            "EMI breakdown structure",
            "impact of tenure on EMI",
            "loan amortization schedule"
        ],
        "response": "EMI is calculated using the standard amortization formula based on principal amount, applicable interest rate, and repayment tenure as per current financial guidelines."
    },

    "Interest Rate Information": {
        "questions": [
            "current loan interest rates",
            "how interest rates are determined",
            "difference between fixed and floating rates",
            "factors affecting interest rate changes",
            "where to check updated loan rates"
        ],
        "response": "Interest rates are determined based on credit assessment, loan type, market benchmarks, and internal financial policy guidelines."
    },

    "Payment Queries": {
        "questions": [
            "how to make a loan repayment",
            "available loan payment methods",
            "loan prepayment process",
            "overdue EMI payment procedure",
            "penalties for delayed payment"
        ],
        "response": "Loan repayments may be completed through authorized digital banking platforms or designated payment channels as per institutional policy."
    },

    "Insurance Queries": {
        "questions": [
            "how to file an insurance claim",
            "documents required for insurance claims",
            "insurance claim processing timeline",
            "how to check claim status",
            "coverage details under my policy"
        ],
        "response": "Insurance claims require submission of relevant documentation and are processed according to policy terms, verification procedures, and regulatory guidelines."
    },

    "Refusal": {
        "questions": [
            "my account balance",
            "my customer data",
            "my loan password",
            "bitcoin price",
            "election results"
        ],
        "response": "I'm unable to assist with that request. Please contact official customer support for further assistance."
    }
}

dataset = []

for category, content in categories.items():
    for question in content["questions"]:
        for prefix in prefixes:
            full_question = (prefix + question).strip().capitalize() + "?"
            dataset.append({
                "instruction": full_question,
                "input": "",
                "output": content["response"]
            })

# Save file
with open("data/alpaca_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)

print(f"{len(dataset)} clean Alpaca samples generated successfully.")




"""
DATASET CREATION

import json
import random

categories = {
    "Loan Eligibility": [
        "What are the eligibility criteria for a personal loan?",
        "Am I eligible for a home loan?",
        "How is loan eligibility determined?",
        "What factors affect my loan approval?",
        "Can I apply for a loan with a low credit score?"
    ],
    "Loan Application Status": [
        "How can I check my loan application status?",
        "Where can I track my loan request?",
        "Is my loan approved?",
        "How long does loan approval take?",
        "What is the current status of my application?"
    ],
    "EMI Details": [
        "How is EMI calculated?",
        "Explain EMI formula.",
        "Can you provide EMI breakdown?",
        "How does loan tenure affect EMI?",
        "What is an amortization schedule?"
    ],
    "Interest Rate Information": [
        "What is the current interest rate?",
        "How are loan interest rates decided?",
        "Is the interest rate fixed or floating?",
        "What affects interest rate changes?",
        "Where can I check latest rates?"
    ],
    "Payment Queries": [
        "How can I make loan payment?",
        "What are available repayment methods?",
        "Can I prepay my loan?",
        "How do I pay overdue EMI?",
        "Are there penalties for late payment?"
    ],
    "Insurance Queries": [
        "How do I file an insurance claim?",
        "What documents are needed for claim?",
        "How long does claim processing take?",
        "Can I check claim status?",
        "What is covered under my policy?"
    ],
    "Refusal": [
        "What is my account balance?",
        "Give me my customer data.",
        "Tell me my loan password.",
        "What is bitcoin price?",
        "Who will win elections?"
    ]
}

responses = {
    "Loan Eligibility": "Eligibility for a loan is assessed based on income stability, credit history, employment type, and repayment capacity as per current bank policy.",
    "Loan Application Status": "Loan application status can be checked using the registered mobile number or application ID through official banking channels.",
    "EMI Details": "EMI is calculated using the standard amortization formula considering principal amount, tenure, and applicable interest rate.",
    "Interest Rate Information": "Interest rates vary based on credit profile, loan type, and current financial policy guidelines.",
    "Payment Queries": "Payments can be made through net banking, mobile banking, or authorized payment channels.",
    "Insurance Queries": "Insurance claims require submission of relevant documents and are processed as per policy terms and conditions.",
    "Refusal": "I'm unable to assist with that request. Please contact official customer support for further assistance."
}

dataset = []

for category, questions in categories.items():
    for i in range(25):
        question = random.choice(questions)
        dataset.append({
            "instruction": question,
            "input": "",
            "output": responses[category]
        })

with open("data/alpaca_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)

print("175 realistic Alpaca samples generated successfully.")
"""
