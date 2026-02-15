import re

# Sensitive patterns to block (strictly private / harmful requests)
SENSITIVE_PATTERNS = [
    r"account number",
    r"password",
    r"\botp\b",
    r"credit card number",
    r"\bcvv\b",
    r"customer data",
    r"\bbalance\b",
    r"pan number",
    r"aadhar number",
    r"ssn",
    r"bitcoin",
    r"crypto",
    r"politics",
    r"election",
    r"elections",
    r"link",
    r"url",
    r"website"
]

REFUSAL_MESSAGE = (
    "I'm unable to assist with that request. "
    "For security reasons, please contact official customer support."
)


def contains_sensitive_request(query: str) -> bool:
    """
    Checks if the query contains sensitive or restricted patterns.
    """
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in SENSITIVE_PATTERNS)


def apply_guardrails(query: str):
    """
    Applies security guardrails.

    Returns:
        (is_blocked: bool, message: str or None)
    """

    # Block sensitive requests
    if contains_sensitive_request(query):
        return True, REFUSAL_MESSAGE

    # Allow everything else
    return False, None




