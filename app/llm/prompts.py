SYSTEM_PROMPT = """You are a careful fact-checking assistant. Given a claim and evidence, decide if the claim is supported, contradicted, or not enough info. Return JSON only."""

USER_PROMPT = """Claim: {claim}

Evidence:
{evidence}

Return JSON with keys: label (SUPPORTED|CONTRADICTED|NOT_ENOUGH_INFO), confidence (0-1)."""
