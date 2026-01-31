from __future__ import annotations

import json
import os
from typing import Optional

import httpx

from app.llm.prompts import SYSTEM_PROMPT, USER_PROMPT


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def enabled(self) -> bool:
        return bool(self.api_key)

    def judge_claim(self, claim: str, evidence: str) -> Optional[dict]:
        if not self.enabled():
            return None
        payload = {
            "model": "gpt-4o-mini",
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(claim=claim, evidence=evidence)},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=30.0) as client:
            resp = client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return None
