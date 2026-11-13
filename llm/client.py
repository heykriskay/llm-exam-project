"""Thin wrapper around the local Ollama chat API."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import ollama

import config

PROMPT_DIR = Path(__file__).parent / "prompts"
_client = ollama.Client(host=config.OLLAMA_HOST)

SUGGESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "intro": {"type": "string"},
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": ["integer", "null"]},
                    "name": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "name", "reason"],
            },
        },
    },
    "required": ["intro", "suggestions"],
}


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """System prompt from llm/prompts/<name>.md (the 4T's live there)."""
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


def chat(system: str, messages: list[dict], schema: dict | None = None,
         temperature: float = config.TEMPERATURE) -> str:
    resp = _client.chat(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": system}, *messages],
        format=schema,
        options={"temperature": temperature, "num_ctx": config.NUM_CTX},
    )
    return resp["message"]["content"]
