from __future__ import annotations

import os
from typing import Callable

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODELS = [
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
    "anthropic/claude-3.5-sonnet",
    "meta-llama/llama-3.1-8b-instruct",
]

# Legacy/invalid IDs sometimes cached in the UI — map to a working model.
MODEL_ALIASES: dict[str, str] = {
    "google/gemini-1.5-flash": "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001": "google/gemini-2.5-flash",
    "google/gemini-flash-1.5": "openai/gpt-4o-mini",
}


class LLMError(RuntimeError):
    """Raised when the OpenRouter chat API fails."""


def has_openrouter_api_key() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY", "").strip())


def get_openrouter_models() -> list[str]:
    raw = os.getenv("OPENROUTER_MODELS", "").strip()
    if raw:
        return [model.strip() for model in raw.split(",") if model.strip()]
    return list(DEFAULT_OPENROUTER_MODELS)


def get_default_openrouter_model() -> str:
    raw = os.getenv("OPENROUTER_MODEL", get_openrouter_models()[0]).strip()
    return resolve_openrouter_model(raw or get_openrouter_models()[0])


def resolve_openrouter_model(model: str | None) -> str:
    selected = (model or get_openrouter_models()[0]).strip()
    if selected in MODEL_ALIASES:
        selected = MODEL_ALIASES[selected]
    allowed = set(get_openrouter_models())
    if selected in allowed:
        return selected
    return get_openrouter_models()[0]


def create_openrouter_llm(model: str | None = None) -> Callable[[str], str]:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to .env")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenRouter requires openai package. Run: pip install openai") from exc

    selected_model = resolve_openrouter_model(model)
    site_url = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8080")
    app_name = os.getenv("OPENROUTER_APP_NAME", "RAG Workflow Studio")

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": site_url,
            "X-Title": app_name,
        },
    )

    def chat_llm(prompt: str) -> str:
        try:
            response = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. Answer concisely using ONLY the provided context. "
                            "Cite context blocks as [1], [2], etc. If context is insufficient, say so."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=int(os.getenv("OPENROUTER_MAX_TOKENS", "1024")),
            )
        except Exception as exc:
            message = str(exc)
            if "404" in message or "No endpoints found" in message or "not a valid model" in message:
                raise LLMError(
                    f"Model '{selected_model}' is not available on OpenRouter. "
                    f"Try: {', '.join(DEFAULT_OPENROUTER_MODELS)}"
                ) from exc
            if "402" in message or "credits" in message.lower():
                raise LLMError(
                    f"OpenRouter credits insufficient for '{selected_model}'. "
                    "Add credits at openrouter.ai/settings/credits or switch to openai/gpt-4o-mini."
                ) from exc
            raise LLMError(f"OpenRouter request failed for '{selected_model}': {message}") from exc
        return response.choices[0].message.content or ""

    chat_llm._model_name = selected_model  # type: ignore[attr-defined]
    return chat_llm
