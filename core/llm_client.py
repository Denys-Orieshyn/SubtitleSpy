"""Synchronous LLM client with MD5-based request caching."""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_SYSTEM_PROMPT = (
    "You translate subtitles into Russian and briefly explain idioms, slang, "
    "or domain-specific terms when it helps understanding. Keep the answer concise."
)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    input_text: str
    output_text: str
    input_hash: str
    from_cache: bool = False
    raw_response: dict[str, Any] | None = None


class LLMClient:
    """OpenAI-compatible synchronous chat client.

    The client keeps a small in-memory cache keyed by MD5 of the source text.
    Repeated calls with the same subtitle text return the cached answer without
    sending a second HTTP request.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1/chat/completions",
        timeout: float = 30.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        cache_size: int = 128,
        client: Any | None = None,
        response_parser: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.cache_size = cache_size
        self._response_parser = response_parser or _parse_openai_chat_response
        self._cache: OrderedDict[str, LLMResponse] = OrderedDict()
        self._owns_client = client is None
        self._client = client if client is not None else self._create_http_client(timeout)

    def translate_subtitle(
        self,
        text: str,
        *,
        target_language: str = "Russian",
        source_language: str | None = None,
    ) -> LLMResponse:
        normalized_text = _normalize_input(text)
        if not normalized_text:
            raise ValueError("text cannot be empty")

        text_hash = md5_text(normalized_text)
        cached = self._cache.get(text_hash)
        if cached is not None:
            self._cache.move_to_end(text_hash)
            return LLMResponse(
                input_text=cached.input_text,
                output_text=cached.output_text,
                input_hash=cached.input_hash,
                from_cache=True,
                raw_response=cached.raw_response,
            )

        if not self.api_key:
            raise RuntimeError(
                "LLM API key is not configured. Pass api_key=... or set OPENAI_API_KEY."
            )

        payload = self._build_payload(
            normalized_text,
            target_language=target_language,
            source_language=source_language,
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = self._client.post(self.base_url, json=payload, headers=headers)
        response.raise_for_status()
        raw = response.json()
        output = self._response_parser(raw).strip()
        result = LLMResponse(
            input_text=normalized_text,
            output_text=output,
            input_hash=text_hash,
            from_cache=False,
            raw_response=raw,
        )
        self._store_cache(text_hash, result)
        return result

    def close(self) -> None:
        if self._owns_client and hasattr(self._client, "close"):
            self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _build_payload(
        self,
        text: str,
        *,
        target_language: str,
        source_language: str | None,
    ) -> dict[str, Any]:
        language_hint = f"Source language: {source_language}." if source_language else "Detect source language."
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{language_hint}\n"
                        f"Target language: {target_language}.\n\n"
                        f"Subtitle text:\n{text}"
                    ),
                },
            ],
            "temperature": 0.2,
        }

    def _store_cache(self, text_hash: str, response: LLMResponse) -> None:
        self._cache[text_hash] = response
        self._cache.move_to_end(text_hash)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    @staticmethod
    def _create_http_client(timeout: float) -> Any:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "The 'httpx' package is required for LLM requests. "
                "Install project dependencies with 'pip install -r requirements.txt'."
            ) from exc
        return httpx.Client(timeout=timeout)


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _normalize_input(text: str) -> str:
    return " ".join(text.split())


def _parse_openai_chat_response(raw: dict[str, Any]) -> str:
    try:
        return str(raw["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected LLM response format") from exc
