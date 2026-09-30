"""Pluggable LLM providers.

  search(system, user)   -> LLMResult(text, sources, searched)   used by monthly deep dives
  complete(system, user) -> LLMResult(text, [], False)           used by news classification

Deep dives REQUIRE live web search. If a provider answers without searching, the caller treats
it as unavailable (numbers-only fallback) - an LLM judging from stale memory is not allowed.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = logging.getLogger(__name__)


class QuotaExhausted(RuntimeError):
    """Provider refused due to rate/daily limits; stop calling it for this run."""


@dataclass
class LLMResult:
    text: str
    sources: list[dict] = field(default_factory=list)
    searched: bool = False


class GeminiProvider:
    """Google Gemini REST API with Google Search grounding (free tier where available)."""

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, cfg: dict, api_key: str | None = None, http: httpx.Client | None = None):
        self.model = cfg["model"]
        self.min_gap = cfg.get("min_seconds_between_calls", 13)
        self.max_retries = cfg.get("max_retries", 3)
        self.key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.http = http or httpx.Client(timeout=180)
        self._last = 0.0

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    def _call(self, system: str, user: str, grounded: bool) -> dict:
        body = {"system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.2}}
        if grounded:
            body["tools"] = [{"google_search": {}}]
        for attempt in range(self.max_retries + 1):
            wait = self.min_gap - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            r = self.http.post(self.URL.format(model=self.model), json=body,
                               headers={"x-goog-api-key": self.key})
            if r.status_code == 429:
                msg = r.text[:300]
                if "per day" in msg.lower() or "daily" in msg.lower() or attempt == self.max_retries:
                    raise QuotaExhausted(msg)
                time.sleep(min(60, 10 * 2 ** attempt))
                continue
            if r.status_code >= 500 and attempt < self.max_retries:
                time.sleep(5 * 2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("unreachable")

    @staticmethod
    def _parse(data: dict) -> LLMResult:
        cands = data.get("candidates") or []
        if not cands:
            raise RuntimeError(f"no candidates: {str(data)[:200]}")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        gm = cands[0].get("groundingMetadata") or {}
        sources = [{"title": c["web"].get("title", ""), "url": c["web"].get("uri", "")}
                   for c in gm.get("groundingChunks", []) if c.get("web")]
        searched = bool(sources or gm.get("webSearchQueries"))
        return LLMResult(text, sources, searched)

    def search(self, system: str, user: str) -> LLMResult:
        return self._parse(self._call(system, user, grounded=True))

    def complete(self, system: str, user: str) -> LLMResult:
        return self._parse(self._call(system, user, grounded=False))


class AnthropicProvider:
    """Claude with the server-side web search tool."""

    def __init__(self, cfg: dict, client=None):
        self.model, self.max_tokens = cfg["model"], cfg.get("max_tokens", 4000)
        self.max_searches = cfg.get("max_searches", 8)
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _run(self, system, user, tools) -> LLMResult:
        messages = [{"role": "user", "content": user}]
        sources, so_far = [], []
        for _ in range(4):
            resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                               system=system, messages=messages, **({"tools": tools} if tools else {}))
            for block in resp.content:
                if getattr(block, "type", "") == "web_search_tool_result":
                    for r in getattr(block, "content", []) or []:
                        if getattr(r, "url", None):
                            sources.append({"title": getattr(r, "title", ""), "url": r.url})
            if resp.stop_reason == "pause_turn":
                so_far = so_far + list(resp.content)
                messages = [messages[0], {"role": "assistant", "content": so_far}]
                continue
            break
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        return LLMResult(text, sources, bool(sources))

    def search(self, system, user) -> LLMResult:
        return self._run(system, user, [{"type": "web_search_20250305", "name": "web_search",
                                         "max_uses": self.max_searches}])

    def complete(self, system, user) -> LLMResult:
        return self._run(system, user, None)


def make_provider(llm_cfg: dict):
    """Returns a provider or None (-> numbers-only research, keyword-only news)."""
    if not llm_cfg.get("enabled", True):
        return None
    name = llm_cfg.get("provider", "gemini")
    sub = llm_cfg.get(name, {})
    if name == "gemini" and GeminiProvider.available():
        return GeminiProvider(sub)
    if name == "anthropic" and AnthropicProvider.available():
        return AnthropicProvider(sub)
    log.warning("LLM provider %s not configured (missing API key); running without LLM", name)
    return None
