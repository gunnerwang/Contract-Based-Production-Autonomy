"""LLM client implementations for the CBPA case study.

Two providers are supported:

* **Claude** — uses the Claude Agent SDK with CLI authorization
  (``claude login``).  No API key needed.
* **OpenAI** — uses the OpenAI Python SDK with an API key
  (``OPENAI_API_KEY`` environment variable).

All clients expose the same interface (``query_text`` / ``query_structured``
/ ``is_available``) so the rest of the framework is provider-agnostic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Protocol

logger = logging.getLogger(__name__)


# ── Common interface ────────────────────────────────────────────────

class LLMClient(Protocol):
    """Protocol that every LLM client must satisfy."""

    def query_text(self, system: str, user_message: str) -> str: ...

    def query_structured(
        self,
        system: str,
        user_message: str,
        tool_name: str,
        tool_schema: dict,
        tool_description: str = "Return structured output",
    ) -> dict: ...

    @property
    def is_available(self) -> bool: ...


# ── Claude (Agent SDK, CLI auth) ───────────────────────────────────

class ClaudeClient:
    """LLM client using the Claude Agent SDK.

    Uses the locally authenticated Claude Code CLI — no API key needed.
    Log in first via ``claude login``.
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.model = model

    def query_text(self, system: str, user_message: str) -> str:
        prompt = f"{system}\n\n{user_message}"
        return asyncio.run(_claude_query_text(prompt, self.model))

    def query_structured(
        self,
        system: str,
        user_message: str,
        tool_name: str,
        tool_schema: dict,
        tool_description: str = "Return structured output",
    ) -> dict:
        json_instruction = (
            "\n\nYou MUST respond with ONLY a JSON object (no markdown fences, "
            "no explanation, no commentary before or after) that conforms to "
            "this schema:\n"
            f"{json.dumps(tool_schema, indent=2)}"
        )
        prompt = system + "\n\n" + user_message + json_instruction
        raw = asyncio.run(_claude_query_text(prompt, self.model))
        return _extract_json(raw)

    @property
    def is_available(self) -> bool:
        return True

    def check_connection(self) -> tuple[bool, str]:
        """Test that the Claude SDK can reach the API.

        Returns ``(ok, message)`` — ``ok`` is True if a short query
        succeeded, False otherwise.
        """
        try:
            result = self.query_text(
                "Reply with exactly: OK", "Ping"
            )
            return True, f"Claude ({self.model}) connected"
        except Exception as exc:
            return False, f"Claude ({self.model}) unreachable: {exc}"


async def _claude_query_text(prompt: str, model: str) -> str:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        TextBlock,
        query,
    )

    import shutil

    # Allow the SDK to launch from within a Claude Code session
    os.environ.pop("CLAUDECODE", None)

    # Prefer the system-installed Claude Code CLI over the bundled binary
    # in the SDK — the bundled binary may be outdated and crash under load.
    system_cli = shutil.which("claude")

    options = ClaudeAgentOptions(
        system_prompt="",
        max_turns=1,
        model=model,
        **({"cli_path": system_cli} if system_cli else {}),
    )
    parts: list[str] = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
    return "".join(parts)


# ── OpenAI (API key) ──────────────────────────────────────────────

class OpenAIClient:
    """LLM client using the OpenAI Python SDK.

    Requires ``OPENAI_API_KEY`` set in the environment (or passed to
    the constructor).  Works with any OpenAI-compatible endpoint —
    set ``OPENAI_BASE_URL`` to override the default.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")

        if not self._api_key:
            logger.warning(
                "OPENAI_API_KEY not set — OpenAI client will not be available. "
                "Set the environment variable or pass api_key= to the constructor."
            )

    def _get_client(self):  # noqa: ANN202
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenAI support. "
                "Install it with:  pip install openai"
            )
        kwargs: dict = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return OpenAI(**kwargs)

    def query_text(self, system: str, user_message: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    def query_structured(
        self,
        system: str,
        user_message: str,
        tool_name: str,
        tool_schema: dict,
        tool_description: str = "Return structured output",
    ) -> dict:
        client = self._get_client()

        json_instruction = (
            "\n\nYou MUST respond with ONLY a JSON object (no markdown fences, "
            "no explanation, no commentary before or after) that conforms to "
            "this schema:\n"
            f"{json.dumps(tool_schema, indent=2)}"
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message + json_instruction},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        return _extract_json(raw)

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def check_connection(self) -> tuple[bool, str]:
        """Test that the OpenAI API is reachable."""
        if not self._api_key:
            return False, "OPENAI_API_KEY not set"
        try:
            result = self.query_text(
                "Reply with exactly: OK", "Ping"
            )
            return True, f"OpenAI ({self.model}) connected"
        except Exception as exc:
            return False, f"OpenAI ({self.model}) unreachable: {exc}"


# ── Factory ────────────────────────────────────────────────────────

def create_client(
    provider: str = "claude",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ClaudeClient | OpenAIClient:
    """Create an LLM client for the given provider.

    Parameters
    ----------
    provider:
        ``"claude"`` (default) — uses Claude Agent SDK with CLI auth.
        ``"openai"`` — uses OpenAI SDK with ``OPENAI_API_KEY``.
    model:
        Model name override.  Defaults: ``"claude-sonnet-4-6"`` for Claude,
        ``"gpt-4o"`` for OpenAI.
    api_key:
        Explicit API key (OpenAI only).  Falls back to env var.
    base_url:
        Custom base URL (OpenAI only).  Falls back to ``OPENAI_BASE_URL``.
    """
    provider = provider.lower().strip()

    if provider == "claude":
        return ClaudeClient(model=model or "claude-sonnet-4-6")
    elif provider in ("openai", "gpt"):
        return OpenAIClient(
            model=model or "gpt-4o",
            api_key=api_key,
            base_url=base_url,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Supported: 'claude' (CLI auth), 'openai' (API key)."
        )


# ── Shared helpers ─────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse JSON from LLM response, returning empty dict")
    return {}
