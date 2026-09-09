"""Tool-use schema helpers for LLM structured output."""

from __future__ import annotations

from typing import Any


def make_tool(
    name: str,
    description: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Create an Anthropic tool definition."""
    return {
        "name": name,
        "description": description,
        "input_schema": schema,
    }
