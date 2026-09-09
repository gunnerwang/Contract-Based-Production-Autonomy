"""Instrumented LLM client wrapper.

Wraps any :class:`cbpa.llm.client.LLMClient` and records every call so that
repeated-run experiments can report, per call site:

* number of calls and mean latency,
* transport exceptions (which the layer agents turn into deterministic
  fallbacks),
* empty outputs (``{}`` returned by ``_extract_json`` when the model did not
  produce parseable JSON),
* JSON-schema violations of the structured output against the tool schema
  that was requested.

The wrapper is transparent: the framework code does not need to know it is
talking to an instrumented client.  Use it as::

    client = InstrumentedLLMClient(create_client("claude"))
    exp = FactoryExperiment(use_llm=True, llm_client=client)
    ...
    client.stats()   # -> dict suitable for JSON export

A companion :class:`FallbackLogCapture` logging handler collects the
``"... falling back to deterministic"`` warnings emitted by the layer agents,
so that the *consequence* of a failed call (which component fell back) is
recorded alongside the call itself.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

try:  # jsonschema is an optional dependency; validation is skipped without it
    import jsonschema as _jsonschema
except Exception:  # pragma: no cover - import guard
    _jsonschema = None


@dataclass
class LLMCallRecord:
    """One LLM invocation."""

    index: int
    kind: str  # "text" | "structured"
    tool_name: str
    started_at: float
    latency_s: float
    ok: bool
    error: str | None = None
    empty_output: bool = False
    schema_valid: bool | None = None
    schema_error: str | None = None
    n_items: int | None = None
    response_chars: int = 0


class InstrumentedLLMClient:
    """Transparent wrapper recording every call made through an LLM client."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.records: list[LLMCallRecord] = []

    # ── passthrough attributes ─────────────────────────────────────────
    @property
    def inner(self) -> Any:
        return self._inner

    @property
    def model(self) -> str | None:
        return getattr(self._inner, "model", None)

    @property
    def is_available(self) -> bool:
        return bool(getattr(self._inner, "is_available", True))

    def check_connection(self) -> tuple[bool, str]:
        return self._inner.check_connection()

    def sampling_settings(self) -> dict[str, Any]:
        """Describe how the wrapped client samples (for the methods section)."""
        name = type(self._inner).__name__
        if name == "ClaudeClient":
            return {
                "provider": "anthropic",
                "model": self.model,
                "transport": "claude-agent-sdk (Claude Code CLI auth), max_turns=1",
                "temperature": None,
                "note": (
                    "The Agent SDK does not expose a temperature parameter; "
                    "responses are sampled at the provider default."
                ),
            }
        if name == "OpenAIClient":
            return {
                "provider": "openai",
                "model": self.model,
                "transport": "chat.completions",
                "temperature": 0.2,
                "response_format": "json_object (structured calls only)",
            }
        return {"provider": name, "model": self.model}

    # ── instrumented calls ─────────────────────────────────────────────
    def query_text(self, system: str, user_message: str) -> str:
        rec = LLMCallRecord(
            index=len(self.records), kind="text", tool_name="text",
            started_at=time.time(), latency_s=0.0, ok=False,
        )
        t0 = time.perf_counter()
        try:
            out = self._inner.query_text(system, user_message)
        except Exception as exc:
            rec.latency_s = time.perf_counter() - t0
            rec.error = f"{type(exc).__name__}: {exc}"[:500]
            self.records.append(rec)
            raise
        rec.latency_s = time.perf_counter() - t0
        rec.ok = True
        rec.response_chars = len(out or "")
        rec.empty_output = not (out or "").strip()
        self.records.append(rec)
        return out

    def query_structured(
        self,
        system: str,
        user_message: str,
        tool_name: str,
        tool_schema: dict,
        tool_description: str = "Return structured output",
    ) -> dict:
        rec = LLMCallRecord(
            index=len(self.records), kind="structured", tool_name=tool_name,
            started_at=time.time(), latency_s=0.0, ok=False,
        )
        t0 = time.perf_counter()
        try:
            out = self._inner.query_structured(
                system, user_message, tool_name, tool_schema, tool_description
            )
        except Exception as exc:
            rec.latency_s = time.perf_counter() - t0
            rec.error = f"{type(exc).__name__}: {exc}"[:500]
            self.records.append(rec)
            raise
        rec.latency_s = time.perf_counter() - t0
        rec.ok = True
        out = out if isinstance(out, dict) else {}
        rec.response_chars = len(str(out))
        rec.empty_output = out == {}
        rec.n_items = _count_items(out)
        if not rec.empty_output:
            rec.schema_valid, rec.schema_error = _validate(out, tool_schema)
        else:
            rec.schema_valid = False
            rec.schema_error = "empty output (JSON extraction failed)"
        self.records.append(rec)
        return out

    # ── reporting ──────────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        """Aggregate call records (overall and per tool name)."""
        by_tool: dict[str, list[LLMCallRecord]] = {}
        for r in self.records:
            by_tool.setdefault(r.tool_name, []).append(r)
        out: dict[str, Any] = {
            "total": _summarise(self.records),
            "by_tool": {k: _summarise(v) for k, v in sorted(by_tool.items())},
        }
        return out

    def export_records(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.records]


def _count_items(out: dict) -> int | None:
    for v in out.values():
        if isinstance(v, list):
            return len(v)
    return None


def _validate(out: dict, schema: dict) -> tuple[bool | None, str | None]:
    if _jsonschema is None or not schema:
        return None, None
    try:
        validator = _jsonschema.Draft7Validator(schema)
        errors = sorted(validator.iter_errors(out), key=lambda e: list(e.path))
    except Exception as exc:  # malformed schema — do not fail the experiment
        return None, f"schema not checkable: {exc}"[:200]
    if not errors:
        return True, None
    first = errors[0]
    path = "/".join(str(p) for p in first.path) or "<root>"
    return False, f"{len(errors)} violation(s); first at {path}: {first.message}"[:300]


def _summarise(records: list[LLMCallRecord]) -> dict[str, Any]:
    n = len(records)
    ok = [r for r in records if r.ok]
    lat = [r.latency_s for r in ok]
    structured = [r for r in records if r.kind == "structured"]
    return {
        "calls": n,
        "exceptions": n - len(ok),
        "empty_output": sum(1 for r in records if r.ok and r.empty_output),
        "schema_invalid": sum(
            1 for r in structured if r.ok and r.schema_valid is False and not r.empty_output
        ),
        "schema_checked": sum(1 for r in structured if r.ok and r.schema_valid is not None),
        "mean_latency_s": round(statistics.fmean(lat), 3) if lat else None,
        "max_latency_s": round(max(lat), 3) if lat else None,
        "total_latency_s": round(sum(lat), 1),
    }


class FallbackLogCapture(logging.Handler):
    """Collect layer warnings about LLM output and classify them.

    Categories:

    * ``fallback``        — an LLM call failed and the deterministic path was used
      ("... falling back to deterministic", "L3 repair agent failed", ...);
    * ``contract_repair`` — the contract validator re-inserted a mandatory hard
      constraint the LLM draft omitted ("LLM omitted mandatory constraint ...");
    * ``other``           — any other warning mentioning the LLM.

    Non-LLM warnings (e.g. the Pareto filter's "infeasible fallback" notice)
    are ignored.
    """

    FALLBACK_KEYWORDS = ("falling back", "agent failed", "reasoner failed", "synthesis failed")
    REPAIR_KEYWORDS = ("omitted mandatory constraint", "canonicalised to",
                       "relaxed mandatory constraint", "cross-check tightened",
                       "omitted typed assumption")

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.events: list[dict[str, str]] = []

    @classmethod
    def classify(cls, msg: str) -> str | None:
        low = msg.lower()
        if any(k in low for k in cls.REPAIR_KEYWORDS):
            return "contract_repair"
        if any(k in low for k in cls.FALLBACK_KEYWORDS):
            return "fallback"
        if "llm" in low:
            return "other"
        return None

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        # The handler is bound to the named logger and to its existing descendants
        # (see :meth:`_targets`), so one record can reach it several times as it
        # propagates.  Mark the record the first time and count it once.
        marker = f"_cbpa_capture_{id(self)}"
        if getattr(record, marker, False):
            return
        setattr(record, marker, True)
        try:
            msg = record.getMessage()
        except Exception:  # pragma: no cover
            return
        cat = self.classify(msg)
        if cat is not None:
            self.events.append({"logger": record.name, "category": cat, "message": msg[:300]})

    def count(self, category: str = "fallback") -> int:
        return sum(1 for e in self.events if e["category"] == category)

    def by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.events:
            counts[e["category"]] = counts.get(e["category"], 0) + 1
        return dict(sorted(counts.items()))

    def by_component(self, category: str = "fallback") -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.events:
            if e["category"] != category:
                continue
            comp = e["logger"].split(".")[-1]
            counts[comp] = counts.get(comp, 0) + 1
        return dict(sorted(counts.items()))

    def _targets(self, logger_name: str) -> list[logging.Logger]:
        """The named logger and every existing descendant.

        The handler is attached to each logger directly rather than relying on
        propagation: some environments install a logger class with
        ``propagate = False`` (e.g. ROS ``launch.logging``), which would
        otherwise swallow the warnings.  Call :meth:`attach` after the
        framework modules have been imported so that all loggers exist.
        """
        prefix = logger_name + "."
        found = [logging.getLogger(logger_name)]
        for name, obj in list(logging.Logger.manager.loggerDict.items()):
            if isinstance(obj, logging.Logger) and name.startswith(prefix):
                found.append(obj)
        return found

    def attach(self, logger_name: str = "cbpa") -> "FallbackLogCapture":
        for lg in self._targets(logger_name):
            if self not in lg.handlers:
                lg.addHandler(self)
        return self

    def detach(self, logger_name: str = "cbpa") -> None:
        for lg in self._targets(logger_name):
            if self in lg.handlers:
                lg.removeHandler(self)
