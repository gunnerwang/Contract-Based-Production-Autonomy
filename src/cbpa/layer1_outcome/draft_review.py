"""Explicit review of uncommitted drafts; in-process waiting without phase replay.

This is a prototype application interface, not authenticated operator identity,
process-crash recovery, or permission to change an active production contract.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cbpa.layer1_outcome.contract_validator import ContractValidator
from cbpa.models.contract import OutcomeContract
from cbpa.models.units import normalize_contract_units


class DraftDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    digest: str
    reviewer_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    confirmed: bool
    source: Literal["human", "scripted"]
    contract: OutcomeContract


def _preserves(candidate, active):
    if candidate is None:
        return False
    if (candidate.operator, candidate.limit) == (active.operator, active.limit):
        return True
    if active.operator in {"<", "<="} and candidate.operator in {"<", "<=", "=="}:
        return candidate.limit < active.limit or (
            candidate.limit == active.limit and active.operator == "<="
        )
    if active.operator in {">", ">="} and candidate.operator in {">", ">=", "=="}:
        return candidate.limit > active.limit or (
            candidate.limit == active.limit and active.operator == ">="
        )
    return False


class DraftReviewRequest:
    """A snapshot of one draft, its intent and the active contract, if any."""

    def __init__(self, draft, intent, phase, *, active=None):
        self.request_id = uuid.uuid4().hex
        self._draft = draft.model_copy(deep=True)
        self._active = active.model_copy(deep=True) if active is not None else None
        self.intent = intent
        self.phase = phase
        self._consumed = False
        self._lock = threading.Lock()
        payload = {"draft": self._draft.model_dump(mode="json"), "intent": intent,
                   "phase": phase, "active": self._active.model_dump(mode="json") if self._active else None}
        self.digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    @property
    def draft(self):
        return self._draft.model_copy(deep=True)

    def resolve(self, decision: DraftDecision):
        """Review may repair an uncommitted draft, never weaken active limits."""
        decision = DraftDecision.model_validate(decision)
        with self._lock:
            if self._consumed:
                raise ValueError("Review request has already been resolved")
            if decision.request_id != self.request_id or decision.digest != self.digest:
                raise ValueError("Confirmation does not match this draft review")
            if not decision.confirmed or not decision.reviewer_id.strip() or not decision.reason.strip():
                raise ValueError("Explicit identified confirmation and a reason are required")
            candidate, conversions = normalize_contract_units(decision.contract)
            if candidate.name != self._draft.name:
                raise ValueError("Draft correction cannot change the contract identity")
            original_context = {k: v for k, v in self._draft.context.items() if k != "_draft_review"}
            proposed_context = {k: v for k, v in candidate.context.items() if k != "_draft_review"}
            # Bind both assumption representations to the reviewed snapshot.
            # Observation fields also remain unchanged here: measurements belong
            # to the runtime tracker, not to a replacement draft submitted for review.
            if (proposed_context != original_context
                    or candidate.assumptions != self._draft.assumptions
                    or candidate.typed_assumptions != self._draft.typed_assumptions):
                raise ValueError("Context and assumptions require their typed clarification workflow")
            # Only a complete reviewed draft may replace the pending proposal.
            original_flags = {f.field for f in self._draft.ambiguities}
            if any(not f.resolved and f.field not in original_flags for f in candidate.ambiguities):
                raise ValueError("New unresolved questions require a new review")
            candidate.ambiguities = [f.model_copy(deep=True) for f in self._draft.ambiguities]
            for flag in candidate.ambiguities:
                if not flag.resolved:
                    flag.resolved = True
                    flag.resolution = decision.reason
                    flag.resolution_source = "confirmed_draft" if decision.source == "human" else "scripted_draft"
                    flag.applied_patch = {"review_request_id": self.request_id, "digest": self.digest}
            errors = ContractValidator().validate(candidate).errors
            floors = {"FatigueIndex": .4, "Noise": 80.0, "CyberRiskLevel": 2.0}
            for name, limit in floors.items():
                c = candidate.get_constraint(name)
                if c is None or c.operator not in {"<", "<=", "=="} or not math.isfinite(c.limit) or c.limit > limit:
                    errors.append(f"{name}: missing or relaxed mandatory bound")
            if self._active is not None:
                active, _ = normalize_contract_units(self._active)
                for c in active.hard_constraints:
                    if not _preserves(candidate.get_constraint(c.name), c):
                        errors.append(f"{c.name}: active constraint requires a separate authorised revision")
            if errors:
                raise ValueError("; ".join(errors))
            # Review metadata comes from this request, never from model output.
            candidate.context.pop("_draft_review", None)
            candidate.context["_draft_review"] = {
                "request_id": self.request_id, "digest": self.digest,
                "reviewer_id": decision.reviewer_id, "reason": decision.reason,
                "source": decision.source, "phase": self.phase, "unit_conversions": conversions,
            }
            self._consumed = True
            return candidate


class ReviewInbox:
    """Wait in the same worker stack for a decision supplied by another caller.

    The timeout is explicit. An unanswered request returns None and the runner
    halts; resumption is supported only while this worker remains waiting.
    """

    def __init__(self, timeout_seconds: float):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("A finite positive review timeout is required")
        self.timeout_seconds = timeout_seconds
        self._condition = threading.Condition()
        self._request = None
        self._decision = None

    def __call__(self, request):
        with self._condition:
            if self._request is not None:
                raise RuntimeError("A review is already pending")
            self._request = request
            deadline = time.monotonic() + self.timeout_seconds
            self._condition.notify_all()
            try:
                while self._decision is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(remaining)
                return self._decision
            finally:
                self._request = None
                self._decision = None
                self._condition.notify_all()

    def pending(self):
        with self._condition:
            r = self._request
            if r is None:
                return None
            return {"request_id": r.request_id, "digest": r.digest, "intent": r.intent,
                    "phase": r.phase, "contract": r.draft.model_dump(mode="json"),
                    "active_contract": r._active.model_dump(mode="json") if r._active else None}

    def submit(self, decision):
        decision = DraftDecision.model_validate(decision)
        with self._condition:
            r = self._request
            if r is None or self._decision is not None:
                raise ValueError("No unanswered review is waiting")
            if decision.request_id != r.request_id or decision.digest != r.digest:
                raise ValueError("Stale or mismatched review decision")
            # Validate atomically before unblocking. Invalid input leaves the
            # worker waiting, so it can receive a corrected human decision.
            candidate = r.resolve(decision)
            self._decision = candidate
            self._condition.notify_all()
