"""Opt-in context comparison pilot; not wired into the desktop generation path.

Summary text and recall ranking are supplied recordings, not inferred memory.
The counter must include the actual model's system prompt and chat template.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    id: str
    user: str
    assistant: str | None

    @property
    def digest(self):
        raw = json.dumps([self.id, self.user, self.assistant], ensure_ascii=False).encode()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class RecordedSummary:
    id: str
    text: str
    sources: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ContextCandidate:
    strategy: str
    messages: tuple[dict, ...]
    included_turn_ids: tuple[str, ...]
    omitted_turn_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    rejected_evidence_ids: tuple[str, ...]
    omitted_evidence_ids: tuple[str, ...]
    token_count: int
    budget: int
    counter_calls: int

    @property
    def fits(self):
        return self.token_count <= self.budget


def select_context(
    history: tuple[Turn, ...],
    current: str,
    *,
    budget: int,
    count: Callable[[list[dict]], int],
    strategy: str,
    summaries: tuple[RecordedSummary, ...] = (),
    recalled_turn_ids: tuple[str, ...] = (),
    invalidated_evidence_ids: tuple[str, ...] = (),
) -> ContextCandidate:
    if strategy not in {"full", "recent", "summary_recall"}:
        raise ValueError("Unknown context strategy")
    if type(budget) is not int or budget < 1 or not isinstance(current, str) or not current.strip():
        raise ValueError("Invalid context input or budget")
    by_id = {turn.id: turn for turn in history}
    if len(by_id) != len(history) or any(
        not isinstance(turn.id, str)
        or not turn.id
        or not isinstance(turn.user, str)
        or (turn.assistant is not None and not isinstance(turn.assistant, str))
        for turn in history
    ):
        raise ValueError("Invalid or duplicate turn")
    if len({summary.id for summary in summaries}) != len(summaries):
        raise ValueError("Duplicate summary id")
    if any(turn_id not in by_id for turn_id in recalled_turn_ids):
        raise ValueError("Unknown recalled turn")
    cache = {}

    def measure(selected, evidence):
        ordered = tuple(turn.id for turn in history if turn.id in selected)
        key = (ordered, tuple(summary.id for summary in evidence))
        if key not in cache:
            messages = []
            for summary in evidence:
                sources = ", ".join(source for source, _ in summary.sources)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"[先导录制摘要 {summary.id}；来源轮次 {sources}]\n{summary.text}"
                        ),
                    }
                )
            for turn_id in ordered:
                turn = by_id[turn_id]
                messages.append({"role": "user", "content": turn.user})
                if turn.assistant is not None:
                    messages.append({"role": "assistant", "content": turn.assistant})
            messages.append({"role": "user", "content": current})
            tokens = count([dict(message) for message in messages])
            if type(tokens) is not int or tokens < 0:
                raise ValueError("Invalid token count")
            cache[key] = tokens, tuple(messages)
        return cache[key]

    if measure(set(), ())[0] > budget:
        raise ValueError("Current input exceeds context budget; no truncation allowed")
    selected, evidence, rejected = set(), [], []
    if strategy == "full":
        selected = set(by_id)
    else:
        if strategy == "summary_recall":
            for summary in summaries:
                valid = (
                    bool(summary.id)
                    and isinstance(summary.text, str)
                    and bool(summary.text.strip())
                    and bool(summary.sources)
                    and summary.id not in invalidated_evidence_ids
                    and all(
                        source in by_id and by_id[source].digest == digest
                        for source, digest in summary.sources
                    )
                )
                if not valid:
                    rejected.append(summary.id)
                elif measure(selected, (*evidence, summary))[0] <= budget:
                    evidence.append(summary)
            for turn_id in recalled_turn_ids:
                if measure(selected | {turn_id}, evidence)[0] <= budget:
                    selected.add(turn_id)
        # A contiguous recent suffix; recall can separately retain an older turn.
        for turn in reversed(history):
            if turn.id in selected:
                continue
            if measure(selected | {turn.id}, evidence)[0] > budget:
                break
            selected.add(turn.id)
    tokens, messages = measure(selected, evidence)
    evidence_ids = tuple(summary.id for summary in evidence)
    return ContextCandidate(
        strategy,
        messages,
        tuple(turn.id for turn in history if turn.id in selected),
        tuple(turn.id for turn in history if turn.id not in selected),
        evidence_ids,
        tuple(rejected),
        tuple(summary.id for summary in summaries if summary.id not in evidence_ids),
        tokens,
        budget,
        len(cache),
    )
