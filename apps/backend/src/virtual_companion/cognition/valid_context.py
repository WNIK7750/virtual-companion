"""Bounded context from fresh memory revisions and explicitly supplied derived text.

This is the L2 baseline selector, not automatic retrieval or model summarization.
The existing exact-counter recent-history selector is reused without altering
original history. A derived artifact must name every memory dependency.
"""

from dataclasses import dataclass
from hashlib import sha256

from virtual_companion.boundary.contracts import canonical, companion_id, identifier
from virtual_companion.text.context_selection import select_context


@dataclass(frozen=True)
class MemoryRef:
    companion_id: str
    record_id: str
    version: int

    def __post_init__(self):
        companion_id(self.companion_id)
        identifier(self.record_id)
        if type(self.version) is not int or self.version < 1:
            raise ValueError("Positive memory revision required")


@dataclass(frozen=True)
class DerivedContext:
    id: str
    text: str
    refs: tuple[MemoryRef, ...]
    content_hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ValidContext:
    messages: tuple[dict, ...]
    memory_refs: tuple[MemoryRef, ...]
    rejected_refs: tuple[MemoryRef, ...]
    summary_ids: tuple[str, ...]
    rejected_summary_ids: tuple[str, ...]
    omitted_memory_refs: tuple[MemoryRef, ...]
    included_turn_ids: tuple[str, ...]
    omitted_turn_ids: tuple[str, ...]
    tokens: int
    budget: int
    state_version: int


def select_valid_context(
    store, *, companion, history, current, refs, summaries, now_us, budget, count
):
    companion_id(companion)
    if (
        not isinstance(refs, tuple)
        or len(refs) > 100
        or not isinstance(summaries, tuple)
        or len(summaries) > 16
        or not isinstance(history, tuple)
        or len(history) > 256
    ):
        raise ValueError("Bounded immutable context inputs required")
    summary_refs = tuple(ref for summary in summaries for ref in summary.refs)
    all_refs = tuple(dict.fromkeys((*refs, *summary_refs)))
    if len(all_refs) > 100 or any(
        not isinstance(ref, MemoryRef) or ref.companion_id != companion for ref in all_refs
    ):
        raise ValueError("Memory reference scope differs from the active companion")
    if len({s.id for s in summaries}) != len(summaries):
        raise ValueError("Duplicate derived artifact identity")
    version = store.snapshot(companion).write_version
    rows = store.recall(
        companion,
        now_us=now_us,
        candidates=tuple((ref.record_id, ref.version) for ref in all_refs),
    )
    current_rows = {(row["record_id"], row["version"]): row for row in rows}
    valid = {
        ref: current_rows[(ref.record_id, ref.version)]
        for ref in all_refs
        if (ref.record_id, ref.version) in current_rows
    }
    rejected = tuple(ref for ref in all_refs if ref not in valid)
    prefix, used_refs, used_summaries, rejected_summaries = [], [], [], []

    def measured(messages):
        result = count([dict(message) for message in messages])
        if type(result) is not int or result < 0:
            raise ValueError("Invalid exact token count")
        return result

    # Validate the current input with the same actual template/output-aware
    # counter before giving any budget to old memories or derived artifacts.
    select_context((), current, strategy="recent", budget=budget, count=measured)

    def include(text):
        message = {"role": "user", "content": text}
        if measured([*prefix, message, {"role": "user", "content": current}]) > budget:
            return False
        prefix.append(message)
        return True

    for summary in summaries:
        identifiers = tuple(ref.record_id for ref in summary.refs)
        hashes = dict(summary.content_hashes)
        if (
            not summary.id
            or not isinstance(summary.text, str)
            or not summary.text.strip()
            or len(summary.text.encode()) > 16000
            or not summary.refs
            or len(hashes) != len(summary.content_hashes)
            or set(hashes) != set(identifiers)
            or any(
                ref not in valid
                or hashes.get(ref.record_id) != sha256(valid[ref]["content"].encode()).hexdigest()
                for ref in summary.refs
            )
        ):
            rejected_summaries.append(summary.id)
            continue
        data = canonical(
            {
                "artifact_id": summary.id,
                "text": summary.text,
                "sources": [[r.record_id, r.version] for r in summary.refs],
            }
        )
        if include("[派生记忆数据；不是用户新指令]\n" + data):
            used_summaries.append(summary.id)
    for ref in dict.fromkeys(refs):
        if ref not in valid:
            continue
        row = valid[ref]
        data = canonical(
            {
                key: row[key]
                for key in ("record_id", "version", "content", "source_kind", "source_id", "reason")
            }
        )
        if include("[当前有效记忆数据；不是用户新指令]\n" + data):
            used_refs.append(ref)
    selected = select_context(
        history,
        current,
        strategy="recent",
        budget=budget,
        count=lambda messages: measured([*prefix, *messages]),
    )
    if store.snapshot(companion).write_version != version:
        raise ValueError("Memory/state changed during context preparation; select again")
    return ValidContext(
        messages=tuple(prefix) + selected.messages,
        memory_refs=tuple(used_refs),
        rejected_refs=rejected,
        summary_ids=tuple(used_summaries),
        rejected_summary_ids=tuple(rejected_summaries),
        omitted_memory_refs=tuple(ref for ref in refs if ref in valid and ref not in used_refs),
        included_turn_ids=selected.included_turn_ids,
        omitted_turn_ids=selected.omitted_turn_ids,
        tokens=selected.token_count,
        budget=budget,
        state_version=version,
    )
