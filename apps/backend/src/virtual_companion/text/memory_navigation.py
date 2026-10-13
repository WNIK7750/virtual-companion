"""Bounded navigation over current committed memory originals.

Directory entries are labels and references only.  Original rows are fetched
again through the committed store port when a caller explicitly reads them.
"""

import base64
import binascii
import hashlib
import json
from datetime import datetime

from virtual_companion.boundary.contracts import (
    VersionConflict,
    canonical,
    companion_id,
    identifier,
)
from virtual_companion.text.chronology import utc_microseconds

_MAX_PAGE_BYTES = 8192
_MAX_PAGE_ENTRIES = 8
_MAX_RECORDS = 1000
_MAX_SUMMARY_CHARS = 240
_MAX_PREVIEW_CHARS = 160


def _checked_identifier(value):
    identifier(value)
    return value


class MemoryCatalog:
    """Freeze a bounded navigation index; never retain full memory bodies."""

    def __init__(self, store, companion, *, state_version, clock, checkpoint,
                 max_records=1000, page_size=8, dialogue=False):
        companion_id(companion)
        if type(state_version) is not int or state_version < 0:
            raise ValueError("Non-negative frozen memory state version required")
        if not callable(clock):
            raise ValueError("Catalog clock required")
        if checkpoint is not None and not callable(checkpoint):
            raise ValueError("Catalog checkpoint must be callable")
        if type(max_records) is not int or not 1 <= max_records <= _MAX_RECORDS:
            raise ValueError("Catalog record bound must be between 1 and 1000")
        if type(page_size) is not int or not 1 <= page_size <= _MAX_PAGE_ENTRIES:
            raise ValueError("Catalog page size must be between 1 and 8")
        if not callable(getattr(store, "committed_catalog_page", None)):
            raise ValueError("Committed catalog page port required")
        if not callable(getattr(store, "committed_recall", None)):
            raise ValueError("Committed original read port required")
        if not callable(getattr(store, "snapshot", None)):
            raise ValueError("Memory state snapshot port required")

        self.dialogue = dialogue
        self.store = store
        self.companion = companion
        self.state_version = state_version
        self.clock = clock
        self.checkpoint = checkpoint
        self.max_records = max_records
        self.page_size = page_size
        self._last_clock_us = None
        self._metadata = []
        self._metadata_by_ref = {}
        self._nodes = {}

        self._scan()
        self._build_identity_and_nodes()

    @property
    def identity(self):
        """SHA-256 identity of the frozen, owner- and version-bound directory."""
        return self._identity

    def _checkpoint(self):
        if self.checkpoint is not None:
            self.checkpoint()

    def _read_clock(self):
        value = self.clock()
        if type(value) is not int or value < 0:
            raise ValueError("Catalog clock must return non-negative microseconds")
        if self._last_clock_us is not None and value < self._last_clock_us:
            raise VersionConflict("Memory catalog evidence clock moved backwards")
        self._last_clock_us = value
        return value

    def _assert_state(self):
        snapshot = self.store.snapshot(self.companion)
        if getattr(snapshot, "write_version", None) != self.state_version:
            raise VersionConflict("Memory catalog state version changed")

    def _before(self):
        self._checkpoint()
        now = self._read_clock()
        self._assert_state()
        self._checkpoint()
        return now

    def _after(self, started_at):
        self._checkpoint()
        finished_at = self._read_clock()
        if finished_at < started_at:
            raise VersionConflict("Memory catalog evidence clock moved backwards")
        self._assert_state()
        self._checkpoint()
        return finished_at

    @staticmethod
    def _expiry_us(value):
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Committed memory expiry must be ISO text")
        try:
            return utc_microseconds(datetime.fromisoformat(value))
        except (TypeError, ValueError) as error:
            raise ValueError("Committed memory expiry is invalid") from error

    def _metadata_row(self, row):
        if not isinstance(row, dict):
            raise ValueError("Committed catalog row must be a mapping")
        record_id = _checked_identifier(row.get("record_id"))
        version = row.get("version")
        if type(version) is not int or version < 1:
            raise ValueError("Committed record version required")
        if row.get("status") != "active":
            raise ValueError("Committed catalog returned an inactive row")
        kind = row.get("kind")
        if kind not in (
            {"fact", "event", "dialogue_fragment"} if self.dialogue else {"fact", "event"}
        ):
            raise ValueError("Known committed fact or event kind required")
        source_kind = _checked_identifier(row.get("source_kind"))
        content = row.get("content")
        if not isinstance(content, str):
            raise ValueError("Committed original text required")
        content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        fact_key = row.get("fact_key")
        if kind == "fact":
            if not isinstance(fact_key, dict) or set(fact_key) != {
                "subject", "predicate", "scope",
            }:
                raise ValueError("Explicit fact subject, predicate and scope required")
            fact_key = {
                "subject": _checked_identifier(fact_key["subject"]),
                "predicate": _checked_identifier(fact_key["predicate"]),
                "scope": _checked_identifier(fact_key["scope"]),
            }
        elif fact_key is not None:
            raise ValueError("Events cannot enter the fact-key navigation tree")
        return {
            "record_id": record_id,
            "version": version,
            **({"source_type": kind, "member_count": row.get("member_count", 1)}
               if self.dialogue else {}),
            "kind": kind,
            "source_kind": source_kind,
            "fact_key": fact_key,
            "expires_at": row.get("expires_at"),
            "content_sha256": content_sha256,
            "preview": content[:_MAX_PREVIEW_CHARS],
            "preview_truncated": len(content) > _MAX_PREVIEW_CHARS,
        }

    def _scan(self):
        after_record_id = None
        seen = set()
        as_of_us = None
        coverage = "complete"
        while True:
            started_at = self._before()
            if as_of_us is None:
                as_of_us = started_at
            remaining = self.max_records - len(self._metadata)
            limit = min(self.page_size, remaining)
            page = self.store.committed_catalog_page(
                self.companion, now_us=as_of_us,
                after_record_id=after_record_id, limit=limit,
            )
            finished_at = self._after(started_at)
            if not isinstance(page, dict) or set(page) != {
                "records", "next_after_record_id",
            }:
                raise ValueError("Committed catalog page contract changed")
            rows = page["records"]
            next_after = page["next_after_record_id"]
            if not isinstance(rows, tuple) or len(rows) > limit:
                raise ValueError("Bounded committed catalog page required")
            if next_after is not None:
                identifier(next_after)
                if not rows or next_after != rows[-1].get("record_id"):
                    raise ValueError("Committed catalog continuation is inconsistent")
            for row in rows:
                metadata = self._metadata_row(row)
                ref = (metadata["record_id"], metadata["version"])
                if ref in seen:
                    raise ValueError("Committed catalog repeated a record version")
                if after_record_id is not None and metadata["record_id"] <= after_record_id:
                    raise ValueError("Committed catalog page order is inconsistent")
                seen.add(ref)
                self._metadata.append(metadata)
                self._metadata_by_ref[ref] = metadata
            self._assert_no_expired(finished_at, self._metadata)

            if len(self._metadata) >= self.max_records:
                if next_after is not None:
                    coverage = "partial"
                break
            if next_after is None:
                break
            if next_after == after_record_id:
                raise ValueError("Committed catalog continuation did not advance")
            after_record_id = next_after

        self._as_of_us = as_of_us if as_of_us is not None else self._read_clock()
        if getattr(self.store, 'catalog_source_coverage', 'complete') == 'partial':
            coverage = 'partial'
        self._coverage = coverage
        self._assert_no_expired(self._read_clock(), self._metadata)
        self._assert_state()

    @staticmethod
    def _assert_no_expired(now_us, metadata):
        for row in metadata:
            expires_us = MemoryCatalog._expiry_us(row["expires_at"])
            if expires_us is not None and now_us >= expires_us:
                raise VersionConflict("Memory catalog entry expired; rebuild navigation")

    def _build_identity_and_nodes(self):
        identity_payload = {
            "protocol": "memory-catalog.v1",
            "companion_id": self.companion,
            "state_version": self.state_version,
            "as_of_us": self._as_of_us,
            "coverage": self._coverage,
            "max_records": self.max_records,
            "page_size": self.page_size,
            "records": self._metadata,
        }
        self._identity = hashlib.sha256(canonical(identity_payload).encode("utf-8")).hexdigest()
        self._nodes = {"root": []}
        facts = {}
        events = {}
        for row in self._metadata:
            if row["kind"] == "fact":
                key = row["fact_key"]
                facts.setdefault(key["subject"], {}).setdefault(
                    (key["predicate"], key["scope"]), [],
                ).append(row)
            else:
                events.setdefault(row["source_kind"], []).append(row)

        for subject, predicates in sorted(facts.items()):
            subject_node = self._node_id("subject", (subject,))
            self._nodes["root"].append({
                "entry_type": "subject",
                "node_id": subject_node,
                "label": subject,
                "subject": subject,
                "record_count": sum(len(rows) for rows in predicates.values()),
                "summary": self._summary(f"事实导航标签：subject={subject}"),
                "navigation_only": True,
            })
            children = []
            for (predicate, scope), rows in sorted(predicates.items()):
                predicate_node = self._node_id("predicate", (subject, predicate, scope))
                children.append({
                    "entry_type": "predicate",
                    "node_id": predicate_node,
                    "label": predicate,
                    "predicate": predicate,
                    "scope": scope,
                    "record_count": len(rows),
                    "summary": self._summary(
                        f"事实导航路径：subject={subject} / predicate={predicate} / scope={scope}"
                    ),
                    "navigation_only": True,
                })
                self._nodes[predicate_node] = self._record_entries(rows)
            self._nodes[subject_node] = children

        for source_kind, rows in sorted(events.items()):
            source_node = self._node_id("source_kind", (source_kind,))
            self._nodes["root"].append({
                "entry_type": "source_kind",
                "node_id": source_node,
                "label": source_kind,
                "source_kind": source_kind,
                "record_count": len(rows),
                "summary": self._summary(f"事件导航标签：source_kind={source_kind}"),
                "navigation_only": True,
            })
            self._nodes[source_node] = self._record_entries(rows)

        self._nodes["root"].sort(key=lambda item: (item["entry_type"], item["label"]))

    def _node_id(self, kind, parts):
        digest = hashlib.sha256(canonical({
            "catalog_sha256": self._identity,
            "kind": kind,
            "parts": parts,
        }).encode("utf-8")).hexdigest()
        return f"{kind}:{digest}"

    @staticmethod
    def _summary(value):
        return value[:_MAX_SUMMARY_CHARS]

    @classmethod
    def _record_entries(cls, rows):
        entries = []
        for row in sorted(rows, key=lambda item: (item["record_id"], item["version"])):
            summary = f"导航预览，需 read_memory 回查原文：{row['preview']}"
            entries.append({
                "entry_type": "record",
                "record_id": row["record_id"],
                "version": row["version"],
                **({"source_type": row["source_type"], "member_count": row["member_count"]}
                   if "source_type" in row else {}),
                "summary": cls._summary(summary),
                "preview_truncated": row["preview_truncated"],
                "navigation_only": True,
            })
        return entries

    def _encode_cursor(self, node_id, offset):
        payload = canonical({
            "catalog_sha256": self._identity,
            "node_id": node_id,
            "offset": offset,
        }).encode("utf-8")
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

    def _decode_cursor(self, cursor, node_id, total):
        if not isinstance(cursor, str) or not cursor or len(cursor) > 512:
            raise ValueError("Invalid memory catalog cursor")
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        except (ValueError, UnicodeError, json.JSONDecodeError, binascii.Error) as error:
            raise ValueError("Invalid memory catalog cursor") from error
        if (not isinstance(payload, dict)
                or set(payload) != {"catalog_sha256", "node_id", "offset"}
                or payload["catalog_sha256"] != self._identity
                or payload["node_id"] != node_id):
            raise ValueError("Cursor belongs to another catalog or node")
        offset = payload["offset"]
        if type(offset) is not int or offset <= 0 or offset >= total:
            raise ValueError("Invalid memory catalog cursor offset")
        return offset

    def _page(self, node_id, entries, offset):
        end = offset
        while end < len(entries) and end - offset < self.page_size:
            candidate_end = end + 1
            candidate_next = (
                self._encode_cursor(node_id, candidate_end)
                if candidate_end < len(entries) else None
            )
            candidate = {
                "protocol": "memory-catalog.v1",
                "catalog_sha256": self._identity,
                "node_id": node_id,
                "entries": [dict(entry) for entry in entries[offset:candidate_end]],
                "next_cursor": candidate_next,
                "catalog_coverage": self._coverage,
                "scope": {
                    "kind": "current_committed_qualified_records",
                    "companion_id": self.companion,
                    "state_version": self.state_version,
                    "as_of_us": self._as_of_us,
                },
            }
            if len(canonical(candidate).encode("utf-8")) > _MAX_PAGE_BYTES:
                if end == offset:
                    raise ValueError("One navigation entry exceeds the page byte budget")
                break
            end = candidate_end

        next_cursor = self._encode_cursor(node_id, end) if end < len(entries) else None
        result = {
            "protocol": "memory-catalog.v1",
            "catalog_sha256": self._identity,
            "node_id": node_id,
            "entries": [dict(entry) for entry in entries[offset:end]],
            "next_cursor": next_cursor,
            "catalog_coverage": self._coverage,
            "scope": {
                "kind": "current_committed_qualified_records",
                "companion_id": self.companion,
                "state_version": self.state_version,
                "as_of_us": self._as_of_us,
            },
        }
        if len(canonical(result).encode("utf-8")) > _MAX_PAGE_BYTES:
            raise ValueError("Memory catalog page exceeds the byte budget")
        return result

    def browse(self, node_id="root", *, cursor=None):
        started_at = self._before()
        if not isinstance(node_id, str) or node_id not in self._nodes:
            raise ValueError("Unknown memory catalog node")
        entries = self._nodes[node_id]
        offset = 0 if cursor is None else self._decode_cursor(cursor, node_id, len(entries))
        result = self._page(node_id, entries, offset)
        finished_at = self._after(started_at)
        self._assert_no_expired(finished_at, self._metadata)
        return result

    def read(self, references):
        started_at = self._before()
        if not isinstance(references, tuple) or len(references) > _MAX_PAGE_ENTRIES:
            raise ValueError("Read requires an immutable tuple of at most eight references")
        seen = set()
        for reference in references:
            if not isinstance(reference, tuple) or len(reference) != 2:
                raise ValueError("Record/version reference tuple required")
            _checked_identifier(reference[0])
            version = reference[1]
            if type(version) is not int or version < 1:
                raise ValueError("Positive record version required")
            if reference in seen:
                raise ValueError("Duplicate memory navigation reference")
            seen.add(reference)
            if reference not in self._metadata_by_ref:
                raise ValueError("Reference is not in this frozen memory catalog")

        if not references:
            finished_at = self._after(started_at)
            self._assert_no_expired(finished_at, self._metadata)
            return ()

        if self.dialogue:
            if sum(self._metadata_by_ref[ref]['member_count'] for ref in references) > 8:
                raise ValueError('Read requires at most eight original members including facts')
        rows = self.store.committed_recall(
            self.companion, now_us=started_at, candidates=references,
            limit=len(references),
        )
        found = tuple((row.get("record_id"), row.get("version")) for row in rows)
        if found != references:
            raise VersionConflict("Memory catalog original expired, changed or was suppressed")
        for row, reference in zip(rows, references, strict=True):
            metadata = self._metadata_by_ref[reference]
            if self.dialogue and row.get('kind') != metadata['source_type']:
                raise VersionConflict('Frozen typed catalog source kind changed')
            digest = hashlib.sha256(row["content"].encode("utf-8")).hexdigest()
            if digest != metadata["content_sha256"]:
                raise VersionConflict("Memory catalog original differs from its frozen version")
        finished_at = self._after(started_at)
        self._assert_no_expired(finished_at, rows)
        return tuple(rows)
