"""Pure operation evidence projection from an already admitted, sealed context.

Only memory save/correct/forget evidence exists here. A shaped context or this
projection alone never establishes a database commit, source eligibility or
semantic truth. The caller must retain the original context/readset and its live
authority validation through approval. User requests, assistant promises, quiet
observations and ordinary past events cannot create execution receipts.
"""

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

from virtual_companion.boundary.contracts import canonical, companion_id, identifier
from virtual_companion.boundary.history_readset import (
    evidence_hash,
    history_readset,
    validate_current_operation,
    validate_source_processing,
)

VERSION = "operation-evidence.v1"
_PUBLIC_FIELDS = ("id", "action", "status", "text")


class AdmittedOperationContext(Protocol):
    """The existing ShortContext interface, without a circular runtime import."""

    companion: str
    input_id: str
    state_version: int
    payload: str


@dataclass(frozen=True)
class OperationEvidence:
    """Immutable carrier; public/host views are fresh copies of the same seal."""

    _host_json: str

    @property
    def receipt_catalog(self):
        return [{key: row[key] for key in _PUBLIC_FIELDS}
                for row in json.loads(self._host_json)["receipts"]]

    @property
    def host_provenance(self):
        return json.loads(self._host_json)

    @property
    def fingerprint(self):
        return sha256(self._host_json.encode()).hexdigest()


def _receipt(basis, action, status, text, provenance):
    value = {"action": action, "status": status, "text": text, "provenance": provenance}
    receipt_id = "operation-" + evidence_hash({"protocol": VERSION, "basis": basis, **value})
    return {"id": receipt_id, **value}


def _current_source(value, input_id, source_sha256):
    if value["source_id"] != input_id or value["source_sha256"] != source_sha256:
        raise ValueError("Operation evidence differs from the exact current input")


def _current_saved_facts(data, input_id):
    seen = set()
    for row in data["memories"]:
        if (row.get("source_id") != input_id or row.get("status") != "active"
                or row.get("source_kind") not in {"input", "manual"}):
            continue
        identifier(row["record_id"])
        if (type(row["version"]) is not int or row["version"] < 1
                or not isinstance(row.get("content"), str) or not row["content"].strip()
                or len(row["content"]) > 16000 or row["record_id"] in seen):
            raise ValueError("Unique complete current saved fact required")
        seen.add(row["record_id"])
        yield row


def project_operation_evidence(
    context: AdmittedOperationContext | None, *, owner: str, input_id: str,
    state_version: int, user_text: str,
) -> OperationEvidence:
    """Bind only the operation permission already present in ShortRequest.

    Commands retain the durable outcome's own action/status and take precedence
    over any replacement record. Save acknowledgements cover each exact active
    current-input record, never the whole request or another source's record.
    Legacy contexts keep their existing memory_status/current-fact permission;
    without source_processing they have no independent stored raw-source hash.
    No lookup, mutation, model call, wall clock or new operation is performed.
    """
    companion_id(owner)
    identifier(input_id)
    if (type(state_version) is not int or state_version < 0
            or not isinstance(user_text, str) or not user_text.strip()
            or len(user_text.encode()) > 128000):
        raise ValueError("Bound current input and state required for operation evidence")
    source_sha256 = sha256(user_text.encode()).hexdigest()
    basis = {"owner": owner, "input_id": input_id, "state_version": state_version,
             "source_sha256": source_sha256, "context_sha256": None, "readset_sha256": None}
    data = None
    if context is not None:
        try:
            identity = (context.companion, context.input_id, context.state_version)
            payload = context.payload
        except AttributeError as error:
            raise ValueError("Sealed operation context required") from error
        if (identity != (owner, input_id, state_version)
                or type(context.state_version) is not int
                or not isinstance(payload, str) or not payload.strip()
                or len(payload.encode()) > 64000):
            raise ValueError("Operation context scope differs from current input")
        readset = history_readset(owner, input_id, state_version, payload)
        data = json.loads(payload)
        basis.update(context_sha256=readset["context_sha256"],
                     readset_sha256=evidence_hash(readset))

    receipts = []
    if data is not None:
        processing = data.get("source_processing")
        if processing is not None:
            validate_source_processing(processing, input_id)
            _current_source(processing, input_id, source_sha256)
        if "current_operation" in data:
            operation = validate_current_operation(data["current_operation"])
            _current_source(operation, input_id, source_sha256)
            action = {"correct": "correct_memory", "forget": "forget_memory"}[operation["action"]]
            text = {"correct": "本次记忆更正已完成并生效。",
                    "forget": "本次记忆遗忘已完成并生效。"}[operation["action"]]
            if operation["status"] == "already_applied":
                text = "记忆目标已处于所要求状态，本次未重复修改。"
            provenance = {"kind": "current_memory_command", "current_operation": operation}
            if processing is not None:
                provenance["source_processing"] = processing
            receipts.append(_receipt(basis, action, operation["status"], text, provenance))
        elif (processing["status"] == "completed" if processing is not None
              else data["memory_status"] == "current"):
            for row in _current_saved_facts(data, input_id):
                provenance = {"kind": "current_saved_fact", "fact": row}
                if processing is not None:
                    provenance["source_processing"] = processing
                receipts.append(_receipt(basis, "save_memory", "committed",
                                         row["content"], provenance))
    return OperationEvidence(canonical({"protocol": VERSION, "basis": basis, "receipts": receipts}))
