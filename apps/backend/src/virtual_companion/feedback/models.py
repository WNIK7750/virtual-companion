from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from .errors import ValidationError

SCHEMA_VERSION = 1
MAX_FREEFORM_LENGTH = 2000
MAX_TAGS = 16
MAX_TAG_LENGTH = 64
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_HTML_TAG_RE = re.compile(r"<\s*/?\s*[A-Za-z!][^>]*>")
_ALLOWED_LAYERS = frozenset(
    {"prompt", "state", "memory", "sampling", "ui", "model_limit", "unknown"}
)
_ALLOWED_CONFIDENCE = frozenset({"low", "medium", "high"})
_ALLOWED_FEEDBACK_STATUS = frozenset({"active", "withdrawn"})
_ALLOWED_EXPORT_STATUS = frozenset({"active", "stale"})


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def feedback_idempotency_fingerprint(event: FeedbackEvent) -> str:
    """Hash the logical request while ignoring retry-generated identity/timestamp fields."""
    value = event.to_mapping()
    value.pop("feedback_id", None)
    value.pop("created_at", None)
    return sha256_json(value)


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string")
    if "\x00" in value:
        raise ValidationError(f"{field} contains NUL")
    return value.strip()


def _timestamp(value: datetime | str | None) -> str:
    if value is None:
        current = datetime.now(UTC)
    elif isinstance(value, datetime):
        current = value
    elif isinstance(value, str):
        try:
            current = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValidationError("created_at must be ISO-8601") from exc
    else:
        raise ValidationError("created_at must be datetime, ISO-8601, or null")
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValidationError("created_at must include an offset")
    return current.isoformat()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValidationError(f"{field} must be a lowercase SHA-256")
    return value


def _target_binding(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    fields = {
        "schema_version", "kind", "status", "attempt", "range", "source_sha256",
        "selection_sha256", "visible_sequence", "receipt_refs",
        "receipt_snapshot_sha256", "received", "visible", "read", "audio",
    }
    if not isinstance(value, Mapping) or set(value) != fields or value.get("schema_version") != 1:
        raise ValidationError("Exact versioned target binding required")
    if value.get("kind") not in {"reply", "bubble", "attempt"}:
        raise ValidationError("Unsupported feedback target kind")
    if value.get("status") not in {"visible", "attempt-only", "unknown"}:
        raise ValidationError("Unsupported feedback target status")
    if ((value["kind"] == "attempt" and value["status"] == "visible")
            or (value["status"] == "visible" and value["kind"] != "attempt"
                and value["visible"] is not True)):
        raise ValidationError("Feedback target status contradicts its visible scope")
    attempt = value.get("attempt")
    if not isinstance(attempt, Mapping) or set(attempt) != {
            "input_id", "boundary_attempt_id", "text_attempt_id"}:
        raise ValidationError("Exact feedback attempt binding required")
    for field in ("input_id", "boundary_attempt_id"):
        if attempt[field] is not None:
            _identifier(attempt[field], f"target attempt {field}")
    _identifier(attempt["text_attempt_id"], "target attempt text_attempt_id")
    if value["kind"] == "attempt":
        if (value["range"] is not None or value["source_sha256"] is not None
                or value["selection_sha256"] is not None):
            raise ValidationError("Attempt-only feedback cannot carry message text")
    else:
        selected_range = value["range"]
        if (not isinstance(selected_range, Mapping)
                or set(selected_range) != {"unit", "start", "end"}
                or selected_range.get("unit") != "unicode-code-point"
                or type(selected_range.get("start")) is not int
                or type(selected_range.get("end")) is not int
                or selected_range["start"] < 0
                or selected_range["end"] <= selected_range["start"]):
            raise ValidationError("Exact Unicode code-point range required")
        _sha(value.get("source_sha256"), "target source_sha256")
        _sha(value.get("selection_sha256"), "target selection_sha256")
    if value.get("visible_sequence") is not None and (
            type(value["visible_sequence"]) is not int or value["visible_sequence"] < 1):
        raise ValidationError("target visible_sequence must be positive or null")
    _sha(value.get("receipt_snapshot_sha256"), "target receipt_snapshot_sha256")
    references = value.get("receipt_refs")
    if (not isinstance(references, Sequence) or isinstance(references, (str, bytes))
            or len(references) > 32):
        raise ValidationError("Bounded feedback receipt references required")
    for reference in references:
        if not isinstance(reference, Mapping) or set(reference) != {
                "sequence", "fingerprint", "status", "received", "visible", "host_receipt_sha256"}:
            raise ValidationError("Exact feedback receipt reference required")
        if type(reference["sequence"]) is not int or reference["sequence"] < 1:
            raise ValidationError("Feedback receipt sequence must be positive")
        _sha(reference["fingerprint"], "receipt fingerprint")
        if reference["status"] not in {"pending", "unknown", "confirmed"}:
            raise ValidationError("Unsupported feedback receipt status")
        for field in ("received", "visible"):
            if (reference[field] is not True and reference[field] is not False
                    and reference[field] is not None):
                raise ValidationError(f"Receipt {field} state must be true, false or unknown")
        if reference["host_receipt_sha256"] is not None:
            _sha(reference["host_receipt_sha256"], "host_receipt_sha256")
    for field in ("received", "visible"):
        if value[field] is not True and value[field] is not False and value[field] is not None:
            raise ValidationError(f"Feedback {field} state must be true, false or unknown")
    if value.get("read") != "unknown" or value.get("audio") != "unknown":
        raise ValidationError("Feedback read and audio facts remain unknown")
    return json.loads(canonical_json_bytes(value))


def _freeform(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError("freeform_reaction must be a string or null")
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > MAX_FREEFORM_LENGTH:
        raise ValidationError(f"freeform_reaction exceeds {MAX_FREEFORM_LENGTH} characters")
    if "\x00" in normalized:
        raise ValidationError("freeform_reaction contains NUL")
    if _HTML_TAG_RE.search(normalized):
        raise ValidationError("freeform_reaction must be plain text")
    return normalized


def _tags(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise ValidationError("quick_tags must be a sequence of strings")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValidationError("quick_tags must contain strings")
        tag = value.strip()
        if not tag:
            continue
        if len(tag) > MAX_TAG_LENGTH:
            raise ValidationError(f"quick tag exceeds {MAX_TAG_LENGTH} characters")
        if "\x00" in tag:
            raise ValidationError("quick tag contains NUL")
        if tag not in result:
            result.append(tag)
    if len(result) > MAX_TAGS:
        raise ValidationError(f"quick_tags cannot contain more than {MAX_TAGS} items")
    return tuple(result)


_PROVIDER_BINDING_FIELDS = (
    "provider_route", "provider_config", "provider_revision",
    "provider_model", "provider_wire_profile",
)


@dataclass(frozen=True, slots=True)
class VersionBinding:
    app_commit: str
    schema_version: int
    persona_sha256: str
    prompt_sha256: str
    model_sha256: str | None
    server_sha256: str | None
    sampling_profile_id: str
    provider_route: str | None = None
    provider_config: str | None = None
    provider_revision: int | None = None
    provider_model: str | None = None
    provider_wire_profile: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in {1, 2}:
            raise ValidationError("version_binding schema_version must be 1 or 2")
        if not isinstance(self.app_commit, str) or not _COMMIT_RE.fullmatch(self.app_commit):
            raise ValidationError(
                "version_binding app_commit must be a 40-character lowercase commit"
            )
        for field in ("persona_sha256", "prompt_sha256"):
            _sha(getattr(self, field), f"version_binding {field}")
        _identifier(self.sampling_profile_id, "version_binding sampling_profile_id")
        if self.schema_version == 1:
            for field in ("model_sha256", "server_sha256"):
                _sha(getattr(self, field), f"version_binding {field}")
            if any(getattr(self, field) is not None for field in _PROVIDER_BINDING_FIELDS):
                raise ValidationError("Local binding cannot carry remote provider identity")
        else:
            if self.model_sha256 is not None or self.server_sha256 is not None:
                raise ValidationError("Remote model and server artifact hashes are unknown")
            for field in ("provider_route", "provider_config"):
                _sha(getattr(self, field), f"version_binding {field}")
            if type(self.provider_revision) is not int or self.provider_revision < 1:
                raise ValidationError("Positive provider config revision required")
            for field in ("provider_model", "provider_wire_profile"):
                _identifier(getattr(self, field), f"version_binding {field}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> VersionBinding:
        if not isinstance(value, Mapping):
            raise ValidationError("version_binding must be an object")
        required = {"app_commit", "schema_version", "persona_sha256", "prompt_sha256",
                    "model_sha256", "server_sha256", "sampling_profile_id"}
        if value.get("schema_version") == 2:
            required.update(_PROVIDER_BINDING_FIELDS)
        if set(value) != required:
            raise ValidationError("Exact versioned binding fields required")
        return cls(**value)

    def to_mapping(self) -> dict[str, Any]:
        result = {
            "app_commit": self.app_commit,
            "schema_version": self.schema_version,
            "persona_sha256": self.persona_sha256,
            "prompt_sha256": self.prompt_sha256,
            "model_sha256": self.model_sha256,
            "server_sha256": self.server_sha256,
            "sampling_profile_id": self.sampling_profile_id,
        }
        if self.schema_version == 2:
            result.update({field: getattr(self, field) for field in _PROVIDER_BINDING_FIELDS})
        return result


@dataclass(frozen=True, slots=True)
class FeedbackEvent:
    schema_version: int
    feedback_id: str
    session_id: str
    turn_id: str
    assistant_message_id: str | None
    quick_tags: tuple[str, ...]
    freeform_reaction: str | None
    feedback_revision: int
    supersedes_revision: int | None
    status: Literal["active", "withdrawn"]
    created_at: str
    version_binding: VersionBinding
    target_binding: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2}:
            raise ValidationError("unsupported FeedbackEvent schema_version")
        for field in ("feedback_id", "session_id", "turn_id"):
            _identifier(getattr(self, field), field)
        target = _target_binding(self.target_binding)
        if target is not self.target_binding:
            object.__setattr__(self, "target_binding", target)
        if self.assistant_message_id is not None:
            _identifier(self.assistant_message_id, "assistant_message_id")
        elif not (self.schema_version == 2 and target is not None and target["kind"] == "attempt"):
            raise ValidationError("assistant_message_id is required for message feedback")
        if self.schema_version == 1 and target is not None:
            raise ValidationError("Version 1 feedback cannot carry a target binding")
        if self.feedback_revision < 1:
            raise ValidationError("feedback_revision must be positive")
        if self.feedback_revision == 1 and self.supersedes_revision is not None:
            raise ValidationError("revision 1 cannot supersede another revision")
        if self.feedback_revision > 1 and self.supersedes_revision != self.feedback_revision - 1:
            raise ValidationError("supersedes_revision must be the immediately previous revision")
        if self.status not in _ALLOWED_FEEDBACK_STATUS:
            raise ValidationError("status must be active or withdrawn")
        _tags(self.quick_tags)
        _freeform(self.freeform_reaction)
        _timestamp(self.created_at)
        if not self.quick_tags and not self.freeform_reaction:
            raise ValidationError("at least one of quick_tags or freeform_reaction is required")

    @classmethod
    def create(
        cls,
        *,
        feedback_id: str,
        session_id: str,
        turn_id: str,
        assistant_message_id: str | None,
        quick_tags: Sequence[str] | None,
        freeform_reaction: str | None,
        version_binding: VersionBinding,
        feedback_revision: int = 1,
        supersedes_revision: int | None = None,
        status: Literal["active", "withdrawn"] = "active",
        created_at: datetime | str | None = None,
        schema_version: int = 1,
        target_binding: Mapping[str, Any] | None = None,
    ) -> FeedbackEvent:
        tags = _tags(quick_tags)
        reaction = _freeform(freeform_reaction)
        if not tags and not reaction:
            raise ValidationError("at least one of quick_tags or freeform_reaction is required")
        return cls(
            schema_version=schema_version,
            feedback_id=_identifier(feedback_id, "feedback_id"),
            session_id=_identifier(session_id, "session_id"),
            turn_id=_identifier(turn_id, "turn_id"),
            assistant_message_id=(None if assistant_message_id is None else
                                  _identifier(assistant_message_id, "assistant_message_id")),
            quick_tags=tags,
            freeform_reaction=reaction,
            feedback_revision=feedback_revision,
            supersedes_revision=supersedes_revision,
            status=status,
            created_at=_timestamp(created_at),
            version_binding=version_binding,
            target_binding=None if target_binding is None else dict(target_binding),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FeedbackEvent:
        if not isinstance(value, Mapping):
            raise ValidationError("FeedbackEvent must be an object")
        return cls.create(
            feedback_id=value.get("feedback_id", ""),
            session_id=value.get("session_id", ""),
            turn_id=value.get("turn_id", ""),
            assistant_message_id=value.get("assistant_message_id", ""),
            quick_tags=value.get("quick_tags", ()),
            freeform_reaction=value.get("freeform_reaction"),
            version_binding=VersionBinding.from_mapping(value.get("version_binding", {})),
            feedback_revision=value.get("feedback_revision", 0),
            supersedes_revision=value.get("supersedes_revision"),
            status=value.get("status", "active"),
            created_at=value.get("created_at"),
            schema_version=value.get("schema_version", 1),
            target_binding=value.get("target_binding"),
        )

    def to_mapping(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "feedback_id": self.feedback_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "assistant_message_id": self.assistant_message_id,
            "quick_tags": list(self.quick_tags),
            "freeform_reaction": self.freeform_reaction,
            "feedback_revision": self.feedback_revision,
            "supersedes_revision": self.supersedes_revision,
            "status": self.status,
            "created_at": self.created_at,
            "version_binding": self.version_binding.to_mapping(),
        }
        if self.schema_version == 2:
            result["target_binding"] = self.target_binding
        return result


@dataclass(frozen=True, slots=True)
class FeedbackSelection:
    feedback_id: str
    feedback_revision: int

    def __post_init__(self) -> None:
        _identifier(self.feedback_id, "selected_feedback feedback_id")
        if self.feedback_revision < 1:
            raise ValidationError("selected feedback_revision must be positive")

    def to_mapping(self) -> dict[str, Any]:
        return {"feedback_id": self.feedback_id, "feedback_revision": self.feedback_revision}


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    preceding_turns: int = 2
    following_turns: int = 0

    def __post_init__(self) -> None:
        if self.preceding_turns < 0 or self.following_turns < 0:
            raise ValidationError("context turns cannot be negative")
        if self.preceding_turns > 2 or self.following_turns > 0:
            raise ValidationError("context policy exceeds the Phase B minimum export window")

    def to_mapping(self) -> dict[str, int]:
        return {
            "preceding_turns": self.preceding_turns,
            "following_turns": self.following_turns,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ContextPolicy:
        return cls(
            preceding_turns=int(value.get("preceding_turns", 2)),
            following_turns=int(value.get("following_turns", 0)),
        )


@dataclass(frozen=True, slots=True)
class FeedbackExport:
    schema_version: int
    export_id: str
    created_at: str
    selected_feedback: tuple[FeedbackSelection, ...]
    context_policy: ContextPolicy
    redaction_confirmed: bool
    external_processing_confirmed: bool
    payload: Mapping[str, Any]
    payload_sha256: str
    status: Literal["active", "stale"] = "active"

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError("unsupported FeedbackExport schema_version")
        _identifier(self.export_id, "export_id")
        _timestamp(self.created_at)
        if not self.selected_feedback:
            raise ValidationError("selected_feedback cannot be empty")
        if len({item.feedback_id for item in self.selected_feedback}) != len(
            self.selected_feedback
        ):
            raise ValidationError("selected_feedback cannot contain duplicate feedback IDs")
        if not self.redaction_confirmed or not self.external_processing_confirmed:
            raise ValidationError("export requires redaction and external processing confirmation")
        if not isinstance(self.payload, Mapping):
            raise ValidationError("export payload must be an object")
        if self.payload_sha256 != sha256_json(self.payload):
            raise ValidationError("payload_sha256 does not match canonical payload")
        if self.status not in _ALLOWED_EXPORT_STATUS:
            raise ValidationError("export status must be active or stale")

    @classmethod
    def create(
        cls,
        *,
        export_id: str,
        selected_feedback: Sequence[FeedbackSelection],
        context_policy: ContextPolicy,
        payload: Mapping[str, Any],
        redaction_confirmed: bool,
        external_processing_confirmed: bool,
        created_at: datetime | str | None = None,
    ) -> FeedbackExport:
        copied_payload = json.loads(json.dumps(payload, ensure_ascii=False))
        return cls(
            schema_version=SCHEMA_VERSION,
            export_id=_identifier(export_id, "export_id"),
            created_at=_timestamp(created_at),
            selected_feedback=tuple(selected_feedback),
            context_policy=context_policy,
            redaction_confirmed=redaction_confirmed,
            external_processing_confirmed=external_processing_confirmed,
            payload=copied_payload,
            payload_sha256=sha256_json(copied_payload),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FeedbackExport:
        if not isinstance(value, Mapping):
            raise ValidationError("FeedbackExport must be an object")
        selected = tuple(
            FeedbackSelection(
                feedback_id=item.get("feedback_id", ""),
                feedback_revision=item.get("feedback_revision", 0),
            )
            for item in value.get("selected_feedback", ())
        )
        return cls(
            schema_version=value.get("schema_version", 0),
            export_id=value.get("export_id", ""),
            created_at=value.get("created_at"),
            selected_feedback=selected,
            context_policy=ContextPolicy.from_mapping(value.get("context_policy", {})),
            redaction_confirmed=value.get("redaction_confirmed", False),
            external_processing_confirmed=value.get("external_processing_confirmed", False),
            payload=value.get("payload", {}),
            payload_sha256=value.get("payload_sha256", ""),
            status=value.get("status", "active"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "export_id": self.export_id,
            "created_at": self.created_at,
            "selected_feedback": [item.to_mapping() for item in self.selected_feedback],
            "context_policy": self.context_policy.to_mapping(),
            "redaction_confirmed": self.redaction_confirmed,
            "external_processing_confirmed": self.external_processing_confirmed,
            "payload": dict(self.payload),
            "payload_sha256": self.payload_sha256,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class FeedbackFinding:
    observation: str
    likely_layer: str
    hypothesis: str
    confidence: str
    evidence_feedback_ids: tuple[str, ...]
    alternative_explanations: tuple[str, ...]
    recommended_actions: tuple[str, ...]
    new_eval_cases: tuple[str, ...]
    needs_human_confirmation: bool

    def __post_init__(self) -> None:
        for field in (
            "observation",
            "hypothesis",
            "confidence",
        ):
            _identifier(getattr(self, field), field)
        if self.likely_layer not in _ALLOWED_LAYERS:
            raise ValidationError("likely_layer is not supported")
        if self.confidence not in _ALLOWED_CONFIDENCE:
            raise ValidationError("confidence is not supported")
        if not self.evidence_feedback_ids:
            raise ValidationError("analysis finding needs evidence_feedback_ids")
        for field in (
            "evidence_feedback_ids",
            "alternative_explanations",
            "recommended_actions",
            "new_eval_cases",
        ):
            values = getattr(self, field)
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise ValidationError(f"{field} must contain non-empty strings")
        if self.needs_human_confirmation is not True:
            raise ValidationError("analysis findings always require human confirmation")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "observation": self.observation,
            "likely_layer": self.likely_layer,
            "hypothesis": self.hypothesis,
            "confidence": self.confidence,
            "evidence_feedback_ids": list(self.evidence_feedback_ids),
            "alternative_explanations": list(self.alternative_explanations),
            "recommended_actions": list(self.recommended_actions),
            "new_eval_cases": list(self.new_eval_cases),
            "needs_human_confirmation": self.needs_human_confirmation,
        }


@dataclass(frozen=True, slots=True)
class FeedbackAnalysis:
    schema_version: int
    analysis_id: str
    input_export_id: str
    input_export_sha256: str
    analysis_model: str
    analysis_prompt_sha256: str
    generated_at: str
    findings: tuple[FeedbackFinding, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError("unsupported FeedbackAnalysis schema_version")
        for field in ("analysis_id", "input_export_id", "analysis_model"):
            _identifier(getattr(self, field), field)
        _sha(self.input_export_sha256, "input_export_sha256")
        _sha(self.analysis_prompt_sha256, "analysis_prompt_sha256")
        _timestamp(self.generated_at)
        if not self.findings:
            raise ValidationError("analysis findings cannot be empty")

    @classmethod
    def create(
        cls,
        *,
        analysis_id: str,
        input_export_id: str,
        input_export_sha256: str,
        analysis_model: str,
        analysis_prompt_sha256: str,
        findings: Sequence[FeedbackFinding],
        generated_at: datetime | str | None = None,
    ) -> FeedbackAnalysis:
        return cls(
            schema_version=SCHEMA_VERSION,
            analysis_id=_identifier(analysis_id, "analysis_id"),
            input_export_id=_identifier(input_export_id, "input_export_id"),
            input_export_sha256=input_export_sha256,
            analysis_model=analysis_model,
            analysis_prompt_sha256=analysis_prompt_sha256,
            generated_at=_timestamp(generated_at),
            findings=tuple(findings),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FeedbackAnalysis:
        findings = tuple(
            FeedbackFinding(
                observation=item.get("observation", ""),
                likely_layer=item.get("likely_layer", ""),
                hypothesis=item.get("hypothesis", ""),
                confidence=item.get("confidence", ""),
                evidence_feedback_ids=tuple(item.get("evidence_feedback_ids", ())),
                alternative_explanations=tuple(item.get("alternative_explanations", ())),
                recommended_actions=tuple(item.get("recommended_actions", ())),
                new_eval_cases=tuple(item.get("new_eval_cases", ())),
                needs_human_confirmation=item.get("needs_human_confirmation", False),
            )
            for item in value.get("findings", ())
        )
        return cls(
            schema_version=value.get("schema_version", 0),
            analysis_id=value.get("analysis_id", ""),
            input_export_id=value.get("input_export_id", ""),
            input_export_sha256=value.get("input_export_sha256", ""),
            analysis_model=value.get("analysis_model", ""),
            analysis_prompt_sha256=value.get("analysis_prompt_sha256", ""),
            generated_at=value.get("generated_at"),
            findings=findings,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "analysis_id": self.analysis_id,
            "input_export_id": self.input_export_id,
            "input_export_sha256": self.input_export_sha256,
            "analysis_model": self.analysis_model,
            "analysis_prompt_sha256": self.analysis_prompt_sha256,
            "generated_at": self.generated_at,
            "findings": [finding.to_mapping() for finding in self.findings],
        }


@dataclass(frozen=True, slots=True)
class DeletionReceipt:
    feedback_id: str
    deleted_revisions: int
    deleted_exports: int
    deleted_analyses: int
    deleted_at: str
    contains_freeform_text: Literal[False] = False


@dataclass(frozen=True, slots=True)
class ExportPreview:
    selected_feedback: tuple[FeedbackSelection, ...]
    context_policy: ContextPolicy
    payload: Mapping[str, Any]
    expected_revisions: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.selected_feedback:
            raise ValidationError("export preview cannot be empty")
        if not isinstance(self.payload, Mapping) or not self.payload:
            raise ValidationError("export preview payload cannot be empty")
