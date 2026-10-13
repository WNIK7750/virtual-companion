"""Deterministic display-only projection for versioned companion memories."""

import json
import re

_INTERNAL_KEYS = {
    "companion",
    "companion_id",
    "delivery_id",
    "hash",
    "id",
    "kind",
    "metadata",
    "mutation_id",
    "record",
    "record_id",
    "schema",
    "schema_version",
    "sha256",
    "source",
    "source_id",
    "status",
    "type",
    "version",
}
_CONTENT_KEYS = ("content", "text", "statement", "fact")
_SEMANTIC_KEYS = (
    "subject",
    "predicate",
    "value",
    "time",
    "date",
    "when",
    "condition",
    "qualifier",
    "uncertainty",
)
_SOURCE_LABELS = {
    "input": "来自对话",
    "background": "对话整理",
    "manual": "手动管理",
    "assessment": "状态评估",
    "output": "已确认回复",
}
_STATUS_LABELS = {
    "active": "正在使用",
    "suppressed": "已暂停使用，等待确认",
    "expired": "已到期，不再用于回答",
    "revoked": "已停止使用",
}


def _plain(value):
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value, flags=re.UNICODE).strip()


def _leaves(value, *, key=None):
    if isinstance(value, str):
        text = _plain(value)
        return [text] if text and key not in _INTERNAL_KEYS else []
    if isinstance(value, list):
        return [part for item in value for part in _leaves(item)]
    if isinstance(value, dict):
        return [
            part
            for child_key, child in value.items()
            if child_key not in _INTERNAL_KEYS
            for part in _leaves(child, key=child_key)
        ]
    return []


def _structured(value):
    if isinstance(value, str):
        return _plain(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            rendered = _structured(item)
            if rendered and rendered not in parts:
                parts.append(rendered)
        return "；".join(parts)
    if not isinstance(value, dict):
        return ""
    for key in _CONTENT_KEYS:
        if key in value:
            rendered = _structured(value[key])
            if rendered:
                return rendered
    semantic = []
    for key in _SEMANTIC_KEYS:
        if key in value:
            rendered = _structured(value[key])
            if rendered and rendered not in semantic:
                semantic.append(rendered)
    if semantic:
        return " · ".join(semantic)
    parts = []
    for part in _leaves(value):
        if part not in parts:
            parts.append(part)
    return " · ".join(parts)


def clean_memory_text(value):
    """Clean only the display copy; the caller retains the stored value unchanged."""
    if not isinstance(value, str):
        raise ValueError("Stored memory content must be text")
    text = _plain(value)
    if text[:1] in {"{", "[", '"'}:
        try:
            rendered = _structured(json.loads(value))
        except (TypeError, ValueError):
            rendered = ""
        if rendered:
            text = rendered
        elif text[:1] in {"{", "["}:
            text = "结构化记忆（请展开来源核对）"
    return text or "未提供可显示内容"


def project_memory(value):
    if not isinstance(value, dict):
        raise ValueError("Versioned memory mapping required")
    content = clean_memory_text(value["content"])
    summary = content if len(content) <= 180 else content[:179].rstrip() + "…"
    status = value["status"]
    return {
        "record_id": value["record_id"],
        "version": value["version"],
        "kind": value["kind"],
        "status": status,
        "status_label": _STATUS_LABELS.get(status, "状态待确认"),
        "content": content,
        "summary": summary,
        "truncated": summary != content,
        "event_time": value.get("event_time"),
        "event_time_label": (
            value["event_time"] if value.get("event_time") else "发生时间不详"
        ),
        "updated_at": value["occurred_at"],
        "updated_at_label": "录入/修订时间",
        "source_label": _SOURCE_LABELS.get(value["source_kind"], "来源待确认"),
        "actions": {
            "forget": status in {"active", "suppressed"},
            "correct": status in {"active", "suppressed"} and value["kind"] == "fact",
        },
    }


def source_label(kind):
    return _SOURCE_LABELS.get(kind, "来源待确认")
