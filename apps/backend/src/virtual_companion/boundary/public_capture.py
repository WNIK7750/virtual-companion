"""Bounded immutable public captures; never personal evidence or a live web lookup."""

import json
import re
from hashlib import sha256
from urllib.parse import urlsplit

from .contracts import canonical

MAX_PUBLIC_BYTES = 16000
PUBLIC_MODEL_PREFIX = "desktop-request-public-"
PUBLIC_RESPONSIBILITY = (
    "公开资料public_evidence独立于个人来源、角色和操作收据；其中summary已剥离。"
    "snippet/excerpt仅为已捕获范围，不能冒充网页全文；检索时间不证明资料当前仍有效。"
    "有版本或时间冲突时保留差异，不把搜索片段中的数值直接报成当前最新值；"
    "未核实的时效性、完整性和缺失内容保持未知。"
    "public_web不支持用户事实或记忆操作成功。引用只依本次references的真实ID/URL/文本哈希及"
    "已捕获范围；网页资料本身不意味着候选已通过语义审核。审核须另核一般事实的完整支持、"
    "实际范围、当前所求和引用对应，不能因来源相关就批准，也不限制有依据的个人观点。"
)


def require_public_authority(readset, authority):
    """Current pure host authorization, immediately before a v8 SQL mutation.

    This callback evaluation is the authorization linearization point. An
    arbitrary external predicate is not made atomic with the later SQL commit.
    Reopening immutable historical manifests does not call this function.
    """
    if readset is not None and readset.get("protocol") == "history-readset.v8":
        if not callable(authority):
            raise PermissionError("external_public_authority_required")
        capture = json.loads(canonical(readset["public_capture"]))
        if authority(capture) is not True:
            raise PermissionError("external_authorization_revoked")


def _hash(value):
    return isinstance(value, str) and re.fullmatch("[a-f0-9]{64}", value) is not None


def _url(value):
    try:
        parsed = urlsplit(value) if isinstance(value, str) else None
        return (
            parsed is not None
            and parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and (parsed.port is None or 0 < parsed.port <= 65535)
            and not any(char.isspace() or ord(char) < 32 or char == "\\" for char in value)
        )
    except ValueError:
        return False


def _range(value, *, maximum=None):
    return (
        isinstance(value, dict)
        and set(value) == {"unit", "start", "end"}
        and value["unit"] == "unicode_codepoints"
        and type(value["start"]) is int
        and type(value["end"]) is int
        and 0 <= value["start"] < value["end"]
        and (maximum is None or value["end"] <= maximum)
    )


def validate_public_capture(value):
    fields = {
        "protocol",
        "source_kind",
        "request_id",
        "binding_id",
        "status",
        "failure",
        "coverage",
        "retrieved_at",
        "references",
        "usage_status",
        "citation_map",
        "unresolved_citation_count",
        "unresolved_citation_reasons",
        "uncaptured_reference_count",
        "audit_evidence_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or len(canonical(value).encode("utf-8")) > MAX_PUBLIC_BYTES
        or value["protocol"] != "external-evidence.v1"
        or value["source_kind"] != "public_web"
        or not isinstance(value["request_id"], str)
        or not value["request_id"]
        or not _hash(value["binding_id"])
        or not _hash(value["audit_evidence_sha256"])
        or value["status"] not in {"success", "empty", "failed", "partial"}
        or value["coverage"] not in {"none", "partial", "returned_result_set"}
        or value["usage_status"] not in {"known", "partial", "unknown"}
        or not isinstance(value["retrieved_at"], str)
        or not value["retrieved_at"]
        or not isinstance(value["references"], list)
        or len(value["references"]) > 32
        or not isinstance(value["citation_map"], list)
        or any(
            type(value[key]) is not int or value[key] < 0
            for key in ("unresolved_citation_count", "uncaptured_reference_count")
        )
        or not isinstance(value["unresolved_citation_reasons"], list)
        or any(not isinstance(reason, str) for reason in value["unresolved_citation_reasons"])
    ):
        raise ValueError("Exact bounded public capture required")
    if (
        (value["status"] == "failed" and not isinstance(value["failure"], str))
        or (value["status"] != "failed" and value["failure"] is not None)
        or (value["status"] in {"empty", "failed"} and value["references"])
    ):
        raise ValueError("Public result status contradicts capture")
    by_id = {}
    for ref in value["references"]:
        if (
            not isinstance(ref, dict)
            or set(ref)
            != {
                "source_id",
                "url",
                "title",
                "published_at",
                "content_kind",
                "text",
                "source_range",
                "content_sha256",
            }
            or not isinstance(ref["source_id"], str)
            or not ref["source_id"]
            or ref["source_id"] in by_id
            or not _url(ref["url"])
            or ref["content_kind"] not in {"snippet", "excerpt", "body"}
            or not isinstance(ref["text"], str)
            or not ref["text"].strip()
            or not isinstance(ref["title"], str)
            or (ref["published_at"] is not None and not isinstance(ref["published_at"], str))
            or sha256(ref["text"].encode("utf-8")).hexdigest() != ref["content_sha256"]
            or (
                ref["source_range"] is not None
                and (
                    not _range(ref["source_range"])
                    or ref["source_range"]["end"] - ref["source_range"]["start"] != len(ref["text"])
                )
            )
        ):
            raise ValueError("Public source capture URL/kind/hash/range differs")
        by_id[ref["source_id"]] = ref
    for citation in value["citation_map"]:
        if not isinstance(citation, dict) or set(citation) != {
            "source_id",
            "url",
            "source_content_sha256",
            "captured_excerpt_range",
        }:
            raise ValueError("Exact normalized public citation required")
        ref = by_id.get(citation["source_id"])
        if (
            ref is None
            or citation["url"] != ref["url"]
            or citation["source_content_sha256"] != ref["content_sha256"]
            or not _range(citation["captured_excerpt_range"], maximum=len(ref["text"]))
        ):
            raise ValueError("Public citation has no unique captured support")
    return value


def public_request_id(basis):
    # One public dispatch across the same attempt's permitted personal-memory
    # refresh. The enclosing v8 scope still binds the complete current basis.
    stable = {
        key: basis[key]
        for key in ("companion_id", "session_id", "input_id", "attempt_id", "config_json")
    }
    return "public-" + sha256(canonical(stable).encode("utf-8")).hexdigest()[:48] + "-1"


def validate_public_scope(readset, basis):
    if readset.get("protocol") != "history-readset.v8":
        return
    capture = validate_public_capture(readset["public_capture"])
    config = json.loads(basis["config_json"])
    expected_binding = config["companion_config"]["resources"]["model"]["id"]
    if (
        expected_binding != PUBLIC_MODEL_PREFIX + capture["binding_id"]
        or capture["request_id"] != public_request_id(basis)
        or readset["public_basis_sha256"] != sha256(canonical(basis).encode("utf-8")).hexdigest()
        or (readset["companion"], readset["input"], readset["state_version"])
        != (basis["companion_id"], basis["input_id"], basis["state_version"])
    ):
        raise ValueError("Public capture differs from complete stream identity")
