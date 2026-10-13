from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = ROOT / "packages" / "contracts" / "feedback"


def test_versioned_feedback_schemas_are_strict_and_use_the_canonical_field() -> None:
    for name in (
        "feedback-event.v1.schema.json",
        "feedback-export.v1.schema.json",
        "feedback-analysis.v1.schema.json",
    ):
        schema = json.loads((SCHEMA_ROOT / name).read_text(encoding="utf-8"))
        assert schema["$schema"].endswith("draft/2020-12/schema")
        assert schema["additionalProperties"] is False
        assert "orthers" not in json.dumps(schema, ensure_ascii=False).casefold()


def test_domain_and_application_do_not_import_infrastructure_or_electron() -> None:
    for relative in (
        "apps/backend/src/virtual_companion/feedback/models.py",
        "apps/backend/src/virtual_companion/feedback/application.py",
        "apps/backend/src/virtual_companion/feedback/ports.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imported_names = [
            name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names
        ]
        assert not any(name in {"sqlite3", "electron", "fastapi"} for name in imports)
        assert not any(name.name in {"sqlite3", "electron", "fastapi"} for name in imported_names)


def test_feedback_application_has_no_network_or_provider_dependency() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "apps" / "backend" / "src").rglob("*.py")
        if "feedback" in path.parts
    )
    for forbidden in ("requests", "httpx", "urllib", "socket", "fetch(", "openai", "anthropic"):
        assert forbidden not in source.casefold()
