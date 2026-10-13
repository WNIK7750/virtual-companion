"""The installed source tree cannot activate the retired predecessor branch."""

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RETIRED = (
    "native", "native_port", "native_protocol", "native_case", "business_context",
    "business_port", "business_preparation", "a3_candidate", "a3_preparation",
    "harness_native", "protocol", "neutral", "assets",
)


def test_retired_native_modules_are_not_importable_from_production():
    for name in RETIRED:
        assert importlib.util.find_spec(f"virtual_companion.cognition.{name}") is None, name


def test_production_source_has_no_import_or_path_activation_of_archive():
    for path in (ROOT / "apps/backend/src").rglob("*.py"):
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "predecessor" not in (node.module or ""), path
            if isinstance(node, ast.Import):
                assert all("predecessor" not in name.name for name in node.names), path
        assert "history/predecessor" not in source, path


def test_active_entrypoints_do_not_read_excluded_predecessor_assets():
    forbidden = ("解决“AI味”输出的探讨、设计和实践", "prompts/persona/base-register.v4.md")
    for directory in ("apps/backend/src", "scripts", "experiments"):
        for path in (ROOT / directory).rglob("*"):
            if (not path.is_file() or path.suffix not in {".py", ".ps1", ".json", ".mjs"}
                    or any(p in path.parts for p in (".venv", "node_modules", "__pycache__"))):
                continue
            source = path.read_text(encoding="utf-8-sig")
            assert not any(value in source for value in forbidden), path
