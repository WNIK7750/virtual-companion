import json
from pathlib import Path

import pytest
from virtual_companion.text.provider_selection import ProviderSelection

ROOT = Path(__file__).parents[2]


def selection():
    value = ProviderSelection(
        ROOT / "configs" / "text-baseline-provider.json",
        {"profile_id": "db022-synthetic-v1"},
    )
    # These switch regressions begin with an explicitly selected local route.
    value.selected = value.bound_provider = "local"
    return value


def test_explicit_local_route_is_visible_without_secret_material():
    value = selection().state()
    assert value["selected_provider"] == "local"
    assert value["default_provider"] == "deepseek-api"
    assert value["active"]["model_profile"] == "db022-synthetic-v1"
    api = next(item for item in value["options"] if item["provider_id"] == "deepseek-api")
    assert api["selectable"] is False
    assert api["network_enabled"] is False
    assert "secret" not in json.dumps(value, ensure_ascii=False).lower()


def test_api_selection_is_explicitly_refused_without_switching_or_sending():
    value = selection()
    result = value.select("deepseek-api")
    assert result["status"] == "unavailable"
    assert result["provider"]["selected_provider"] == "local"
    assert "没有发送请求" in result["message"]


def test_local_selection_clears_api_error():
    value = selection()
    value.select("deepseek-api")
    result = value.select("local")
    assert result["status"] == "selected"
    assert result["provider"]["error"] == ""


def test_foreign_provider_selection_is_rejected():
    with pytest.raises(ValueError, match="Unknown provider"):
        selection().select("other")


def test_api_requires_confirmation_and_only_commits_after_binding_succeeds():
    value = selection()
    calls = []
    value.configured = True
    value.switch = calls.append
    assert value.select("deepseek-api")["status"] == "unavailable"
    assert calls == []
    assert value.select("deepseek-api", confirmed=True)["status"] == "selected"
    assert calls == ["deepseek-api"]
    assert value.state()["active"]["kind"] == "api"
    value.select("local")
    assert calls == ["deepseek-api", "local"]


def test_failed_binding_keeps_previous_route():
    value = selection()
    value.configured = True

    def fail(_):
        raise ValueError("private failure detail")

    value.switch = fail
    assert value.select("deepseek-api", confirmed=True)["status"] == "unavailable"
    assert value.selected == "local"
    assert "private failure" not in str(value.state())
