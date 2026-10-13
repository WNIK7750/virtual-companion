import json
from dataclasses import asdict

import pytest
from test_text_provider_model import Counter, model
from virtual_companion.providers.budget import ProviderBudgetLedger
from virtual_companion.text.provider_model import load_provider_model


@pytest.fixture
def configured(tmp_path, monkeypatch):
    from virtual_companion.providers import windows_secrets
    from virtual_companion.text import provider_counter

    protected = tmp_path / "managed.bin"
    protected.write_bytes(b"synthetic-ciphertext")

    class Secrets:
        def __init__(self, path):
            pass

        def _path(self, reference):
            return protected

        def get(self, *args):
            raise AssertionError("Selecting a route must not decrypt or expose credentials")

    monkeypatch.setattr(windows_secrets, "WindowsSecretStore", Secrets)
    monkeypatch.setattr(provider_counter, "DeepSeekCounter", lambda _: Counter())
    value, _ = model()
    policy = {
        "id": "provider-config-test",
        "max_calls": 10,
        "max_input_tokens": 10000,
        "max_output_tokens": 5120,
        "max_cost_microusd": 10000,
        "routes": {value.route.fingerprint: {"kind": "api", "input_rate": 1, "output_rate": 1}},
    }
    prior = tmp_path / "prior.sqlite3"
    ProviderBudgetLedger(prior, policy)
    config = {
        "revision": 1,
        "data_root": str(tmp_path),
        "route": asdict(value.route),
        "secret_store": str(tmp_path),
        "tokenizer": "synthetic-tokenizer",
        "ledger": str(tmp_path / "current.sqlite3"),
        "policy": policy,
        "prior_ledgers": [str(prior)],
        "cumulative_token_limit": 100000000,
        "output_tokens": 512,
        "timeout_seconds": 45,
    }
    path = tmp_path / "provider.json"

    def write():
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    return config, write, protected, tmp_path


def test_explicit_configuration_selects_without_reading_key_or_sending(configured):
    config, write, _, root = configured
    value = load_provider_model(write(), data_root=root)
    assert value.route.model == "deepseek-flash"
    assert value.config_revision == 1
    assert value.transport.ledger.ledger.ledger.summary()["reserved_calls"] == 0


@pytest.mark.parametrize("failure", ["missing_key", "scope", "budget", "capability", "data_root"])
def test_invalid_configuration_never_reserves_or_sends(configured, failure):
    config, write, protected, root = configured
    if failure == "missing_key":
        protected.unlink()
    elif failure == "scope":
        config["route"]["data_scope"] = ["current"]
    elif failure == "budget":
        config["policy"]["max_calls"] = 0
    elif failure == "capability":
        config["route"]["capabilities"]["structured_output"] = False
    else:
        config["data_root"] = str(root / "another-root")
    with pytest.raises(ValueError):
        load_provider_model(write(), data_root=root)
