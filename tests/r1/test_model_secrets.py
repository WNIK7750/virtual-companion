import os

import pytest
from virtual_companion.providers.windows_secrets import WindowsSecretStore


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI target-environment contract")
def test_real_dpapi_reopen_is_encrypted_and_bound_to_endpoint(tmp_path):
    store = WindowsSecretStore(tmp_path / "vault")
    endpoint = "https://synthetic.example/v1/chat/completions"
    value = "synthetic-fixture-credential-never-an-account-key"
    ref = store.put(value, endpoint)
    assert ref.startswith("dpapi:")
    assert all(value.encode() not in p.read_bytes() for p in store.root.iterdir())
    assert WindowsSecretStore(store.root).get(ref, endpoint) == value
    with pytest.raises(ValueError, match="decrypt"):
        store.get(ref, "https://other.example/v1/chat/completions")
    store.remove(ref)
    with pytest.raises(ValueError, match="unavailable"):
        store.get(ref, endpoint)


def test_vault_references_cannot_traverse_or_store_plain_keys_in_repository(tmp_path):
    vault = WindowsSecretStore(tmp_path / "vault")
    with pytest.raises(ValueError, match="reference"):
        vault.get("../../secrets", "https://synthetic.example")
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError, match="outside"):
        WindowsSecretStore(tmp_path / "in-repository")
