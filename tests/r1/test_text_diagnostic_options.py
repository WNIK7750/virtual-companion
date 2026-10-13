import hashlib

import pytest
from virtual_companion.text.__main__ import diagnostic_options, request_contract_sha256


def test_no_diagnostic_selection_keeps_current_prompt_binding():
    assert diagnostic_options({}) == {}
    assert request_contract_sha256("control", {}) == hashlib.sha256(b"control").hexdigest()


def test_review_and_history_are_independent_and_both_bind_queued_inputs():
    selected = diagnostic_options({
        "VC_TEXT_DIAGNOSTIC_REVIEW_AUDIT_VERSION": "candidate-unit-scope.v4",
        "VC_TEXT_DIAGNOSTIC_HISTORY_PROJECTION": "native-json.v1",
    })
    assert selected == {"review_audit_version": "candidate-unit-scope.v4",
                        "history_projection": "native-json.v1"}
    assert len({request_contract_sha256("control", options)
                for options in ({}, selected, {"review_audit_version": "candidate-unit-scope.v4"})
                }) == 3


@pytest.mark.parametrize("name", ["VC_TEXT_DIAGNOSTIC_REVIEW_AUDIT_VERSION",
                                 "VC_TEXT_DIAGNOSTIC_HISTORY_PROJECTION"])
def test_unknown_diagnostic_option_is_rejected_before_provider_loading(name):
    with pytest.raises(ValueError):
        diagnostic_options({name: "unknown"})
