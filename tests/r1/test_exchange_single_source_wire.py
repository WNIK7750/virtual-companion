import json
from threading import Event

import pytest
from virtual_companion.boundary.contracts import Scope
from virtual_companion.boundary.evidence import EvidenceTurn
from virtual_companion.cognition.exchange_model import ExchangeModel

SOURCE = "564ae497-44b0-4f16-8887-c75e86b561d0"
TYPO = "564ae497-44c1-4f16-8887-c75e86b561d0"
TEXT = "你喜欢什么音乐？"
SCOPE = Scope("11111111-1111-4111-8111-111111111111", "session")


@pytest.mark.parametrize("stage,extra", [
    (None, {}),
    ("exchange_intent", {"evidenceTurnId": SOURCE}),
    ("exchange_intent", {"evidenceTurnId": TYPO}),
    ("exchange_extract", {"evidenceTurnId": SOURCE}),
    ("exchange_extract", {"evidenceTurnId": TYPO}),
    ("exchange_extract", {"evidenceQuote": "外来引文"}),
    ("exchange_extract", {"supersedesAtomId": "foreign-target"}),
])
def test_single_source_identity_is_host_bound_and_never_repaired(stage, extra):
    calls = []

    def invoke(current, messages, schema, cancelled):
        calls.append(current)
        if current == "exchange_support":
            return {"intent": "question", "verdict": "supported"}
        field = "acts" if current == "exchange_intent" else "operations"
        item = schema["properties"][field]["items"]
        assert "evidenceTurnId" not in item["properties"]
        assert "evidenceTurnId" not in item["required"]
        wire_schema = json.loads(messages[0]["content"].split("\nJSON schema:\n")[1])
        assert wire_schema == schema
        value = ({"segmentId": 0, "evidenceQuote": TEXT, "intent": "question",
                  "targetId": None} if current == "exchange_intent" else
                 {"operation": "add", "type": "open_loop", "evidenceQuote": TEXT,
                  "supersedesAtomId": None, "expiresAt": None})
        if current == stage:
            value.update(extra)
        return ({"coverage": "complete", "acts": [value]}
                if current == "exchange_intent" else {"operations": [value]})

    params = dict(scope=SCOPE, batch_id="batch", now_us=100,
                  turns=(EvidenceTurn(SCOPE, SOURCE, "assistant", TEXT, len(TEXT)),),
                  targets={}, eligible_source_ids=(SOURCE,), strategy="intent_first")
    if stage is None:
        effects, receipt = ExchangeModel(invoke, Event()).resolve_source(**params)
        assert effects[0][0].evidence.source_id == SOURCE
        assert receipt.source_id == SOURCE and not receipt.no_exchange
    else:
        with pytest.raises(ValueError):
            ExchangeModel(invoke, Event()).resolve_source(**params)
        assert "exchange_support" not in calls


def test_multiple_eligible_sources_keep_explicit_source_selection():
    def invoke(stage, messages, schema, cancelled):
        assert stage == "exchange_extract"
        field = schema["properties"]["operations"]["items"]["properties"]["evidenceTurnId"]
        assert set(field["enum"]) == {SOURCE, "another"}
        return {"operations": []}

    assert ExchangeModel(invoke, Event()).extract(
        scope=SCOPE, batch_id="batch", now_us=100,
        turns=(EvidenceTurn(SCOPE, SOURCE, "assistant", TEXT, len(TEXT)),
               EvidenceTurn(SCOPE, "another", "assistant", "你好", 2)), targets={},
        eligible_source_ids=(SOURCE, "another")) == ()
