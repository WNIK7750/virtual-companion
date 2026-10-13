from __future__ import annotations

import json


def synthetic_unit_opinion(packet, **checks):
    """Explicit neutral opinion for known synthetic fixtures, not a classifier.

    A fixture with personal or fictional content must supply its own commitments.
    This helper performs no semantic analysis and is never a real-model fallback.
    """
    from virtual_companion.cognition.stream_unit_review import (
        AUDIT_VERSION,
        GROUNDED_UNIT_AUDIT_VERSION,
        UNIT_SCOPE_AUDIT_VERSION,
        USE_UNIT_AUDIT_VERSION,
        VERDICT_UNIT_AUDIT_VERSION,
    )

    if packet.get("audit_version") == VERDICT_UNIT_AUDIT_VERSION:
        supplied = synthetic_unit_opinion({**packet, "audit_version": USE_UNIT_AUDIT_VERSION})
        return {"audit_version": VERDICT_UNIT_AUDIT_VERSION, "decision": "approve",
                "checks": ["pass"] * 4, "failure": None, "units": supplied["units"]} | checks
    if packet.get("audit_version") == USE_UNIT_AUDIT_VERSION:
        supplied = synthetic_unit_opinion({**packet, "audit_version": UNIT_SCOPE_AUDIT_VERSION},
                                          **checks)
        return {**supplied, "audit_version": USE_UNIT_AUDIT_VERSION, "units": [dict(item,
            meaning="显式CPU供应商录制此完整合成单元无个人前提；不证明含义忠实。",
            use="显式CPU供应商录制当前回应关系。", response_fit="fits")
            for item in supplied["units"]]}
    if packet.get("audit_version") == GROUNDED_UNIT_AUDIT_VERSION:
        supplied = synthetic_unit_opinion({**packet, "audit_version": UNIT_SCOPE_AUDIT_VERSION},
                                          **checks)
        return {**supplied, "audit_version": GROUNDED_UNIT_AUDIT_VERSION,
                "units": [dict(item, meaning=(
                    "显式CPU供应商把此合成单元声明为无个人断言；"
                    "这是录制意见，不是语义准确性证明。")) for item in supplied["units"]]}
    if packet.get("audit_version") == UNIT_SCOPE_AUDIT_VERSION:
        units = packet["units"] if "units" in packet else packet["review_segments"]
        return dict(decision='approve', self_contained=True, no_open_dependency=True,
                    consistent_with_displayed=True, constraints_satisfied=True,
                    audit_version=UNIT_SCOPE_AUDIT_VERSION, units=[dict(
                        segment_id=item['segment_id'], personal_support='none',
                        personal_source_ids=[], fiction_support='none', fiction_source_ids=[])
                        for item in units]) | checks

    return dict(decision='approve', self_contained=True, no_open_dependency=True,
                consistent_with_displayed=True, constraints_satisfied=True,
                audit_version=AUDIT_VERSION, segments=[dict(
                    segment_id=item['segment_id'], commitments=[], questions=[])
                    for item in packet['review_segments']]) | checks


class FakeModel:
    def review_peer(self):
        return FakeModel()

    def check_context(self, messages):
        pass

    def stream(self, messages, cancelled):
        answer = "这是开发模式的合成回复。真实模型尚未参与这段对话。"
        system = messages[0].get('content', '') if messages else ''
        if '你只审查候选文字' in system:
            yield json.dumps(synthetic_unit_opinion(json.loads(messages[-1]['content'])))
            return
        if any(version in system for version in (
            'desktop-demand.v1', 'desktop-demand.v2', 'desktop-demand.v3',
            'desktop-demand.v4',
        )):
            answer = json.dumps({'action': 'answer', 'text': answer}, ensure_ascii=False)
        for character in answer:
            if cancelled.wait(0.025):
                return
            yield character

    def cancel(self):
        pass
