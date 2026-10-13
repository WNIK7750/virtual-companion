"""Explicit diagnostic projection of admitted visible history into chat roles.

The immutable admission remains the reviewer/readset authority. Assistant
history uses the same JSON answer carrier as this desktop route; unseen received
output stays in the evidence background and is never asserted as visible speech.
"""

import json
from dataclasses import replace

from virtual_companion.boundary.contracts import canonical

NATIVE_HISTORY_PROJECTION = "native-json.v1"


def native_history_request(request, output_tokens, system, **rendering):
    data = json.loads(request.context.payload) if request.context is not None else {}
    history = data.get("history", [])
    if not history:
        return request.model_request(output_tokens, system, **rendering)
    # This copy is only a wire rendering input, never admitted, validated or stored.
    # No second copy of historical text remains in the evidence message.
    projected = replace(request, context=replace(
        request.context, payload=canonical({**data, "history": []})))
    body = projected.model_request(output_tokens, system, **rendering)
    references = [{"id": row["id"], "role": row["role"]} for row in history]
    native = []
    for row in history:
        if row["role"] not in {"user", "assistant"}:
            raise ValueError("Admitted user/assistant history required")
        content = (canonical({"action": row.get("response_type", "answer"), "text": row["text"]})
                   if row["role"] == "assistant" else row["text"])
        native.append({"role": row["role"], "content": content})
    body["messages"][-1:-1] = [{"role": "system", "content": (
        "下列历史消息按原说话方承载，均为已确认可见历史；引用目录只定位来源。"
        "助手历史的JSON仅承载当时原文，不证明用户事实或本轮操作。"
        "背景中received_output仍是可靠收到但未确认可见的自身表达。"
        + canonical({"history_projection": NATIVE_HISTORY_PROJECTION,
                     "source_references": references}))}, *native]
    return body
