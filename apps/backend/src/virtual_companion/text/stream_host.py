"""Optional single-run host bridge, using the existing authenticated HTTP adapter.

The model thread waits outside SQLite. HTTP ACK commits before releasing it.
After timeout/restart, state exposes reconciliation candidates, never reoffers them.
"""

import threading
import time
from copy import deepcopy

from .transport import create_server
from .transport import validate as validate_payload


class StreamHostBridge:
    def __init__(self, *, store, claim, cancelled, timeout=5):
        if type(timeout) not in {int, float} or not 0 < timeout <= 10:
            raise ValueError("Bounded host receipt wait required")
        self.store, self.claim, self.cancelled, self.timeout = store, claim, cancelled, timeout
        self._condition = threading.Condition()
        self._pending = None
        self.ready = threading.Event()

    def present(self, event):
        with self._condition:
            if self._pending is not None:
                raise ValueError("Only one outstanding host delivery")
            rows = self.store.segments(self.claim.companion_id, self.claim.input_id)
            original = next((r for r in rows if r["fingerprint"] == event.get("fingerprint")), None)
            if original != event or original["status"] != "unknown":
                raise ValueError("Host may only receive the exact persisted dispatch")
            if self.cancelled.is_set():
                return None
            self._pending = deepcopy(event)
            self.ready.set()
            self._condition.notify_all()
            deadline = time.monotonic() + self.timeout
            try:
                while not self.cancelled.is_set():
                    rows = self.store.segments(self.claim.companion_id, self.claim.input_id)
                    current = rows[event["sequence"] - 1]
                    if current["status"] == "confirmed":
                        return current["host_receipt"]
                    if getattr(self.store, 'supports_received_text', False):
                        received = self.store.received_segment(
                            self.claim.companion_id, self.claim.input_id, event['sequence'])
                        if received:
                            return received
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(min(remaining, 0.025))
                return None
            finally:
                self._pending = None
                self.ready.clear()
                self._condition.notify_all()

    def poll(self, wait_ms=0, *, active=None):
        if type(wait_ms) is not int or not 0 <= wait_ms <= 1000:
            raise ValueError("Invalid bounded display wait")
        deadline = time.monotonic() + wait_ms / 1000
        with self._condition:
            while not self.cancelled.is_set() and (active is None or active()):
                if self._pending is not None:
                    rows = self.store.segments(self.claim.companion_id, self.claim.input_id)
                    if rows[self._pending["sequence"] - 1]["status"] == "unknown":
                        return {"event": deepcopy(self._pending)}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                # The offer/ACK/stop notify directly. External cancellation and
                # scope changes are also bounded even when no offer is pending.
                self._condition.wait(min(remaining, 0.025))
            return {"event": None}

    def receipt_committed(self):
        with self._condition:
            self._condition.notify_all()

    def ack(self, payload):
        if not isinstance(payload, dict) or set(payload) != {
            "sequence",
            "fingerprint",
            "host_receipt",
        }:
            raise ValueError("Exact scoped text receipt required")
        receipt = self.store.confirm_segment(
            self.claim.companion_id,
            self.claim.input_id,
            payload["sequence"],
            fingerprint=payload["fingerprint"],
            host_receipt=payload["host_receipt"],
            channel="text_visible",
        )
        with self._condition:
            self._condition.notify_all()
        return {"status": receipt["status"]}

    def state(self):
        return {
            "state": self.store.stream_state(self.claim.companion_id, self.claim.input_id),
            "reconcile": [
                event
                for event in self.store.segments(self.claim.companion_id, self.claim.input_id)
                if event["status"] == "unknown"
            ],
        }

    def stop(self):
        # Persist cancellation first. Unknown display intents remain reconcilable.
        self.store.cancel(self.claim.companion_id, self.claim.input_id)
        self.cancelled.set()
        with self._condition:
            self._condition.notify_all()
        return self.state()


def create_stream_server(token, bridge, *, port=0):
    def validate(action, payload):
        if action == "streamPoll":
            return validate_payload(action, payload)
        fields = {"sequence", "fingerprint", "host_receipt"} if action == "streamAck" else set()
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("Invalid scoped stream request")
        return payload

    def dispatch(action, payload):
        if action == "streamPoll":
            return bridge.poll(payload.get("wait_ms", 0))
        if action == "streamAck":
            return bridge.ack(payload)
        if action == "streamStop":
            return bridge.stop()
        return bridge.state()

    return create_server(
        token,
        dispatch,
        port=port,
        actions={"streamPoll", "streamAck", "streamState", "streamStop"},
        validate_payload=validate,
    )
