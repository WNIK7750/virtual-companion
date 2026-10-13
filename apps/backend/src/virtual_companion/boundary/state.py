"""Pure online-state transition; no model, storage, wall clock or automatic decay policy."""

from dataclasses import replace

from virtual_companion.text.chronology import utc_microseconds

from .contracts import StateEvent, StateReceipt, StateSnapshot, VersionConflict


def transition(before: StateSnapshot, event: StateEvent):
    if before.companion_id != event.scope.companion_id:
        raise ValueError("State scope mismatch")
    moment = utc_microseconds(event.occurred_at)
    after = replace(before, write_version=before.write_version + 1, sequence=before.sequence + 1)
    disposition = "applied"
    period_action = None
    if before.watermark_us is not None and moment < before.watermark_us:
        disposition = "late"
    else:
        after = replace(after, watermark_us=moment)
        if event.kind == "presence" and event.availability != before.availability:
            after = replace(after, availability=event.availability, period_id=event.fingerprint)
            period_action = "replace"
        elif event.kind == "interrupt":
            disposition = (
                "interrupted" if before.availability in {"busy", "sleeping"} else "available"
            )
            if before.availability is None:
                disposition = "uninitialized"
            if disposition == "interrupted":
                period_action = "increment"
        elif event.kind == "state":
            if event.business_revision < before.business_revision:
                raise VersionConflict("Business revision cannot move backwards")
            if (
                before.decay_watermark_us == moment
                and event.business_revision == before.business_revision
            ):
                disposition = "same_time"
            else:
                after = replace(
                    after,
                    decay_watermark_us=moment,
                    values_json=event.values_json,
                    business_revision=event.business_revision,
                )
    receipt = StateReceipt(
        event.scope.companion_id,
        event.scope.session_id,
        event.event_id,
        event.fingerprint,
        after.sequence,
        after.write_version,
        disposition,
        after.period_id,
        event.occurred_at.isoformat(),
        event.received_at.isoformat(),
    )
    return after, receipt, period_action
