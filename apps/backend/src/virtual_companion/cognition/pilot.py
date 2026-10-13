"""A2 isolated preparation route for stable, complete synthetic text stimuli.

Persists an auditable pilot decision through the existing state transaction. This
is not the B scheduler, a replacement Persona, or the full legacy affect evolver.
The model still appraises and directs; deterministic rules only bind evidence,
preserve the fixed fallback boundaries and the narrow social fast paths.
"""

import json
import math
import re
from datetime import UTC, datetime, timedelta

from virtual_companion.boundary.contracts import Scope, StateEvent, canonical, identifier
from virtual_companion.boundary.runtime import KernelFailure

EVENTS = {
    "routine",
    "disclosure",
    "reliability",
    "boundary_respect",
    "repair",
    "conflict",
    "reciprocity",
}
IMPACTS = ("familiarity", "trust", "respect", "tension", "reciprocity")
EMOTIONS = set(
    (
        "calm joy affection amusement curiosity surprise shyness embarrassment anticipation relief "
        "pride worry fear startle sadness loneliness hurt disappointment frustration annoyance "
        "anger jealousy disgust guilt shame desire"
    ).split()
)
MODES = set(
    (
        "neutral warm joyful playful curious surprised shy embarrassed tender flirtatious "
        "clingy relieved hopeful worried startled afraid sad lonely hurt jealous disgusted "
        "guilty tired guarded concerned annoyed angry cold"
    ).split()
)


def number(value, minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Finite numeric evidence required")
    if minimum is not None and not minimum <= value <= maximum:
        raise ValueError("Evidence outside fixed range")
    return value


def appraise(raw, input_id, *, failure=None):
    identifier(input_id)
    noop = {
        "event": {
            "event_type": "routine",
            "source_input_ids": [input_id],
            "confidence": 0,
            "novelty": 0,
        },
        "affect_delta": {},
        "emotion": None,
        "fallback_reason": failure or "routine_or_unclear",
        "source_rebound": False,
    }
    if failure:
        return noop
    try:
        if not isinstance(raw, dict):
            raise ValueError("Appraisal object required")
        canonical(raw)
        event = dict(raw.get("event", {}))
        kind = event.get("event_type", "routine")
        if kind not in EVENTS:
            raise ValueError("Known evidence event required")
        confidence = number(event.get("confidence", 0), 0, 1)
        novelty = number(event.get("novelty", 0), 0, 1)
        impacts = {f"{k}_impact": number(event.get(f"{k}_impact", 0), -1, 1) for k in IMPACTS}
        delta = raw.get("affect_delta", {})
        if not isinstance(delta, dict) or len(delta) > 8:
            raise ValueError("Bounded affect evidence required")
        for key, value in delta.items():
            identifier(key)
            number(value)
        emotion = raw.get("emotion")
        emotion_valid = False
        if emotion is not None:
            if not isinstance(emotion, dict) or emotion.get("primary", "calm") not in EMOTIONS:
                raise ValueError("Known emotion required")
            intensity = number(emotion.get("intensity", 0.2), 0, 1)
            emotion_valid = (
                number(emotion.get("confidence", 0.5), 0, 1) >= 0.2 and intensity >= 0.12
            )
        relationship_valid = kind != "routine" and confidence >= 0.2 and novelty >= 0.1
        if (
            not relationship_valid
            and not emotion_valid
            and not any(abs(v) >= 0.01 for v in delta.values())
        ):
            return noop
        rebound = event.get("source_input_ids", []) != [input_id]
        result_event = {
            **event,
            "event_type": kind if relationship_valid else "routine",
            "source_input_ids": [input_id],
            "confidence": confidence if relationship_valid else 0,
            "novelty": novelty if relationship_valid else 0,
            **(impacts if relationship_valid else {k: 0 for k in impacts}),
        }
        if emotion_valid:
            rebound |= emotion.get("source_input_ids", []) != [input_id]
            emotion = {**emotion, "source_input_ids": [input_id]}
        return {
            "event": result_event,
            "affect_delta": dict(delta),
            "emotion": emotion if emotion_valid else None,
            "fallback_reason": None,
            "source_rebound": rebound,
        }
    except (ValueError, TypeError):
        return {**noop, "fallback_reason": "invalid_appraisal"}


def engagement(text, availability, input_id, now_us):
    """Fixed control subset: complete must-answer stimulus, no floor continuation."""
    if availability not in {"available", "busy", "sleeping"}:
        raise ValueError("Known availability required")
    identifier(input_id)
    repair = re.search(
        r"(?:为什么|怎么)(?:一直|还|都)?(?:不说话|没说话|不回|没回)|(?:你)?在听吗|人呢|是不是不想理我",
        text,
    )
    quiet = re.search(r"不用回|不用回复|别回|不要回复|先别说话", text)
    safety = re.search(r"不想活|想死|自杀|结束自己的生命|伤害自己|活着没意思", text)
    action = (
        "respond_now"
        if repair
        else "respect_silence"
        if quiet and not safety
        else "schedule_later"
        if availability != "available"
        else "respond_now"
    )
    obligations = (
        []
        if action == "respect_silence"
        else [
            {
                "source_input_id": input_id,
                "status": "delayed" if action == "schedule_later" else "pending",
                "due_us": now_us + 3_000_000 if action == "schedule_later" else now_us,
                "reason": f"character_{availability}"
                if action == "schedule_later"
                else "response_obligation",
            }
        ]
    )
    return {"action": action, "obligations": obligations}


def direct(context, raw, *, routine=False, failure=None):
    state, relationship = context["state_revision"], context["relationship_revision"]
    if any(type(v) is not int or v < 0 for v in (state, relationship)):
        raise ValueError("Explicit director versions required")
    base = {
        "state_revision": state,
        "relationship_revision": relationship,
        "mode": "neutral",
        "fallback_reason": None,
    }
    if routine:
        if (
            context.get("user_act") != "greeting"
            or context.get("appraisal_requires_model", False)
            or context.get("interruption_state", "none") != "none"
            or any(k != "opening_return" for k in context.get("obligation_types", []))
            or context.get("emotion_intensity", 0.2) > 0.55
            or context.get("tension", 0.05) > 0.25
            or any(c != "none" for c in context.get("scene_cues", []))
        ):
            return None
        warm = context.get("valence", 0.15) >= 0.08 and context.get("social_openness", 0.65) >= 0.5
        return {
            **base,
            "mode": "warm" if warm else "neutral",
            "fallback_reason": "deterministic_routine",
            "target": "user",
            "verbosity": "terse",
        }
    if failure:
        return {**base, "fallback_reason": failure}
    if not isinstance(raw, dict) or any(
        type(raw.get(k)) is not int or raw[k] != base[k]
        for k in ("state_revision", "relationship_revision")
    ):
        return {**base, "fallback_reason": "invalid_or_stale_output"}
    # This pilot requests a deliberately small directive; unsupported structures
    # stay an explicit fallback rather than masquerading as migrated full assets.
    if set(raw) - {"state_revision", "relationship_revision", "mode", "instructions"}:
        return {**base, "fallback_reason": "invalid_or_stale_output"}
    for key in ("mode", "instructions"):
        if key in raw and (not isinstance(raw[key], str) or len(raw[key]) > 1000):
            return {**base, "fallback_reason": "invalid_or_stale_output"}
    if raw.get("mode", "neutral") not in MODES:
        return {**base, "fallback_reason": "invalid_or_stale_output"}
    return {**base, **raw}


class PilotPreparation:
    def __init__(self, *, user_act="share", availability=None):
        if availability is not None and availability not in {"available", "busy", "sleeping"}:
            raise ValueError("Known explicit pilot availability required")
        if user_act not in {
            "identity_probe",
            "self_introduction",
            "follow_up",
            "question",
            "request",
            "challenge",
            "affection",
            "conflict",
            "greeting",
            "share",
            "silence",
            "unclear",
        }:
            raise ValueError("Known pilot semantic act required")
        self.user_act, self.availability = user_act, availability

    def prepare(self, context, calls):
        data = json.loads(context.payload)
        decision = engagement(
            context.current_text,
            self.availability or context.state.availability or "available",
            context.claim.input_id,
            data["observed_at_us"],
        )

        def request(stage, contract, **extra):
            result = calls.request(
                {
                    "stage": stage,
                    "schema_version": 1,
                    "context_sha256": context.fingerprint,
                    "context": data,
                    "response_contract": {"context_sha256": "echo input hash", "output": contract},
                    **extra,
                }
            )
            if result.get("context_sha256") != context.fingerprint or set(result) != {
                "context_sha256",
                "output",
            }:
                raise KernelFailure("preparation_binding")
            return result["output"]

        try:
            raw = request(
                "appraise",
                {
                    "event": "typed relationship evidence and source_input_ids",
                    "affect_delta": "finite axis deltas",
                    "emotion": "optional grounded emotion",
                },
            )
            appraisal = appraise(raw, context.claim.input_id)
        except TimeoutError:
            appraisal = appraise(
                None, context.claim.input_id, failure="appraisal_invocation_failed"
            )
        except KernelFailure as error:
            if error.reason not in {"model_contract", "empty_model_result"}:
                raise
            appraisal = appraise(None, context.claim.input_id, failure="invalid_appraisal")
        director_context = {
            "state_revision": context.state.write_version,
            "relationship_revision": context.state.business_revision,
            "user_act": self.user_act,
            "interruption_state": "none"
            if (self.availability or context.state.availability or "available") == "available"
            else "busy",
        }
        prior = json.loads(context.state.values_json)
        emotion = appraisal["emotion"] or prior.get("emotion", {})
        director_context.update(
            emotion_intensity=number(emotion.get("intensity", 0.2), 0, 1),
            tension=number(
                prior.get("relationship", {}).get("tension", {}).get("value", 0.05), 0, 1
            ),
            valence=number(prior.get("mood", {}).get("valence", 0.15), -1, 1),
            social_openness=number(prior.get("mood", {}).get("social_openness", 0.65), 0, 1),
            appraisal_requires_model=(
                appraisal["event"]["event_type"] != "routine"
                or bool(appraisal["affect_delta"])
                or appraisal["emotion"] is not None
            ),
        )
        directive = direct(director_context, None, routine=True)
        if directive is None:
            try:
                raw = request(
                    "direct",
                    {
                        "state_revision": "echo state version",
                        "relationship_revision": "echo relationship version",
                        "mode": "expression mode",
                        "instructions": "expression direction",
                    },
                    director_context=director_context,
                    appraisal=appraisal,
                    engagement=decision,
                )
                directive = direct(director_context, raw)
            except TimeoutError:
                directive = direct(director_context, None, failure="director_invocation_failed")
            except KernelFailure as error:
                if error.reason not in {"model_contract", "empty_model_result"}:
                    raise
                directive = direct(director_context, None, failure="invalid_or_stale_output")
        observation = {"appraisal": appraisal, "engagement": decision, "directive": directive}
        return {
            "context_sha256": context.fingerprint,
            "engagement": {
                "mode": {
                    "respond_now": "respond",
                    "respect_silence": "silence",
                    "schedule_later": "deferred",
                }[decision["action"]],
                "reason": decision["action"],
            },
            "appraisal": {
                "source_input_id": context.claim.input_id,
                "state_version": context.state.write_version,
                "observation": canonical(observation),
            },
            "director": {
                "state_version": context.state.write_version,
                "instructions": canonical(directive),
            },
        }

    def duty_sources(self, prepared):
        data = json.loads(prepared["appraisal"]["observation"])
        fallback = data["directive"]["fallback_reason"]
        return {
            "engagement": ("deterministic_engagement", None),
            "state_appraisal": (
                data["appraisal"]["fallback_reason"] or "model_appraisal",
                "appraise",
            ),
            "expression_director": (
                fallback or "model_director",
                None if fallback == "deterministic_routine" else "direct",
            ),
        }

    def project(self, context, prepared):
        observed = json.loads(prepared["appraisal"]["observation"])
        values = json.loads(context.state.values_json)
        previous = values.get("a2_pilot", {})
        pending = previous.get("pending_obligations", [])
        if not isinstance(pending, list):
            raise ValueError("Pilot pending obligation capacity exceeded")
        pending = [p for p in pending if p["source_input_id"] != context.claim.input_id]
        pending.extend(o for o in observed["engagement"]["obligations"] if o["status"] == "delayed")
        if len(pending) > 100:
            raise ValueError("Pilot pending obligation capacity exceeded")
        if "revision" in values:
            if (
                type(values["revision"]) is not int
                or values["revision"] != context.state.business_revision
            ):
                raise ValueError(
                    "Incompatible business projection; explicit reconciliation required"
                )
            values["revision"] += 1
        values["a2_pilot"] = {**observed, "pending_obligations": pending}
        now = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(
            microseconds=json.loads(context.payload)["observed_at_us"]
        )
        return StateEvent(
            Scope(context.claim.companion_id, context.claim.session_id),
            context.claim.attempt_id + "-pilot",
            "state",
            now,
            now,
            values_json=canonical(values),
            business_revision=context.state.business_revision + 1,
        )
