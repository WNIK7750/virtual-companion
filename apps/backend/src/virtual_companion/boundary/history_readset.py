"""Versioned evidence dependencies of an actual admitted model context.

This is provenance, not a model verdict or authorization. Storage rechecks every
reference against its own current records before accepting a new segment.
"""

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256

from .contracts import canonical, companion_id, identifier

_TIME_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def context_time(observed_at_us):
    """One host UTC observation, not an event time, TTL renewal or model verdict."""
    if type(observed_at_us) is not int or not 0 <= observed_at_us < 2**63:
        raise ValueError("Host context time requires UTC epoch microseconds")
    try:
        observed = _TIME_EPOCH + timedelta(microseconds=observed_at_us)
    except OverflowError as error:
        raise ValueError("Host context time outside representable UTC range") from error
    return {
        "protocol": "host-context-time.v1", "source": "admission.evidence_clock",
        "clock": "utc_epoch_microseconds", "observed_at_us": observed_at_us,
        "observed_at_utc": observed.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "selection": "effective_at_observation",
    }


def validate_context_time(value, data):
    """Validate the sourced observation and selected absolute expiry boundaries.

    Context/readset fingerprints bind this metadata with the original evidence.
    Neither its field names nor its shape independently prove a live host source.
    """
    if not isinstance(value, dict) or value != context_time(value.get("observed_at_us")):
        raise ValueError("Exact host context time observation required")
    for item in [*data.get("memories", []), *data.get("exchanges", [])]:
        expiry = item.get("expires_at")
        if expiry is None:
            continue
        if not isinstance(expiry, str):
            raise ValueError("Context time requires an aware absolute expiry")
        try:
            instant = datetime.fromisoformat(expiry)
            if instant.tzinfo is None or instant.utcoffset() is None:
                raise ValueError("Naive context expiry")
            delta = instant.astimezone(UTC) - _TIME_EPOCH
            expiry_us = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
        except (ValueError, OverflowError) as error:
            raise ValueError("Context time requires an aware absolute expiry") from error
        if value["observed_at_us"] >= expiry_us:
            raise ValueError("Context time selected an expired record")
    return value


def evidence_hash(value):
    return sha256(canonical(value).encode()).hexdigest()


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def validate_evidence_access(value, *, facts=None, fact_count=None, retrieved_dialogue=None):
    """Host readonly access scope, never a completeness or semantic verdict."""
    from .memory_relevance import validate_recall_query

    dialogue = isinstance(value, dict) and value.get('protocol') == 'dialogue-memory.v1'
    navigation = isinstance(value, dict) and value.get("protocol") in (
        "deep-memory.v1",
        "dialogue-memory.v1",
    )
    fields = {
        "protocol", "scope", "coverage", "queries", "calls", "stop_reason", "limits",
    }
    if navigation:
        fields.add('navigation')
    if isinstance(value, dict) and 'query_contract' in value:
        if not navigation or value['query_contract'] != 'constrained-recall.v1':
            raise ValueError('Unknown readonly query contract')
        fields.add('query_contract')
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or (value["protocol"], value["scope"], value["coverage"])
        != (
            "dialogue-memory.v1"
            if dialogue
            else "deep-memory.v1"
            if navigation
            else "readonly-recall.v1",
            "current_committed_facts_and_dialogue" if dialogue else "current_committed_facts",
            "partial",
        )
    ):
        raise ValueError("Exact partial readonly evidence access required")
    limits = value["limits"]
    if not isinstance(limits, dict) or set(limits) != {"calls", "records", "evidence_bytes"}:
        raise ValueError("Exact readonly evidence limits required")
    for key, maximum in (("calls", 6 if navigation else 2),
                         ("records", 24), ("evidence_bytes", 64000)):
        if type(limits[key]) is not int or not 1 <= limits[key] <= maximum:
            raise ValueError("Bounded readonly evidence limits required")
    queries, calls, stop = value["queries"], value["calls"], value["stop_reason"]
    if (type(calls) is not int or not 0 <= calls <= limits["calls"]
            or not isinstance(queries, list) or len(queries) != calls
            or any(not isinstance(query, str) for query in queries)
            or (not navigation and len(set(queries)) != len(queries))
            or (not navigation and any(validate_recall_query(query) != query for query in queries))
            or (navigation and any(len(query.encode('utf-8')) > 2048 for query in queries))):
        raise ValueError("Exact executed readonly queries required")
    if navigation:
        page = value['navigation']
        if (not isinstance(page, dict) or page.get('protocol') != 'memory-catalog.v1'
                or not _digest(page.get('catalog_sha256'))
                or not isinstance(page.get('node_id'), str)
                or not isinstance(page.get('entries'), list) or len(page['entries']) > 8
                or page.get('catalog_coverage') not in ('complete', 'partial')
                or len(canonical(page).encode('utf-8')) > 8192):
            raise ValueError('Bounded memory navigation page required')
    if stop not in (None, "call_limit", "duplicate_query", "no_new_evidence", "empty",
                    "content_limit", "record_limit"):
        raise ValueError("Known readonly stop reason required")
    if ((stop is None and calls >= limits["calls"])
            or (stop == "call_limit" and calls != limits["calls"])
            or (stop not in (None, "content_limit", "record_limit") and calls == 0)):
        raise ValueError("Readonly stop reason differs from consumed calls")
    if facts is not None:
        if not isinstance(facts, list):
            raise ValueError("Complete readonly fact array required")
        fact_count = len(facts)
        originals = facts
        if dialogue:
            from .dialogue_retrieval import validate_fragments

            fragments = validate_fragments(retrieved_dialogue or [])
            fact_count += sum(len(fragment['members']) for fragment in fragments)
            originals = [*facts, *fragments]
        elif retrieved_dialogue:
            raise ValueError('Dialogue originals require the selected dialogue capability')
        evidence_bytes = (len(canonical(facts).encode('utf-8'))
                          + len(canonical(fragments).encode('utf-8')) - 2 if dialogue
                          else len(canonical(originals).encode('utf-8')))
        if evidence_bytes > limits["evidence_bytes"]:
            raise ValueError("Readonly original evidence exceeds byte limit")
    if (fact_count is not None and (type(fact_count) is not int
            or not 0 <= fact_count <= limits["records"])):
        raise ValueError("Readonly fact count exceeds executed access")
    return value


def validate_current_operation(value):
    """Exact content-free host outcome; this shape alone does not prove a commit."""
    if not isinstance(value, dict) or set(value) != {
        "protocol", "source_id", "source_sha256", "operation_id", "action", "status",
        "receipt_sha256",
    } or value["protocol"] != "memory-command-outcome.v1":
        raise ValueError("Known current memory operation outcome required")
    identifier(value["source_id"])
    identifier(value["operation_id"])
    if (value["action"] not in {"correct", "forget"}
            or value["status"] not in {"committed", "already_applied"}
            or not _digest(value["source_sha256"]) or not _digest(value["receipt_sha256"])):
        raise ValueError("Explicit successful current memory operation required")
    return value


def validate_source_processing(value, input_id):
    """Bound host lifecycle metadata; this shape is not completion authorization."""
    if not isinstance(value, dict) or set(value) != {
        "protocol", "source_id", "source_sha256", "status", "error", "receipt_sha256",
    } or value["protocol"] != "current-source-processing.v1":
        raise ValueError("Exact current source processing metadata required")
    identifier(value["source_id"])
    if value["source_id"] != input_id:
        raise ValueError("Source processing must belong to this current input")
    if value["status"] not in {"queued", "running", "completed", "retryable", "failed"} or (
        not _digest(value["source_sha256"])
        or value["receipt_sha256"] is not None and not _digest(value["receipt_sha256"])
        or value["error"] is not None and (
            not isinstance(value["error"], str) or not 1 <= len(value["error"]) <= 128)
    ):
        raise ValueError("Known source lifecycle and bounded receipt identity required")
    return value


def received_output_references(items):
    """Bind own received output without promoting it to shared history or facts."""
    if not isinstance(items, list) or not 1 <= len(items) <= 32:
        raise ValueError('Bounded received output required')
    references = []
    for item in items:
        if (not isinstance(item, dict) or set(item) != {
            'input_id', 'sequence', 'fingerprint', 'text', 'response_type', 'visibility',
        } or item['visibility'] != 'not_confirmed_at_admission'
                or not isinstance(item['text'], str) or not item['text']):
            raise ValueError('Exact received output observation required')
        references.append({key: item[key] for key in (
            'input_id', 'sequence', 'fingerprint', 'response_type')}
            | {'sha256': evidence_hash(item['text'])})
    return validate_received_references(references)


def validate_received_references(items):
    if not isinstance(items, list) or not 1 <= len(items) <= 32:
        raise ValueError('Bounded received output references required')
    seen = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {
            'input_id', 'sequence', 'fingerprint', 'response_type', 'sha256',
        }:
            raise ValueError('Exact received output reference required')
        identifier(item['input_id'])
        if (type(item['sequence']) is not int or item['sequence'] < 1
                or not _digest(item['fingerprint']) or not _digest(item['sha256'])
                or item['response_type'] not in {'answer', 'ack'}):
            raise ValueError('Bound received output identity required')
        identity = item['input_id'], item['sequence']
        if identity in seen:
            raise ValueError('Duplicate received output reference')
        seen.add(identity)
    return items


def validate_dialogue_references(items):
    if not isinstance(items, list) or len(items) > 24:
        raise ValueError('Bounded retrieved dialogue references required')
    seen = set()
    for item in items:
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "fragment_id",
                "version",
                "origin_session",
                "input_id",
                "config_sha256",
                "sha256",
                "members",
                "source_type",
                "part",
                "response_completeness", "membership_policy",
            }
            or item["version"] != 1
        ):
            raise ValueError('Exact retrieved dialogue reference required')
        if (item['source_type'] != 'dialogue_fragment' or type(item['part']) is not int
                or item['part'] < 0 or item['response_completeness'] != 'unknown'
                or item['membership_policy'] != 'frozen_confirmed_subset'):
            raise ValueError('Explicit dialogue type and part required')
        for key in ('fragment_id', 'origin_session', 'input_id'):
            identifier(item[key])
        if (
            item["fragment_id"] in seen
            or not _digest(item["sha256"])
            or not _digest(item["config_sha256"])
        ):
            raise ValueError('Unique retrieved dialogue digest required')
        seen.add(item['fragment_id'])
        if not isinstance(item['members'], list) or not 2 <= len(item['members']) <= 8:
            raise ValueError('Bounded exact dialogue original references required')
        member_ids = set()
        for index, member in enumerate(item['members']):
            if not isinstance(member, dict) or set(member) != {
                'source_id', 'role', 'sha256', 'response_type', 'visibility',
            }:
                raise ValueError('Exact dialogue member reference required')
            identifier(member['source_id'])
            if (
                member["source_id"] in member_ids
                or not _digest(member["sha256"])
                or member["role"] != ("user" if index == 0 else "assistant")
                or member["response_type"] not in ((None,) if index == 0 else ("answer", "ack"))
                or member["visibility"] != ("submitted" if index == 0 else "confirmed_visible")
            ):
                raise ValueError('Dialogue reference role, visibility or hash differs')
            member_ids.add(member['source_id'])
        if item['members'][0]['source_id'] != item['input_id']:
            raise ValueError('Dialogue reference input anchor differs')
    return items


def validate_readset(value):
    if isinstance(value, dict) and value.get('protocol') == 'history-readset.v8':
        from .public_capture import validate_public_capture

        extra = {'local_protocol', 'public_capture', 'public_capture_sha256', 'public_basis_sha256'}
        if not extra <= set(value) or value['local_protocol'] == 'history-readset.v8':
            raise ValueError('Complete bounded public readset required')
        local = {key: item for key, item in value.items() if key not in extra}
        local['protocol'] = value['local_protocol']
        validate_readset(local)
        validate_public_capture(value['public_capture'])
        if (evidence_hash(value['public_capture']) != value['public_capture_sha256']
                or not _digest(value['public_basis_sha256'])):
            raise ValueError('Public capture readset hash differs')
        return value
    fields = {
        "protocol", "companion", "input", "state_version", "context_sha256",
        "facts", "history", "exchanges",
    }
    if not isinstance(value, dict):
        raise ValueError("Known complete history readset required")
    if value.get("protocol") == "history-readset.v2":
        fields.add("current_operation")
    elif value.get("protocol") in {
        "history-readset.v3",
        "history-readset.v4",
        "history-readset.v5",
        "history-readset.v6",
        "history-readset.v7",
    }:
        if value['protocol'] in {'history-readset.v6', 'history-readset.v7'}:
            fields.add('evidence_access')
            if not isinstance(value.get('facts'), list):
                raise ValueError("Complete readonly readset facts required")
            if value['protocol'] == 'history-readset.v7':
                validate_dialogue_references(value.get('retrieved_dialogue'))
                if value.get('evidence_access', {}).get('protocol') != 'dialogue-memory.v1':
                    raise ValueError('Dialogue readset requires selected dialogue access')
            validate_evidence_access(
                value.get("evidence_access"),
                fact_count=len(value.get("facts", []))
                + (
                    sum(
                        len(fragment["members"]) for fragment in value.get("retrieved_dialogue", [])
                    )
                    if value["protocol"] == "history-readset.v7"
                    else 0
                ),
            )
            if value['protocol'] == 'history-readset.v7':
                fields.add('retrieved_dialogue')
                validate_dialogue_references(value.get('retrieved_dialogue'))
            if 'receiving_sha256' in value:
                fields.add('receiving_sha256')
        if value['protocol'] == 'history-readset.v5':
            fields.add('receiving_sha256')
        if (
            value["protocol"] == "history-readset.v4"
            or value["protocol"]
            in {"history-readset.v5", "history-readset.v6", "history-readset.v7"}
            and "received_output" in value
        ):
            fields.add('received_output')
            validate_received_references(value.get('received_output'))
        if value['protocol'] == 'history-readset.v3' or 'character_sha256' in value:
            fields.add("character_sha256")
        if "current_operation" in value:
            fields.add("current_operation")
    elif value.get("protocol") != "history-readset.v1":
        raise ValueError("Known complete history readset required")
    if set(value) != fields:
        raise ValueError("Known complete history readset required")
    companion_id(value["companion"])
    identifier(value["input"])
    if type(value["state_version"]) is not int or value["state_version"] < 0:
        raise ValueError("History readset state version required")
    if "current_operation" in value:
        operation = validate_current_operation(value["current_operation"])
        if operation["source_id"] != value["input"]:
            raise ValueError("Memory operation must belong to this current input")
    hashes = [value["context_sha256"]]
    if "character_sha256" in value:
        hashes.append(value["character_sha256"])
    if 'receiving_sha256' in value:
        hashes.append(value['receiving_sha256'])
    for group, maximum, fields, identity, version in (
        ("facts", 24 if value['protocol'] in {'history-readset.v6', 'history-readset.v7'} else 8,
         {"record_id", "version", "source_id", "sha256"}, "record_id", "version"),
        ("history", 32, {"id", "role", "sha256"}, "id", None),
        ("exchanges", 8, {"exchange_id", "revision", "source_id", "sha256"},
         "exchange_id", "revision"),
    ):
        values = value[group]
        if not isinstance(values, list) or len(values) > maximum:
            raise ValueError("Bounded history readset required")
        seen = set()
        for item in values:
            expected = fields
            if group == "history" and isinstance(item, dict) and "response_type" in item:
                expected = fields | {"response_type"}
                if item.get("role") != "assistant" or item["response_type"] != "ack":
                    raise ValueError("Only acknowledged assistant history carries an explicit type")
            if not isinstance(item, dict) or set(item) != expected:
                raise ValueError("Exact history readset reference required")
            identifier(item[identity])
            if item[identity] in seen:
                raise ValueError("Duplicate history readset reference")
            seen.add(item[identity])
            if version is not None:
                identifier(item["source_id"])
                if type(item[version]) is not int or item[version] < 1:
                    raise ValueError("Positive history reference version required")
            elif item["role"] not in {"user", "assistant"}:
                raise ValueError("History readset role required")
            hashes.append(item["sha256"])
    if any(not _digest(h) for h in hashes):
        raise ValueError("History readset digest required")
    return value


def history_readset(companion, input_id, state_version, payload, *, public_evidence=None,
                    public_basis=None):
    data = json.loads(payload)
    fields = {
        "history", "memories", "exchanges", "memory_status",
    }
    if (
        not isinstance(data, dict)
        or not fields <= set(data)
        or set(data)
        - fields
        - {
            "current_operation",
            "source_processing",
            "memory_access",
            "context_time",
            "character_context",
            "received_output",
            "receiving_context",
            "evidence_access",
            "retrieved_dialogue",
        }
    ):
        raise ValueError("Complete actual context required for history readset")
    if ("source_processing" in data) != ("memory_access" in data):
        raise ValueError("Current source and access projections must be supplied together")
    if "context_time" in data:
        validate_context_time(data["context_time"], data)
    if "source_processing" in data:
        validate_source_processing(data["source_processing"], input_id)
        if data["memory_access"] not in {
            "not_accessed", "completed", "unresolved", "needs_clarification",
        }:
            raise ValueError("Known current memory delegation status required")
    if data["memory_status"] not in {"current", "pending_source", "needs_clarification"}:
        raise ValueError("Known context memory status required")
    if "evidence_access" in data:
        validate_evidence_access(data["evidence_access"], facts=data["memories"],
                                 retrieved_dialogue=data.get("retrieved_dialogue"))
    for field, maximum in (("history", 32),
                           ("memories", 24 if "evidence_access" in data else 8), ("exchanges", 8)):
        if (not isinstance(data[field], list) or len(data[field]) > maximum
                or any(not isinstance(item, dict) for item in data[field])):
            raise ValueError("Bounded actual context evidence required")
    if any(set(r) not in ({"id", "role", "text"}, {"id", "role", "text", "response_type"})
           or not isinstance(r["text"], str) or ("response_type" in r
           and (r["role"] != "assistant" or r["response_type"] != "ack"))
           for r in data["history"]):
        raise ValueError("Exact actual history evidence required")
    value = {
        "protocol": "history-readset.v1", "companion": companion, "input": input_id,
        "state_version": state_version, "context_sha256": evidence_hash(data),
        "facts": [{"record_id": r["record_id"], "version": r["version"],
                   "source_id": r["source_id"], "sha256": evidence_hash(r)}
                  for r in data["memories"]],
        "history": [{"id": r["id"], "role": r["role"], "sha256": evidence_hash(r["text"]),
                     **({"response_type": "ack"} if r.get("response_type") == "ack" else {})}
                    for r in data["history"]],
        "exchanges": [{"exchange_id": r["exchange_id"], "revision": r["revision"],
                       "source_id": r["evidence"]["source_id"], "sha256": evidence_hash(r)}
                      for r in data["exchanges"]],
    }
    if "current_operation" in data:
        value.update(protocol="history-readset.v2",
                     current_operation=dict(validate_current_operation(data["current_operation"])))
    if "character_context" in data:
        from virtual_companion.text.character import CharacterSnapshot

        character = CharacterSnapshot.from_mapping(data["character_context"])
        if character.companion_id != companion:
            raise ValueError("Character companion differs from history readset")
        value.update(protocol="history-readset.v3", character_sha256=character.reference.sha256)
    if 'received_output' in data:
        value.update(protocol='history-readset.v4',
                     received_output=received_output_references(data['received_output']))
    if 'receiving_context' in data:
        from .receiving_context import validate_receiving_context

        receiving = validate_receiving_context(data['receiving_context'])
        value.update(protocol='history-readset.v5', receiving_sha256=evidence_hash(receiving))
    if 'evidence_access' in data:
        value.update(protocol='history-readset.v6', evidence_access=data['evidence_access'])
    if 'retrieved_dialogue' in data:
        from .dialogue_retrieval import fragment_reference

        value.update(
            protocol="history-readset.v7",
            retrieved_dialogue=[fragment_reference(row) for row in data["retrieved_dialogue"]],
        )
    validate_readset(value)
    if public_evidence is not None:
        from .public_capture import validate_public_capture, validate_public_scope

        capture = json.loads(public_evidence)
        validate_public_capture(capture)
        value.update(local_protocol=value['protocol'], protocol='history-readset.v8',
                     public_capture=capture, public_capture_sha256=evidence_hash(capture),
                     public_basis_sha256=evidence_hash(public_basis))
        validate_public_scope(value, public_basis)
    elif public_basis is not None:
        raise ValueError('Public basis requires an actual capture')
    return validate_readset(value)
