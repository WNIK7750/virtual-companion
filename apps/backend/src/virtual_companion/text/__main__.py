from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
import threading
from pathlib import Path

from virtual_companion.providers.searxng_search import DEFAULT_ENDPOINT
from virtual_companion.providers.searxng_search import VERSION as SEARXNG_VERSION

from . import default_runtime
from .application import TextSession
from .character import load_character
from .deep_memory import VERSION as DEEP_MEMORY_VERSION
from .default_runtime import DIAGNOSTIC_PROMPT_REVISIONS, DefaultTextRuntime
from .evidence_recall import VERSION as READONLY_RECALL_VERSION
from .fake import FakeModel
from .feedback import FeedbackBridge
from .identity import _identity
from .llama import LlamaModel
from .memory_runtime import DEFAULT_COMPANION_ID, TextMemoryRuntime
from .native_history import NATIVE_HISTORY_PROJECTION
from .profile import PROFILE_SHA256, load_profile
from .provider_model import load_provider_model
from .provider_selection import ProviderSelection
from .public_web import build_public_web_capability
from .runtime_skills import SUPPORTED_VERSIONS as MEMORY_SKILLS_VERSIONS
from .runtime_skills import VERSION as MEMORY_SKILLS_VERSION
from .runtime_skills import load_memory_skills
from .semantic_config import load_semantic_retriever
from .sqlite_store import SQLiteConversationStore
from .transport import create_server
from .windows_job import contain_current_process

_READY_HANDSHAKE_SENT = False
_STARTUP_CONFIGURATION_PHASE = False
_STARTUP_CONFIGURATION_ERRORS = (
    ValueError,
    OSError,
    ImportError,
    RuntimeError,
    sqlite3.Error,
)


def _api_binding(local_binding, model):
    if model.route.model != "deepseek-flash":
        raise ValueError("The desktop API route requires DeepSeek Flash")
    return dict(local_binding, schema_version=2, model_sha256=None, server_sha256=None,
                sampling_profile_id=model.route.wire_profile,
                provider_route=model.route.fingerprint, provider_revision=model.config_revision,
                provider_config=model.config_fingerprint, provider_model=model.route.model,
                provider_wire_profile=model.route.wire_profile)


def _api_identity(model):
    return {"config_revision": model.config_revision,
            "route_fingerprint": model.route.fingerprint,
            "config_fingerprint": model.config_fingerprint,
            "model_profile": model.route.wire_profile}


def diagnostic_options(environment):
    from virtual_companion.cognition.stream_unit_review import SUPPORTED_AUDIT_VERSIONS

    options = {}
    for key, name, allowed in (
        ("VC_TEXT_DIAGNOSTIC_REVIEW_AUDIT_VERSION", "review_audit_version",
         SUPPORTED_AUDIT_VERSIONS),
        ("VC_TEXT_DIAGNOSTIC_HISTORY_PROJECTION", "history_projection",
         (NATIVE_HISTORY_PROJECTION,)),
        ("VC_TEXT_DIAGNOSTIC_READONLY_RECALL", "readonly_recall",
         (READONLY_RECALL_VERSION, DEEP_MEMORY_VERSION, 'dialogue-memory.v1')),
        ('VC_TEXT_RUNTIME_MEMORY_SKILLS', 'memory_skills', MEMORY_SKILLS_VERSIONS),
    ):
        value = environment.get(key)
        if value is not None:
            if value not in allowed:
                raise ValueError("Unsupported diagnostic request selection")
            options[name] = value
    return options


def request_contract_sha256(control, options):
    from virtual_companion.boundary.contracts import canonical

    encoded = canonical([control, options]) if options else control
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_identity_sha256(value):
    from virtual_companion.boundary.contracts import canonical

    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _path_identity_sha256(value):
    resolved = Path(value).resolve()
    normalized = os.path.normcase(str(resolved))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main():
    global _READY_HANDSHAKE_SENT, _STARTUP_CONFIGURATION_PHASE
    _READY_HANDSHAKE_SENT = False
    runtime_options = diagnostic_options(os.environ)
    prompt_revision = os.environ.get("VC_TEXT_DIAGNOSTIC_PROMPT_REVISION")
    if prompt_revision is not None and prompt_revision not in DIAGNOSTIC_PROMPT_REVISIONS:
        raise ValueError("Unsupported diagnostic prompt revision")
    control = (default_runtime.CONTROL if prompt_revision is None
               else default_runtime.control_for_prompt_revision(prompt_revision))
    token = sys.stdin.readline().strip()
    if len(token) < 32:
        raise ValueError("Missing launch credential")
    job = contain_current_process()
    text, base_model, memory = None, None, None

    def watch_parent():
        # Pipe is only held by Electron: both normal exit and SIGKILL cause EOF.
        sys.stdin.read()
        try:
            if text is not None:
                text.stop()
            if base_model is not None:
                base_model.cancel()
            if text is not None and text.worker is not None:
                text.worker.join(5)
            if memory is not None:
                memory.close()
        except Exception:
            # A process death still leaves conservative reserved usage intact.
            print('{"event":"shutdown_incomplete"}', file=sys.stderr, flush=True)
        os._exit(0)  # Job handle closes atomically with backend process death.

    fake = "--fake" in sys.argv
    repository = Path(__file__).resolve().parents[5]
    profile = load_profile(repository / "configs/text-baseline-runtime.json")
    provider = ProviderSelection(repository / "configs/text-baseline-provider.json", profile)
    local_model = FakeModel() if fake else None
    if fake:
        provider.selected = provider.bound_provider = "local"
    data_root = Path(os.environ.get("VC_DATA_ROOT", str(repository / ".local/dogfood")))
    provider_config = data_root / "provider.json"
    memory_skills, skills_error = None, ""
    if runtime_options.get("memory_skills") is not None or runtime_options.get(
        "readonly_recall"
    ) in (DEEP_MEMORY_VERSION, "dialogue-memory.v1"):
        try:
            selected_skills = runtime_options.get('memory_skills') or MEMORY_SKILLS_VERSION
            memory_skills = (load_memory_skills() if selected_skills == MEMORY_SKILLS_VERSION
                             else load_memory_skills(selected_skills))
        except (ValueError, OSError):
            skills_error = "记忆 skill 配置无法校验；请恢复软件附带的 skill 资源后重启。"
    runtime_arguments = {**runtime_options,
        **({'loaded_memory_skills': memory_skills} if memory_skills is not None else {})}
    memory_arguments = {
        'memory_skills': runtime_options.get('memory_skills'),
        'loaded_memory_skills': memory_skills,
    }
    semantic_retriever, semantic_error = None, ""
    try:
        semantic_retriever = load_semantic_retriever(data_root / "semantic-retrieval.json")
    except (ValueError, OSError, ImportError, RuntimeError):
        semantic_error = ("语义记忆检索尚未就绪；请修复所选数据目录中的 "
                          "semantic-retrieval.json 和本地模型，或移除该候选配置后重启。")
    # Optional native initialization must finish before a concurrent stdin reader.
    threading.Thread(target=watch_parent, daemon=True).start()

    def api_configured():
        return (not fake and provider_config.is_file()
                and importlib.util.find_spec("tokenizers") is not None)

    provider.configured = api_configured()
    memory_override = os.environ.get("VC_MEMORY_MANAGEMENT_DB")
    companion_override = os.environ.get("VC_MEMORY_COMPANION_ID")
    if (memory_override is None) != (companion_override is None):
        raise ValueError("Memory management override requires an explicit database and companion")
    if memory_override is not None and not memory_override.strip():
        raise ValueError("Memory database path must not be empty")
    memory_companion = _identity(
        companion_override if companion_override is not None else DEFAULT_COMPANION_ID
    )
    character, character_error = None, ""
    try:
        character = load_character(data_root / "character.json", memory_companion)
    except (ValueError, OSError):
        character_error = ("角色配置无法校验或与当前伴侣不匹配；"
                           "请修复数据目录中的 character.json 后重启。")
    memory_path = memory_override or str(data_root / "memory.sqlite3")
    scope_identity = {
        "companion_id": memory_companion,
        "data_root_sha256": _path_identity_sha256(data_root),
        "memory_path_sha256": _path_identity_sha256(memory_path),
    }
    store = SQLiteConversationStore(data_root, companion_id=memory_companion)
    local_binding = {
        "app_commit": os.environ.get("VC_APP_COMMIT", "0" * 40),
        "schema_version": 1,
        "persona_sha256": (character.reference.sha256 if character is not None else
                           hashlib.sha256(b"synthetic-no-persona").hexdigest()),
        "prompt_sha256": request_contract_sha256(control, {
            **runtime_options,
            **({"semantic_retriever": semantic_retriever.identity}
               if semantic_retriever is not None else {}),
            **({'memory_skills_identity': memory_skills.identity}
               if memory_skills is not None else {}),
        }),
        "model_sha256": hashlib.sha256(b"fake").hexdigest()
        if fake
        else profile["artifacts"][1]["sha256"],
        "server_sha256": hashlib.sha256(b"fake").hexdigest()
        if fake
        else profile["artifacts"][0]["sha256"],
        "sampling_profile_id": profile["profile_id"],
    }
    local_state = {"state": "ready" if fake else "stopped",
                   "mode": "fake" if fake else "local", "error": ""}
    binding = dict(local_binding)
    if provider.selected == "local":
        if local_model is None:
            local_model = LlamaModel(profile, repository)
        base_model = local_model
        state = dict(local_state)
    else:
        base_model, binding = None, None
        state = {"state": "provider_unavailable", "mode": "api", "error": ""}
        try:
            if not provider.configured:
                raise ValueError("Managed provider configuration unavailable")
            candidate = load_provider_model(provider_config, data_root=data_root)
            binding = _api_binding(local_binding, candidate)
            base_model = candidate
            provider.active_identity = _api_identity(candidate)
            provider.bound_provider = provider.selected
            state.update(state="ready")
        except Exception:
            state["error"] = (
                "DeepSeek Flash 尚未就绪：请检查所选数据目录中的 provider.json、"
                "受管凭据、计数器和继承预算。没有发送请求，也未自动改用本地模型。"
            )
            provider.last_error = state["error"]

    def session_binding(value):
        return {"runtime_profile_sha256": PROFILE_SHA256,
                **({"version_binding": value} if value is not None else {})}

    memory = None
    memory_error = ""
    try:
        if character_error or semantic_error or skills_error:
            raise ValueError("Invalid character, semantic or skill configuration")
        memory = TextMemoryRuntime(Path(memory_path), memory_companion, model=base_model,
                                   fake=fake, character=character, **memory_arguments)
    except Exception:
        memory_error = (character_error or semantic_error or skills_error
                        or "记忆与对话权威库暂时无法打开；请恢复数据库后发送。")
    public_web_error = ""
    if memory is not None and not fake:
        try:
            web_backend = os.environ.get("VC_PUBLIC_WEB_BACKEND", "pi")
            if web_backend not in {"pi", "searxng"}:
                raise ValueError("Unknown public web backend")
            web_options = ({
                "searxng_endpoint": os.environ.get("VC_SEARXNG_ENDPOINT", DEFAULT_ENDPOINT),
                "searxng_version": os.environ.get("VC_SEARXNG_VERSION", SEARXNG_VERSION),
            } if web_backend == "searxng" else {})
            runtime_arguments["public_query"] = build_public_web_capability(
                memory.config.character.sha256, **web_options)
        except ValueError:
            public_web_error = "搜索服务配置无效，请检查 SearXNG 地址和固定版本。没有发送请求。"
    runtime = (
        DefaultTextRuntime(memory, store, base_model, binding=binding,
                           prompt_revision=prompt_revision, semantic_retriever=semantic_retriever,
                           **runtime_arguments)
        if memory is not None and base_model is not None
        else None
    )
    text = TextSession(
        base_model,
        store,
        session_binding(binding),
        prepare_turn=runtime.prepare_turn if runtime is not None else None,
        before_start=runtime.before_start if runtime is not None else None,
    )
    feedback = FeedbackBridge(
        store, receipt_store_provider=lambda: memory.store if memory is not None else None)
    if memory is not None:
        memory.attach_foreground(lambda: text.snapshot()["busy"])
    lifecycle_lock = threading.RLock()

    configuration_error = (
        character_error or semantic_error or skills_error or memory_error or public_web_error
    )
    if configuration_error:
        state.update(state="configuration_unavailable", error=configuration_error)

    def startup_report():
        route = getattr(base_model, "route", None)
        provider_component = {
            "status": "bound" if base_model is not None and binding is not None else "unavailable",
            "provider_id": provider.selected,
        }
        if route is not None:
            provider_component["route_fingerprint"] = getattr(route, "fingerprint", None)
            provider_component["model_id"] = getattr(route, "model", None)
        if base_model is not None:
            provider_component["config_fingerprint"] = getattr(
                base_model, "config_fingerprint", None)
            provider_component["config_revision"] = getattr(base_model, "config_revision", None)

        components = {
            "runtime_profile": {"status": "verified", "sha256": PROFILE_SHA256},
            "provider_route": provider_component,
            "role": {
                "status": ("unavailable" if character_error else
                           "configured" if character is not None else "absent"),
                "sha256": (character.reference.sha256 if character is not None else None),
            },
            "semantic_retrieval": {
                "status": ("unavailable" if semantic_error else
                           "configured" if semantic_retriever is not None else "disabled"),
                "identity_sha256": (
                    _safe_identity_sha256(semantic_retriever.identity)
                    if semantic_retriever is not None else None
                ),
            },
            "memory_authority": {
                "status": "unavailable" if memory is None else "ready",
            },
            "conversation_store": {"status": "ready"},
            "public_web": {
                "status": ("unavailable" if public_web_error else
                           "configured" if "public_query" in runtime_arguments else "disabled"),
                "identity_sha256": (
                    runtime_arguments["public_query"].binding.identity
                    if "public_query" in runtime_arguments else None
                ),
            },
            "scope": scope_identity,
            "request_contract": {
                "prompt_sha256": local_binding["prompt_sha256"],
                "selection_sha256": _safe_identity_sha256(runtime_options),
            },
        }
        blocked_reasons = []
        if memory_skills is not None or skills_error:
            components['runtime_skills'] = {
                'status': 'unavailable' if skills_error else 'configured',
                'protocol': (memory_skills.version if memory_skills else
                             runtime_options.get('memory_skills') or MEMORY_SKILLS_VERSION),
                'bundle_sha256': memory_skills.fingerprint if memory_skills else None,
                'skills': memory_skills.identity if memory_skills else [],
            }
        if skills_error:
            blocked_reasons.append('runtime_skills')
        if public_web_error:
            blocked_reasons.append("public_web_configuration")
        if character_error:
            blocked_reasons.append("role_configuration")
        if semantic_error:
            blocked_reasons.append("semantic_configuration")
        if memory_error and not (character_error or semantic_error or skills_error):
            blocked_reasons.append("memory_authority")
        if base_model is None or binding is None:
            blocked_reasons.append("provider_binding")
        if runtime is None and not configuration_error:
            blocked_reasons.append("text_runtime")
        if state["state"] != "ready" and not blocked_reasons:
            blocked_reasons.append(
                "model_loading" if state["state"] == "loading_model" else "model_not_ready")
        return {
            "status": ("configuration_unavailable" if configuration_error else "configured"),
            "can_send": (not configuration_error and state["state"] == "ready"
                         and runtime is not None and base_model is not None),
            "identity_sha256": _safe_identity_sha256(components),
            "components": components,
            "blocked_reasons": blocked_reasons,
        }

    def switch_provider(provider_id):
        nonlocal base_model, local_model, memory, runtime, text, binding
        if text.snapshot()["busy"] or state["state"] == "loading_model":
            raise ValueError("Provider switch requires idle foreground")
        if memory is None:
            raise ValueError("Provider switch requires available authority")
        if runtime is not None:
            runtime.before_start(False)
        candidate = (
            (local_model if local_model is not None else LlamaModel(profile, repository))
            if provider_id == "local"
            else load_provider_model(provider_config, data_root=data_root)
        )
        next_binding = (dict(local_binding) if provider_id == "local"
                        else _api_binding(local_binding, candidate))
        # A new owner is never attached until the old worker is fully stopped.
        if provider.selected == "local" and not provider.recovery_required:
            local_state.update(state)
        try:
            memory.close()
        except Exception:
            provider.recovery_required = True
            state.update(
                state="provider_recovery_required",
                error="后台尚未完成关闭；请在模型面板重新选择路由恢复。",
            )
            raise
        try:
            replacement = TextMemoryRuntime(
                Path(memory_path),
                memory_companion,
                model=candidate,
                fake=fake and provider_id == "local",
                character=character,
                **memory_arguments,
            )
            next_runtime = DefaultTextRuntime(replacement, store, candidate, binding=next_binding,
                                              prompt_revision=prompt_revision,
                                              semantic_retriever=semantic_retriever,
                                              **runtime_arguments)
            next_text = TextSession(
                candidate,
                store,
                session_binding(next_binding),
                prepare_turn=next_runtime.prepare_turn,
                before_start=next_runtime.before_start,
            )
        except Exception:
            # Restore background ownership on the same store; never fallback a request.
            try:
                restored = TextMemoryRuntime(
                    Path(memory_path),
                    memory_companion,
                    model=base_model,
                    fake=fake and provider.selected == "local",
                    character=character,
                    **memory_arguments,
                )
                restored_runtime = (DefaultTextRuntime(
                    restored, store, base_model, binding=binding, prompt_revision=prompt_revision,
                    semantic_retriever=semantic_retriever,
                    **runtime_arguments,
                ) if base_model is not None else None)
                restored_text = TextSession(
                    base_model,
                    store,
                    text.binding,
                    prepare_turn=restored_runtime.prepare_turn if restored_runtime else None,
                    before_start=restored_runtime.before_start if restored_runtime else None,
                )
                memory, runtime, text = restored, restored_runtime, restored_text
                memory.attach_foreground(lambda: text.snapshot()["busy"])
            except Exception:
                provider.recovery_required = True
                state.update(
                    state="provider_recovery_required",
                    error="原路由恢复未完成；请在模型面板重新选择路由恢复。",
                )
                raise
            raise
        # Provider HTTP clients are per-call contexts. The stopped worker owns
        # no live provider operation, so committing the new owner needs no I/O.
        base_model, memory, runtime, text, binding = (
            candidate,
            replacement,
            next_runtime,
            next_text,
            next_binding,
        )
        if provider_id == "local":
            local_model = candidate
        state.update(
            local_state
            if provider_id == "local"
            else {"state": "ready", "mode": "api", "error": ""}
        )
        memory.attach_foreground(lambda: text.snapshot()["busy"])
        return (
            {}
            if provider_id == "local"
            else _api_identity(candidate)
        )

    provider.switch = switch_provider

    def load_model():
        try:
            base_model.start()
            state.update(state="ready", error="")
        except Exception:
            base_model.close()
            state.update(state="failed", error="模型加载失败。请核对固定制品、CUDA 环境和端口。")

    queue_waiting = ''

    def pump_inputs():
        nonlocal queue_waiting
        if (configuration_error or state['state'] != 'ready'
                or runtime is None or text.snapshot()['busy']):
            return
        pending = store.queued_inputs()
        queue_waiting = ''
        if not pending or store.input_queue_paused():
            return
        frozen = pending[0]['binding']
        # A changed role may still serve its own frozen source. Provider/control
        # changes require restoring that binding rather than silently rerouting.
        def route(value):
            return {k: v for k, v in value.items() if k != 'persona_sha256'}
        if (frozen.get('runtime_profile_sha256') != text.binding.get('runtime_profile_sha256')
                or route(frozen.get('version_binding', {}))
                != route(text.binding.get('version_binding', {}))
                or frozen.get('request_control_sha256')
                != hashlib.sha256(runtime.control.encode()).hexdigest()):
            queue_waiting = '待处理消息的模型或规则版本已变化；原文已保存，请恢复原配置后继续。'
            return
        try:
            runtime.before_start(False)
        except ValueError:
            queue_waiting = '新消息已收到；上一条输入或呈现收据核对完成后继续处理。'
            return
        text.start_queued(pending[0])

    def queue_state():
        return {'queued_inputs': [{'turn_id': value['turn_id'],
                                  'attempt_id': value['attempt_id']}
                                 for value in store.queued_inputs()],
                'input_queue_paused': store.input_queue_paused(),
                'input_queue_waiting': queue_waiting}

    def receive_input(value, receiving_context=None):
        frozen = runtime.config.attempt_binding(text.binding)
        if receiving_context is not None:
            frozen['receiving_context'] = receiving_context
        frozen['request_control_sha256'] = hashlib.sha256(runtime.control.encode()).hexdigest()
        return store.enqueue_input(value, frozen)

    def dispatch(action, payload):
        if action == "state":
            provider.configured = api_configured()
            if (
                provider.selected == "local"
                and not fake
                and state["state"] == "ready"
                and base_model.process.poll() is not None
            ):
                state.update(state="failed", error="模型进程已退出，可以重新启动。")
            pump_inputs()
            snapshot = text.snapshot()
            memory_state = (
                memory.state()
                if memory is not None
                else {
                    "memory_management": False,
                    "memory_status": "unavailable",
                    "memory_error": memory_error,
                }
            )
            return {
                **state,
                **snapshot,
                "startup": startup_report(),
                **queue_state(),
                **memory_state,
                **(runtime.snapshot() if runtime is not None else {}),
                "provider": provider.state(),
                "error": (configuration_error or state["error"] or snapshot["error"]
                          or memory_error),
                "backend_pid": os.getpid(),
                "server_pid": (local_model.process.pid
                               if not fake and local_model is not None and local_model.process
                               else None),
            }
        if action == "start":
            with lifecycle_lock:
                if (
                    provider.selected == "local"
                    and not fake
                    and state["state"] in {"stopped", "failed"}
                ):
                    state.update(state="loading_model", error="")
                    threading.Thread(target=load_model, daemon=True).start()
            snapshot = text.snapshot()
            return {**state, **snapshot, "startup": startup_report(),
                    "error": (configuration_error or state["error"] or snapshot["error"])}
        if action == "history":
            if runtime is not None and runtime.claim is not None:
                runtime.reconcile(input_id=runtime.claim.input_id)
            return text.history(**payload)
        if action in {"streamPoll", "streamState", "streamAck", "streamReceive", "streamStop"}:
            if runtime is None:
                raise ValueError("Delivery authority unavailable")
            if action == "streamPoll":
                return runtime.poll()
            if action == "streamState":
                return runtime.state()
            if action in {"streamAck", "streamReceive"}:
                receipt = runtime.ack(payload, received=action == 'streamReceive')
                pump_inputs()
                return receipt
            store.pause_input_queue(bool(store.queued_inputs()))
            runtime.stop()
            return text.stop()
        if action == "sessions":
            offset = payload.get("offset", 0)
            page = store.sessions(offset=offset, limit=101)
            return {
                "sessions": page[:100],
                "next_offset": offset + 100 if len(page) > 100 else None,
            }
        if action == "selectSession":
            return text.select_session(payload["session_id"])
        if action == "feedback":
            return feedback.dispatch(payload)
        if action == "selectProvider":
            provider.configured = api_configured()
            return provider.select(
                payload["provider_id"], confirmed=payload.get("confirmed", False)
            )
        if action == "memories" and memory is not None:
            return memory.list_memories(**payload)
        if action == "memoryDetails" and memory is not None:
            return memory.memory_details(payload["record_id"])
        if action == "memoryAction" and memory is not None:
            return memory.apply_action(payload)
        if action == "memoryAbandon" and memory is not None:
            return memory.abandon(payload)
        if action in {"memories", "memoryDetails", "memoryAction", "memoryAbandon"}:
            raise ValueError("Memory management pilot is not enabled")
        if action == "newSession":
            return text.new_session()
        if action == "stop":
            store.pause_input_queue(bool(store.queued_inputs()))
            if runtime is not None:
                runtime.stop()
            return text.stop()
        if action == 'resumeInputs':
            store.pause_input_queue(False)
            pump_inputs()
            return queue_state()
        if configuration_error:
            raise RuntimeError(configuration_error)
        if state["state"] != "ready":
            raise RuntimeError("Model not ready")
        if action == "send":
            if runtime is None:
                raise RuntimeError("Delivery authority unavailable")
            if text.snapshot()['busy'] or store.queued_inputs() or store.input_queue_paused():
                received = receive_input(payload['text'], payload.get('receiving_context'))
                pump_inputs()
                return received
            try:
                runtime.before_start(False)
            except ValueError:
                return receive_input(payload['text'], payload.get('receiving_context'))
            return text.send(payload["text"], receiving_context=payload.get('receiving_context'))
        if action == "retry":
            if runtime is None:
                raise RuntimeError("Delivery authority unavailable")
            if store.queued_inputs() and not store.input_queue_paused():
                raise ValueError('请先处理待接续输入，或停止后重试当前输入。')
            return text.retry()
        raise ValueError("Unknown action")

    def serialized_dispatch(action, payload):
        # Capture one authority under the admission lock; readiness waiting must
        # leave state, ACK, stop and selection free to use that same boundary.
        if action == "streamPoll" and payload.get("wait_ms", 0):
            with lifecycle_lock:
                current_runtime = runtime
                if current_runtime is None:
                    raise ValueError("Delivery authority unavailable")
                bound_poll = current_runtime.bind_poll()
            result = bound_poll(payload['wait_ms'])
            with lifecycle_lock:
                if runtime is not current_runtime or result != bound_poll():
                    return {"event": None}
                return result
        # Legacy immediate polls and turn/session starts keep their boundary.
        with lifecycle_lock:
            return dispatch(action, payload)

    _STARTUP_CONFIGURATION_PHASE = False
    server = create_server(token, serialized_dispatch)
    print(json.dumps({"event": "ready", "pid": os.getpid(), "job": bool(job)}), flush=True)
    _READY_HANDSHAKE_SENT = True
    server.serve_forever(poll_interval=0.1)


def _run_module():
    global _READY_HANDSHAKE_SENT, _STARTUP_CONFIGURATION_PHASE
    _READY_HANDSHAKE_SENT = False
    _STARTUP_CONFIGURATION_PHASE = True
    try:
        main()
    except _STARTUP_CONFIGURATION_ERRORS:
        if not _STARTUP_CONFIGURATION_PHASE or _READY_HANDSHAKE_SENT:
            raise
        print(json.dumps({"event": "startup_failed", "category": "configuration"}),
              flush=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    _run_module()
