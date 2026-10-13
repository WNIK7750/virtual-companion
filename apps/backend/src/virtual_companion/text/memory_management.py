"""Application surface for the opt-in desktop memory-management pilot."""

from virtual_companion.boundary.contracts import Scope, companion_id, identifier
from virtual_companion.boundary.memory_management import (
    ExplicitMemoryCommand,
    list_source_digest,
)
from virtual_companion.boundary.memory_projection import (
    clean_memory_text,
    project_memory,
    source_label,
)


class MemoryManagementService:
    def __init__(self, store, companion, *, clock):
        companion_id(companion)
        if not callable(clock):
            raise ValueError("Memory management clock required")
        self.store, self.companion, self.clock = store, companion, clock

    def _attention(self):
        results = []
        for row in self.store.background_sources(self.companion, limit=100):
            if row["role"] != "user" or row["status"] != "failed":
                continue
            if row["error"] not in self.store.clarification_errors:
                continue
            source = self.store.background_source_text(self.companion, row["job"])
            mixed = row["error"] == "needs_clarification.mixed_source"
            results.append(
                {
                    "job": row["job"],
                    "status": "needs_clarification",
                    "message": (
                        "这条消息还包含其他请求，不能只选择一条记忆就完成整条消息；"
                        "可以放弃其中的记忆处理，让其他请求继续。"
                        if mixed
                        else "有一条记忆处理需要你确认；也可以明确放弃本次处理。"
                    ),
                    "source": clean_memory_text(source),
                    "reason": row["error"],
                    "resolution_mode": "whole_source" if mixed else "record_selection",
                }
            )
        return results

    def list_memories(self, *, offset=0, limit=50):
        now_us = self.clock()
        page = self.store.memory_records(
            self.companion, offset=offset, limit=limit, now_us=now_us
        )
        consumed = offset + len(page["rows"])
        return {
            "items": [project_memory(value) for value in page["rows"]],
            "total": page["total"],
            "next_offset": consumed if consumed < page["total"] else None,
            "stats": self.store.memory_statistics(self.companion, now_us=now_us),
            "attention": self._attention(),
        }

    def memory_details(self, record_id):
        identifier(record_id)
        value = self.store.memory_record(self.companion, record_id, now_us=self.clock())
        if value is None:
            raise ValueError("Memory record no longer exists")
        source = value["source"]
        return {
            "item": project_memory(value["current"]),
            "history": [project_memory(version) for version in value["history"]],
            "source": {
                "label": source_label(source["kind"]),
                "role": source["role"],
                "text": source["text"],
            },
        }

    @staticmethod
    def _action_payload(payload):
        if not isinstance(payload, dict) or set(payload) not in (
            {"operation_id", "action", "record_id", "expected_version", "replacement"},
            {
                "operation_id",
                "action",
                "record_id",
                "expected_version",
                "replacement",
                "resolves_job",
            },
        ):
            raise ValueError("Exact memory action payload required")
        for key in ("operation_id", "record_id"):
            identifier(payload[key])
        if payload.get("resolves_job") is not None:
            identifier(payload["resolves_job"])
        if payload["action"] not in {"correct", "forget"}:
            raise ValueError("Known memory list action required")
        if type(payload["expected_version"]) is not int or payload["expected_version"] < 1:
            raise ValueError("Positive expected memory version required")
        if not isinstance(payload["replacement"], str) or len(
            payload["replacement"].encode()
        ) > 16000:
            raise ValueError("Bounded memory correction required")
        if payload["action"] == "correct" and not payload["replacement"].strip():
            raise ValueError("Correction text required")
        if payload["action"] == "forget" and payload["replacement"]:
            raise ValueError("Forget action cannot replace content")
        return payload

    def apply_action(self, payload):
        value = self._action_payload(payload)
        digest = list_source_digest(
            action=value["action"],
            record_id=value["record_id"],
            expected_version=value["expected_version"],
            replacement=value["replacement"],
        )
        command = ExplicitMemoryCommand(
            Scope(self.companion, "memory-management"),
            value["operation_id"],
            value["action"],
            value["record_id"],
            value["expected_version"],
            "list",
            value["operation_id"],
            digest,
            "single_record",
            "explicit_selection",
            "host-list.v1",
            digest,
            replacement=value["replacement"],
            resolves_job=value.get("resolves_job"),
        )
        receipt = self.store.execute_memory_command(command, now_us=self.clock())
        status = receipt["status"]
        if status == "committed" and value["action"] == "forget":
            message = (
                "已停止在后续回答中使用这条记忆。聊天记录和必要的修订历史仍会保留。"
            )
        elif status == "committed":
            message = "更正已提交，后续回答将使用新版本。"
        elif status == "already_applied":
            message = "当前记忆已经是你要求的状态，无需重复处理。"
        elif status == "version_changed":
            message = "这条记忆已经变化，已刷新为当前版本；没有覆盖新内容。"
        else:
            message = "记忆操作未保存，请重新打开后重试。"
        current = self.store.memory_record(
            self.companion, value["record_id"], now_us=self.clock()
        )
        return {
            **receipt,
            "message": message,
            "review_calls": 0,
            "current": project_memory(current["current"]) if current else None,
        }

    def abandon(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"job", "operation_id"}:
            raise ValueError("Exact memory abandonment payload required")
        identifier(payload["job"])
        identifier(payload["operation_id"])
        receipt = self.store.abandon_memory_source(
            self.companion,
            payload["job"],
            payload["operation_id"],
            now_us=self.clock(),
        )
        if receipt["status"] == "continued":
            message = "已放弃其中的记忆处理；原消息里的其他请求仍会继续处理。"
        elif receipt.get("suppressed_records"):
            message = (
                "已放弃本次未决处理；相关旧记忆继续暂停用于回答，"
                "你仍可在列表中更正或忘记。"
            )
        else:
            message = "已放弃本次未决记忆处理；现有有效记忆没有因此恢复或改写。"
        return {
            **receipt,
            "message": message,
            "review_calls": 0,
        }
