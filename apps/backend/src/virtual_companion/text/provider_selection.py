"""Visible provider selection, distinct from a successfully bound route."""

from __future__ import annotations

import json
from pathlib import Path


class ProviderSelection:
    def __init__(self, catalog_path: Path, profile: dict, *, switch=None, configured=False):
        self.catalog_path = Path(catalog_path)
        self.catalog = self._load(self.catalog_path)
        self.profile = profile
        self.selected = self.catalog["default_provider"]
        self.last_error = ""
        self.switch = switch
        self.configured = configured
        self.recovery_required = False
        self.active_identity = {}
        self.bound_provider = "local" if self.selected == "local" else None

    @staticmethod
    def _load(path: Path):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("Provider catalog unavailable") from error
        if (
            not isinstance(value, dict)
            or set(value) != {"schema_version", "default_provider", "local", "api"}
            or value["schema_version"] != 1
        ):
            raise ValueError("Unsupported provider catalog")
        for key in ("local", "api"):
            route = value[key]
            if not isinstance(route, dict) or route.get("kind") not in {"local", "api"}:
                raise ValueError("Invalid provider catalog route")
            if not isinstance(route.get("provider_id"), str) or not route["provider_id"]:
                raise ValueError("Invalid provider catalog identity")
            if not isinstance(route.get("capabilities"), list):
                raise ValueError("Invalid provider catalog capabilities")
        if value["local"]["provider_id"] != "local":
            raise ValueError("Local provider identity must be stable")
        if value["default_provider"] not in {"local", value["api"]["provider_id"]}:
            raise ValueError("Unsupported default provider")
        return value

    def _local_model(self):
        return self.profile.get("profile_id", "fixed-local-profile")

    def state(self):
        local, api = self.catalog["local"], self.catalog["api"]
        active = local if self.selected == "local" else api
        bound = self.bound_provider == self.selected and not self.recovery_required
        api_bound = bound and self.selected == api["provider_id"]
        return {
            "selected_provider": self.selected,
            "recovery_required": self.recovery_required,
            "default_provider": self.catalog["default_provider"],
            "active": {
                "provider_id": active["provider_id"],
                "label": active["label"],
                "model_label": active["model_label"],
                "kind": active["kind"],
                "model_profile": self._local_model()
                if self.selected == "local"
                else None,
                "status": "active" if bound else "unavailable" if self.configured
                or self.selected == "local" else "unconfigured",
                "capabilities": list(active["capabilities"]),
                **self.active_identity,
            },
            "options": [
                {
                    "provider_id": local["provider_id"],
                    "label": local["label"],
                    "model_label": local["model_label"],
                    "status": "available",
                    "selectable": True,
                    "message": "本地运行；重开程序按默认模型重新绑定："
                    + (local if self.catalog["default_provider"] == "local" else api)["model_label"]
                    + "。",
                },
                {
                    "provider_id": api["provider_id"],
                    "label": api["label"],
                    "model_label": api["model_label"],
                    "status": (
                        "active"
                        if api_bound
                        else "requires_confirmation"
                        if self.configured and self.selected != api["provider_id"]
                        else "unavailable" if self.configured
                        else "unconfigured"
                    ),
                    "selectable": self.configured,
                    "network_enabled": api_bound,
                    "credential_store": api["credential_store"],
                    "data_scope": api["data_scope"],
                    "requires_confirmation": self.selected != api["provider_id"]
                    or not api_bound,
                    "message": "选择后，本伴侣的当前输入、历史、有效记忆及后台整理内容"
                    "将发给 DeepSeek。"
                    "切回本地停止后续远程处理；已发送内容无法撤回。"
                    + (
                        ""
                        if self.configured
                        else " 尚缺受管配置或可选 provider 依赖；没有发送请求。"
                    ),
                },
            ],
            "error": self.last_error,
        }

    def select(self, provider_id: str, *, confirmed=False):
        if provider_id not in {"local", self.catalog["api"]["provider_id"]}:
            raise ValueError("Unknown provider selection")
        if (provider_id == self.selected and self.bound_provider == provider_id
                and not self.recovery_required):
            self.last_error = ""
            return {"status": "selected", "message": "当前已使用此路由。", "provider": self.state()}
        if provider_id != "local" and (not self.configured or not confirmed):
            self.last_error = "尚未配置 API 或尚未确认远程处理范围。没有发送请求，当前路由不变。"
            return {"status": "unavailable", "message": self.last_error, "provider": self.state()}
        try:
            if self.switch is None:
                if provider_id != "local":
                    raise ValueError("Provider binding unavailable")
            else:
                identity = self.switch(provider_id)
                self.active_identity = identity if isinstance(identity, dict) else {}
        except Exception:
            self.last_error = (
                ("原路由已暂停：旧后台未能完成关闭。请再次选择路由恢复；恢复完成前不会发送新请求。")
                if self.recovery_required
                else (
                    "路由未切换：请检查进行中的回复、呈现收据、受管凭据、"
                    "计数器和预算配置。没有自动换路。"
                )
            )
            return {"status": "unavailable", "message": self.last_error, "provider": self.state()}
        self.selected = provider_id
        self.bound_provider = provider_id
        self.recovery_required = False
        self.last_error = ""
        return {
            "status": "selected",
            "message": "已切换；后续输入和后台整理使用所选路由。",
            "provider": self.state(),
        }
