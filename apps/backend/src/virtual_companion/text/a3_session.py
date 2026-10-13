"""Optional local A3 session; no process starts until explicitly entered."""

import json
import threading
from copy import deepcopy
from hashlib import sha256

from virtual_companion.cognition.a3_transport import SchemaTransport

from .exact_llama import ExactLlamaModel


class LocalSchemaSession:
    """Explicit owner for one pinned local process and stage-specific reserves.

    Construction does not launch anything. Enter only after the experiment's model
    execution gate; measurements can use ExactLlamaModel independently without this.
    """

    def __init__(
        self,
        profile,
        repository,
        systems,
        reservations,
        *,
        max_calls,
        max_output_tokens,
        observe,
        model_factory=None,
    ):
        self.models = {}
        for stage, reserve in reservations.items():
            configured = deepcopy(profile)
            configured["sampling"]["max_tokens"] = reserve
            self.models[stage] = (model_factory or ExactLlamaModel)(
                configured,
                repository,
                system_sha256=frozenset(systems[stage]),
            )
        self.owner = next(iter(self.models.values()))
        self.invoker = SchemaTransport(
            self.models,
            reservations,
            max_calls=max_calls,
            max_output_tokens=max_output_tokens,
            observe=observe,
        )
        self.wall_seconds = None
        self._timer = None
        self._entered = False

    @classmethod
    def from_freeze(cls, freeze, repository, *, seed, observe, model_factory=None):
        config = json.loads(freeze.read_text(encoding="utf-8"))
        if config["schema_version"] != 1 or seed not in config["seeds"]:
            raise ValueError("Unfrozen A3 seed or protocol version")
        for name, expected in config["input_pins"].items():
            source = (repository / name).resolve()
            if not source.is_relative_to(repository.resolve()):
                raise ValueError("Frozen input must be repository-owned")
            if sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError("A3 input changed after freeze")
        path = (repository / config["runtime_profile"]).resolve()
        if not path.is_relative_to(repository.resolve()):
            raise ValueError("Frozen profile must be repository-owned")
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != config["runtime_profile_sha256"]:
            raise ValueError("A3 profile changed after freeze")
        profile = json.loads(raw)
        profile["sampling"]["seed"] = seed
        budget = config["per_seed_session_budget"]
        if type(budget["max_wall_seconds"]) is not int or budget["max_wall_seconds"] <= 0:
            raise ValueError("Positive frozen wall budget required")
        session = cls(
            profile,
            repository,
            config["system_sha256"],
            config["output_reservations"],
            max_calls=budget["max_calls"],
            max_output_tokens=budget["max_output_tokens"],
            observe=observe,
            model_factory=model_factory,
        )
        session.wall_seconds = budget["max_wall_seconds"]
        for stage, model in session.models.items():
            if hasattr(model, "bind_schema"):
                model.schema_sha256 = frozenset(config["schema_sha256"][stage])
        return session

    def _expire(self):
        self.invoker.deadline_expired = True
        self.invoker.cancel()

    def __enter__(self):
        if self._entered:
            raise RuntimeError("A3 session is single-use")
        self._entered = True
        try:
            self.owner.start()
            for model in self.models.values():
                model.process = self.owner.process
                model.token = self.owner.token
            if self.wall_seconds is not None:
                self._timer = threading.Timer(self.wall_seconds, self._expire)
                self._timer.daemon = True
                self._timer.start()
            return self.invoker
        except BaseException:
            self.owner.close()
            raise

    def __exit__(self, *exc):
        if self._timer is not None:
            self._timer.cancel()
            self._timer.join()
        try:
            self.invoker.cancel()
        finally:
            self.owner.close()
