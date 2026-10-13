"""Use the owned llama.cpp runtime's actual template and tokenizer, without inference.

The runtime must already have verified and loaded its frozen artifacts. This is
not a remote-provider estimator and never adds the desktop baseline system text.
"""

from copy import deepcopy

from virtual_companion.boundary.contracts import canonical


class LocalWireCounter:
    def __init__(self, runtime):
        self.runtime = runtime

    def __call__(self, body):
        if self.runtime.process is None or self.runtime.process.poll() is not None:
            raise RuntimeError("Verified local model is not running")
        if not isinstance(body, dict) or len(canonical(body).encode()) > 2_000_000:
            raise ValueError("Bounded exact local request required")
        template = self.runtime._context_request("/apply-template", deepcopy(body)).get("prompt")
        if not isinstance(template, str) or not template or len(template.encode()) > 2_000_000:
            raise ValueError("Local template unavailable")
        tokens = self.runtime._context_request(
            "/tokenize", {"content": template, "add_special": True, "parse_special": True}
        ).get("tokens")
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise ValueError("Local tokenizer result unavailable")
        return len(tokens)
