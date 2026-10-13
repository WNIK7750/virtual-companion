"""A model Port shared by the Harness and staged native cognition.

Capacity and sending use the identical final body. Counter implementations must
be verified for the selected endpoint/model; a missing counter fails explicitly.
"""

import math
from copy import deepcopy

from virtual_companion.boundary.contracts import canonical, identifier

from .contracts import ProviderRoute
from .openai_compatible import build_request


class BoundProviderCall:
    def __init__(
        self,
        route,
        *,
        transport,
        counter,
        stage,
        cancelled,
        clock,
        deadline,
        output_tokens,
        data_scope,
        schema=None,
        on_content=None,
    ):
        if not isinstance(route, ProviderRoute) or not callable(counter) or not callable(clock):
            raise ValueError("Verified route, exact counter and monotonic clock required")
        if type(deadline) not in {int, float} or not math.isfinite(deadline):
            raise ValueError("Finite absolute model deadline required")
        if (
            type(output_tokens) is not int
            or not 0 < output_tokens < route.capabilities.context_tokens
        ):
            raise ValueError("Positive stage output budget required")
        if (
            not isinstance(data_scope, tuple)
            or not data_scope
            or set(data_scope) - set(route.data_scope)
        ):
            raise ValueError("Model context sending scope not authorized")
        identifier(stage)
        if on_content is not None and (not callable(on_content) or schema is not None):
            raise ValueError("Unapproved content observer requires plain text")
        self.on_content = on_content
        self.route, self.transport, self.counter, self.stage = route, transport, counter, stage
        self.cancelled, self.clock, self.deadline = cancelled, clock, deadline
        self.output_tokens, self.data_scope, self.schema = (
            output_tokens,
            data_scope,
            deepcopy(schema),
        )
        self._prepared, self._last, self._failed = None, None, False
        self.last_usage = None

    def _checkpoint(self):
        if self.cancelled.is_set():
            raise InterruptedError("Model call cancelled")
        now = self.clock()
        if (
            self._failed
            or type(now) not in {int, float}
            or not math.isfinite(now)
            or now >= self.deadline
            or (self._last is not None and now < self._last)
        ):
            self._failed = True
            raise TimeoutError("Model absolute deadline or clock contract failed")
        self._last = now
        return now

    def cancel(self):
        self.transport.cancel()

    @property
    def context_tokens(self):
        return self.route.capabilities.context_tokens

    def count(self, request):
        self._prepared = None
        self._checkpoint()
        if (
            not isinstance(request, dict)
            or set(request) != {"messages", "tools", "max_output_tokens"}
            or request["max_output_tokens"] != self.output_tokens
        ):
            raise ValueError("Exact Harness model request required")
        identity = canonical(request)
        if self.on_content is not None and request["tools"]:
            raise ValueError("Unapproved text observer cannot consume tool requests")
        body = build_request(
            self.route,
            request["messages"],
            output_tokens=self.output_tokens,
            tools=request["tools"],
            schema=self.schema,
            stage=self.stage,
        )
        count = self.counter(deepcopy(body))
        self._checkpoint()
        if type(count) is not int or count < 0:
            raise ValueError("Exact final-wire count unavailable")
        if count + self.output_tokens + 1 > self.route.capabilities.context_tokens:
            raise ValueError("Model final request capacity exceeded")
        self._prepared = (identity, body, count)
        return count

    def invoke(self, request):
        now = self._checkpoint()
        if self._prepared is None or canonical(request) != self._prepared[0]:
            raise ValueError("Model request count/send binding changed")
        _, body, count = self._prepared
        self._prepared = None
        result = self.transport.invoke(
            deepcopy(body),
            self.cancelled,
            stage=self.stage,
            data_scope=self.data_scope,
            input_tokens=count,
            timeout_seconds=min(300, self.deadline - now),
            **({"on_content": self.on_content} if self.on_content is not None else {}),
        )
        self._checkpoint()
        if not isinstance(result, dict) or set(result) != {
            "text",
            "tool_calls",
            "finish_reason",
            "usage",
        }:
            raise ValueError("Complete provider terminal contract required")
        self.last_usage = deepcopy(result["usage"])
        return {key: deepcopy(result[key]) for key in ("text", "tool_calls", "finish_reason")}
