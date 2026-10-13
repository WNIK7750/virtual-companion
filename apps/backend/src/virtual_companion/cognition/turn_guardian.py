"""One serialized lease/cancellation observer for the entire cognitive turn."""

import threading


class TurnGuardian:
    def __init__(self, control, model, *, cancelled=None):
        self.control, self.model = control, model
        self.cancelled = threading.Event() if cancelled is None else cancelled
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.failure = None
        self.observer = threading.Thread(target=self._observe, name="cognition-turn-control")

    def checkpoint(self):
        with self.lock:
            if self.failure is not None:
                raise self.failure
            try:
                return self.control.checkpoint()
            except Exception as error:
                self.failure = error
                self.cancelled.set()
        # Do not hold the lease-state lock while interrupting transport I/O.
        try:
            self.model.cancel()
        except Exception as error:
            self.control.diagnostics["cancel_error"] = type(error).__name__
        raise self.failure

    def raise_if_failed(self):
        if self.failure is not None:
            raise self.failure

    def _observe(self):
        while not self.stopped.wait(0.025):
            try:
                self.checkpoint()
            except Exception:
                return

    def __enter__(self):
        self.checkpoint()
        self.observer.start()
        return self

    def stop(self):
        """Quiesce observation before the owner writes its own terminal state."""
        self.stopped.set()
        if self.observer.is_alive():
            self.observer.join()

    def __exit__(self, *_):
        self.stop()
        self.raise_if_failed()
