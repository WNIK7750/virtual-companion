"""Generic strict structured-response errors, independent of historical schemas."""


class ProtocolInvalid(ValueError):
    """A requested model result is invalid; its semantic judgment remains unknown."""


class FactSupportUnresolved(ValueError):
    """A valid semantic assessment does not authorize the proposed fact operation."""
