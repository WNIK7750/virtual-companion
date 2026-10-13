"""An input-time observation from one receiving client, not an action receipt."""
import json

_IDENTITY = {'protocol': 'receiving-context.v1', 'source': 'receiving-client',
             'scope': 'this_client', 'observation': 'at_input_submission'}


def validate_receiving_context(value):
    if (not isinstance(value, dict) or set(value) != {*_IDENTITY, 'quiet'}
            or any(value[key] != expected for key, expected in _IDENTITY.items())
            or type(value['quiet']) is not bool):
        raise ValueError('Exact receiving observation required')
    return dict(value)


def receiving_from_binding(binding):
    data = json.loads(binding) if isinstance(binding, str) else binding
    if 'receiving_context' not in data:
        return None
    return validate_receiving_context(data['receiving_context'])
