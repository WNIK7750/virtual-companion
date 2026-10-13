"""Load explicitly selected text strategies before preparing model requests."""

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from virtual_companion.boundary.contracts import canonical

VERSION = 'memory-skills.v1'
MAINTENANCE_VERSION = 'memory-skills.v2'
SUPPORTED_VERSIONS = (VERSION, MAINTENANCE_VERSION)
NAMES = ('vc-memory-retrieval', 'vc-context-composition')
MAINTENANCE_NAME = 'vc-memory-maintenance'
_MAINTENANCE_STAGES = frozenset({
    'memory_command', 'memory_command_support', 'fact_target_select',
    'fact_extract', 'fact_support', 'fact_repair', 'fact_assess',
    'source_propose', 'source_repair', 'exchange_intent', 'exchange_extract',
    'exchange_assess', 'exchange_support',
})


@dataclass(frozen=True)
class MemorySkills:
    instructions: tuple[tuple[str, str, str], ...]
    version: str = VERSION

    @property
    def names(self):
        return tuple(name for name, _, _ in self.instructions)

    @property
    def identity(self):
        return [{'name': name, 'sha256': digest} for name, _, digest in self.instructions]

    @property
    def fingerprint(self):
        return sha256(canonical([self.version, self.identity]).encode('utf-8')).hexdigest()

    def messages(self, *, navigation):
        return [{'role': 'system', 'content': text} for name, text, _ in self.instructions
                if name != MAINTENANCE_NAME and (navigation or name != 'vc-memory-retrieval')]

    def maintenance_messages(self, *, stage, source_role):
        """User-source maintenance only; this policy grants no additional tools."""
        if (self.version != MAINTENANCE_VERSION or source_role != 'user'
                or stage not in _MAINTENANCE_STAGES):
            return []
        return [{'role': 'system', 'content': text} for name, text, _ in self.instructions
                if name == MAINTENANCE_NAME]


def load_memory_skills(version=VERSION):
    """An allowlist of packaged text only; no scripts, discovery or extra permissions."""
    root = Path(__file__).parent / 'skills'
    if version not in SUPPORTED_VERSIONS:
        raise ValueError('Unsupported runtime memory skills')
    loaded = []
    names = (*NAMES, MAINTENANCE_NAME) if version == MAINTENANCE_VERSION else NAMES
    for name in names:
        raw = (root / name / 'SKILL.md').read_bytes()
        if len(raw) > 16000:
            raise ValueError('Bundled memory skill exceeds instruction capacity')
        text = raw.decode('utf-8')
        sections = text.split('---', 2)
        if (len(sections) != 3 or sections[0].strip()
                or re.findall(r'^name:\s*([^\r\n]+)', sections[1], re.MULTILINE) != [name]
                or not sections[2].strip()):
            raise ValueError('Bundled memory skill requires its named SKILL entry')
        loaded.append((name, sections[2].strip(), sha256(raw).hexdigest()))
    return MemorySkills(tuple(loaded), version)
