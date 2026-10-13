import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = REPOSITORY_ROOT / "apps/backend/src"
for path in (REPOSITORY_ROOT, BACKEND_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
