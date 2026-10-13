# Repository guidance

- Read `README.md` and `docs/development.md` before changing code. Product responsibilities are in `docs/architecture.md`.
- Keep changes scoped and preserve unrelated working-tree edits. Do not commit local data, secrets, model weights, generated environments or build artifacts.
- Keep shared conversation state under the backend owner. Adapt external providers and platform APIs at the boundary.
- Preserve character differences, memory provenance/revisions, honest unknowns and the distinction between generated, delivered and visible messages.
- Run relevant tests while iterating and `pwsh -File scripts/check.ps1` for an integrated change. Tests must use synthetic fixtures by default; do not activate paid APIs or inspect personal data to make tests pass.
- New dependencies and copied/adapted code need fixed sources and third-party notices. The root MIT license does not replace upstream licenses.
- Report the final behavior, validation and remaining limits accurately. Do not infer dialogue quality or a stable release from unit-test counts.
