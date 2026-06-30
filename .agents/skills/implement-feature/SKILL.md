---
description: Implement a new feature or extend existing functionality in this active Python project.
argument-hint: "[feature request or target module]"
---

# Implement feature

Add or extend functionality with minimal, well-tested changes.

## Principles
- Inspect existing patterns before coding (esp. the FEM "one definition, two backends" structure and
  `envs/geometry.py` as geometry single source of truth).
- Summarize the implementation plan before large edits (**always before editing more than 3 files**).
- Modify source code when needed; add/update tests when behavior changes; update docs/examples when
  user-facing behavior changes.
- Avoid broad rewrites unless requested. Avoid new dependencies unless necessary (never edit
  `pyproject.toml`/`requirements*` unless explicitly asked).
- Preserve backward compatibility unless the user explicitly accepts breaking changes.
- Do not silently change engineering meaning (units, sign conventions, boundary conditions, tolerances, solver conventions).

## Process
1. Understand the requested behavior.
2. Inspect relevant files and existing patterns.
3. Identify implementation points and likely affected modules.
4. Make a small plan (and share it if the change is non-trivial).
5. Implement.
6. Add or update tests (numerical: explicit tolerances; OpenSees: `importorskip` guard).
7. Run focused tests, then a broader run if the change is significant.
8. Summarize changes and risks.

## Output
- Feature implemented
- Files changed
- Behavior added or changed
- Tests added or updated
- Tests run
- Remaining TODOs or risks
