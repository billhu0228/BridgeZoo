---
description: Run a pre-release checklist for a Python milestone before tagging, packaging, delivery, or handoff.
disable-model-invocation: true
---

# Release check (read-only)

Pre-release / pre-handoff checklist. **Read-only by default** — do not edit code unless explicitly asked after the check.

## Checks
1. **Working tree clean** — `git status` (uncommitted or untracked files that should be committed/ignored?).
2. **Version consistency** — `version` in `pyproject.toml` (and `bridgezoo/__init__.py` `__version__`) agree;
   matches the intended tag.
3. **README usage** — install + run commands in `README.md` still accurate.
4. **Package metadata** — `pyproject.toml`: name, version, `requires-python`, dependencies, package discovery
   (`include`/`exclude` keeps `archive/` and tests out of the build).
5. **Test status** — `python -m pytest -q` (prefer an OpenSeesPy-capable interpreter so cross-checks run); note skips.
6. **TODO/FIXME** — scan touched / release-relevant files for unfinished items.
7. **Dependency changes** — any new/changed deps since last release; `requirements*.txt` in sync with `pyproject`.
8. **Release notes** — determine whether notes/changelog are needed and what they should say.
9. **CI sanity** — `.github/workflows/publish.yml` triggers on `v*.*.*` and uses `python -m build` + twine.
   Verify the CI Python version is compatible with `requires-python`.

## Output
- **Release status:** Pass / Conditional pass / Fail
- Must fix before release
- Should fix soon
- Suggested release note
- Suggested tag name if obvious (e.g. `vX.Y.Z` matching `pyproject` version)
- Commands run
