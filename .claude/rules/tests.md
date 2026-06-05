---
paths:
  - "tests/**/*.py"
  - "**/test_*.py"
  - "**/*_test.py"
---

# Test rules

Framework: **pytest** (`testpaths=["tests"]`, `addopts="-ra"` in `pyproject.toml`).

## Core
- Add or update tests whenever behavior changes.
- Add a **regression test** for every fixed bug (capture the failing case so it cannot silently return).
- Prefer deterministic tests; seed any randomness explicitly.
- Use fixtures for repeated setup (see `tests/conftest.py`, e.g. the geometry fixture).
- Never use machine-specific absolute paths; build paths from the repo root or `tmp_path`.

## Numerical assertions
- Use explicit tolerances (`pytest.approx`, `np.isclose`, or `abs(a-b) < tol`) and briefly explain the tolerance choice.
- For solver cross-checks, normalize relative error by the quantity's overall magnitude (avoid blow-up on near-zero components).
- OpenSeesPy-dependent tests must guard with `pytest.importorskip("openseespy")` so the suite stays green where OpenSees is unavailable.

## Integrity
- **Do not weaken a test just to make it pass.**
- If a test is outdated because the *intended* behavior changed, explain why the old expectation is no longer correct
  before updating it — do not edit assertions blindly.
- Keep tests fast; the default suite should run without heavy training or long simulations.
