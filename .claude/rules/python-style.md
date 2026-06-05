---
paths:
  - "**/*.py"
---

# Python style rules

Apply to Python source under `bridgezoo/`, `scripts/`, `tools/`, `tests/`.
(`archive/` is legacy — do not edit or import from it.)

## General
- Add type hints to new or modified public functions and methods.
- Keep functions small, single-purpose, and testable.
- Prefer explicit errors (raise with a clear message) over silent fallbacks or returning sentinel values.
- Do not swallow exceptions with broad `except:` / `except Exception:` unless re-raising or logging with intent.
- Avoid introducing new dependencies; never add one to satisfy a minor convenience.
- Preserve the existing architectural style unless the user explicitly requests a redesign.

## Separation of concerns
- Keep these layers distinct (this project already does): geometry/parameter definition, structural assembly/calculation,
  solver backends, environment/RL, IO, CLI scripts, reporting, and visualization.
- Keep geometry in `envs/geometry.py` (single source of truth). Solver backends consume the neutral model IR;
  do not bake geometry assumptions into solvers or scripts.

## Numerical code
- Use **explicit tolerances**; never compare floats with `==` for physical quantities.
- Document units and assumptions (SI: N, m, Pa) in docstrings; do not change units/sign conventions implicitly.
- Keep the linear vs. geometric (OpenSees corot) distinction explicit and commented where it matters.
- Do not optimize prematurely; prefer clear, correct code first.

## State & imports
- Do not introduce global mutable state unless it is already part of the project's design.
- Heavy/optional imports (e.g. `openseespy`, `torch`, `matplotlib`) should be imported lazily inside functions
  when the module must remain importable without them — follow the existing lazy-import pattern.
