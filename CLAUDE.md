# CLAUDE.md — BridgeZoo

Project-level instructions for active development. Keep edits minimal and behavior-safe.

## Project status
- **Active development, milestone stage.** Not frozen, not release-locked.
- Feature work, bug fixing, refactoring, and test improvement are all allowed.
- Stability, regression safety, and maintainability are priorities.

## What this is
- Research framework: **MAPPO reinforcement learning for staged cable-tensioning of a 2D cable-stayed bridge.**
- Two pillars:
  1. **Structural FEM** — "one model definition, two backends, consistent results": a self-written
     linear direct-stiffness solver (fast, the RL kernel) cross-checked against OpenSeesPy.
  2. **Multi-agent RL (MAPPO)** — currently skeleton (see `TODO.md`).
- It is an **engineering / numerical** project plus RL. `scripts/` are CLI entry points run as `python -m scripts.X`.

## Structure
- `bridgezoo/` — main package
  - `fem/` — `model.py` (solver-neutral IR), `builder.py`, `linear_frame.py` (self-written direct stiffness),
    `staged.py` (staged construction + builders), `opensees_backend.py` / `opensees_staged.py` / `opensees_ref.py` (OpenSees).
  - `envs/` — `geometry.py` (**geometry single source of truth**), `cable_agent.py`, `cable_construction.py` (PettingZoo env, skeleton).
  - `mappo/` — `config.py` (real) + `actor_critic.py` / `buffer.py` / `trainer.py` (skeleton).
  - `render/` — `pygame_render.py`, `mpl_cjk.py` (matplotlib Chinese-font setup).
- `scripts/` — entry points: `validate_fem`, `validate_staged`, `oneshot_opensees`, `staged_demo`,
  `plot_staged_deck_growth`, `train`, `evaluate`, `baselines`.
- `tools/` — developer utilities. `tests/` — pytest suite. `docs/` — `DESIGN_MAPPO.md`, `ARCHITECTURE.md`.
- `archive/` — **legacy code; do NOT import or modify.** Excluded from build and tests.
- `results/` — generated output (gitignored).

## Environment & package manager
- **pip + `pyproject.toml`** (setuptools backend). `requirements.txt` (runtime), `requirements-dev.txt` (dev). No poetry/uv/lock files.
- Python: developed on **3.12**. OpenSeesPy-dependent code needs **Python 3.11–3.13** (Verify exact range);
  it fails to load on 3.14. Pure code (geometry, direct solver, non-OpenSees tests) runs on 3.10+.
- Install: `pip install -e .` then `pip install -r requirements-dev.txt`. OpenSees is optional: `pip install -e ".[ref]"`.

## Common commands
- Tests: `python -m pytest -q` (config: `testpaths=["tests"]`). Use an OpenSeesPy-capable interpreter for cross-checks.
- FEM validation: `python -m scripts.validate_fem --n 6`; `python -m scripts.validate_staged --n 6 --cable-element linear`.
- Demos / viz: `python -m scripts.oneshot_opensees --n 6 --plot results/x.png`; `python -m scripts.staged_demo --n 6`;
  `python -m scripts.plot_staged_deck_growth --n 6`.
- RL (skeleton, milestone M4): `scripts.train`, `scripts.evaluate`, `scripts.baselines`.
- Build/publish: CI builds on tag `v*.*.*` (`python -m build` + twine). Verify: CI pins Python 3.8 while `pyproject` requires `>=3.10` — likely needs alignment.
- Windows: use `py -3.12 -m ...` to select the OpenSeesPy-capable interpreter.
- Lint/format/type-check: **none configured** (Verify — ruff/black/mypy not set up). Do not assume they exist.

## Development workflow
- Source, tests, docs, and config **may be modified** when the user asks (implementation, bug fix, refactor, tests).
- Prefer **minimal, localized changes**. Do not rewrite unrelated modules.
- **Before editing more than 3 files, summarize the intended plan** and proceed after it's clear.
- Do not change public APIs unless the task requires it or the user approves.
- Milestone stage: **avoid speculative refactoring**.

## Testing rules
- Run the smallest relevant test first, then focused tests, then the broader suite before finishing.
- Add or update tests when behavior changes; add a **regression test** for every fixed bug.
- Numerical assertions use explicit tolerances; **do not weaken tests just to make them pass.**
- OpenSees-dependent tests use `pytest.importorskip("openseespy")` and stay skipped where unavailable — keep that pattern.

## Bug-fix rules
- Reproduce the issue first (or explain why reproduction is not possible). Identify the root-cause category.
- Apply the smallest fix; add a regression test; re-run focused tests.

## Feature-development rules
- Inspect existing patterns before coding. For FEM, follow the "one definition, two backends" structure and keep
  all geometry in `envs/geometry.py`.
- Add/update tests and docs when user-facing behavior changes.
- Do not add dependencies unless necessary, and **never edit `pyproject.toml`/`requirements*.txt`/lock files unless explicitly asked.**

## Refactoring rules
- Preserve behavior and public function signatures unless explicitly approved.
- Do not combine refactoring with feature/behavior changes.
- Confirm before editing more than 3 files. Run focused tests after.

## Engineering / numerical rules (important)
- **Do NOT silently change:** units (SI: N, m, Pa; cable stress reported in MPa where labeled), sign conventions,
  boundary conditions, gravity/load direction, tangent-activation / displacement lock-in semantics, cable prestress
  conventions (linear `Truss`: `σ0 = T/A`; `corotTruss`: `initStrain = T/(E·A) − ε_geo`), tolerances, or numerical assumptions.
- The self-written direct solver is **linear small-displacement**; OpenSees `corotTruss` is geometrically exact.
  Expect machine-precision agreement for completed-structure linear-vs-linear, and small staged differences
  (~0.01% at operating deflections, growing to a few % at large deflection) — **this is expected, not a bug.**
- When touching solvers, validate with `scripts.validate_fem` / `scripts.validate_staged`.

## Git / change-summary workflow
- **Do not commit unless asked.** Development happens on `master` (current convention).
- After code changes, always summarize: **(1) files changed, (2) behavior changed, (3) tests run, (4) remaining risks.**
- Commit messages: `type: short summary` then bullet details; end with the required `Co-Authored-By` trailer.
