---
description: Review this active Python project before or during a development milestone. Use when asked for milestone review, readiness check, stability review, delivery readiness, or pre-release risk review.
disable-model-invocation: true
---

# Milestone review (read-only)

Read-only assessment of project health at a milestone. **Do not edit files** unless the user explicitly
asks for follow-up fixes afterward.

## Steps
1. `git status` and `git diff --stat` (and `git log --oneline -10`) to see working-tree and recent state.
2. Detect commands from `pyproject.toml`, `README.md`, `TODO.md`, `.github/workflows/`.
3. Run safe tests if appropriate: `python -m pytest -q` (use an OpenSeesPy-capable interpreter, e.g. `py -3.12`,
   so cross-check tests run instead of skip). Do not run training, long simulations, or destructive commands.
4. Review, reading only what's needed:
   - release/milestone blockers
   - fragile modules (solvers, staged construction, env)
   - dependency risk (OpenSeesPy/Python version coupling; torch availability)
   - documentation mismatch (`README.md`, `docs/`, `TODO.md` vs. code)
   - public API risk (signatures recently changed)
   - test status (passed/skipped, and why skipped)
   - unfinished `TODO`/`FIXME` in touched or milestone-relevant files
   - numerical/engineering risks: units, sign conventions, tolerances, linear-vs-corot expectations, lock-in semantics

## Output
1. **Milestone readiness:** Ready / Almost ready / Not ready
2. Blocking issues
3. Non-blocking risks
4. Suggested next 3 actions
5. Tests run (and skipped, with reason)
6. Remaining uncertainty / items to Verify
