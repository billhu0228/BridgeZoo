---
description: Safely refactor Python code while preserving behavior. Use when asked to clean up, simplify, reorganize, or refactor code.
argument-hint: "[file, module, or scope]"
disable-model-invocation: true
---

# Safe refactor

Behavior-preserving cleanup at milestone stage. Source code may be modified.

## Rules
- **Preserve behavior.** Preserve public function/method signatures unless the user explicitly approves a change.
- **Do not combine refactoring with feature or behavior changes** — refactor only.
- Do not change numerical assumptions, units, tolerances, sign conventions, boundary conditions, or algorithms
  unless explicitly requested.
- Do not touch `archive/`, `pyproject.toml`, `requirements*`, or lock files.

## Process
1. Summarize the current responsibilities of the target (what it does now).
2. Propose a small refactor plan (what moves/renames/extracts, what stays).
3. **Ask for confirmation before editing more than 3 files.**
4. Apply the refactor in small steps.
5. Run focused tests after changes; if a solver is touched, run the relevant `scripts.validate_*` to confirm
   results are unchanged.

## Output
- What changed
- What did **not** change (signatures, behavior, numerics)
- Behavior-preservation evidence (tests / validation that pass identically)
- Tests run
- Remaining risks
