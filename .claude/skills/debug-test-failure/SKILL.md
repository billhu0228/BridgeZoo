---
description: Debug and fix failing Python tests, pytest errors, CI failures, tracebacks, or regression failures.
argument-hint: "[optional pytest target or traceback summary]"
---

# Debug test failure

Diagnose and (when safe) fix a failing test.

## Steps
1. Run the **smallest relevant test first**: `python -m pytest <target> -q` (e.g. `tests/test_staged.py::test_name`).
   If no target is given, inspect recent failure context (last run output, recent diff) before running broad tests.
   Use an OpenSeesPy-capable interpreter (`py -3.12`) when the failure involves OpenSees.
2. Identify the **root-cause category**:
   - implementation bug
   - outdated test (intended behavior changed)
   - changed API
   - dependency/environment issue (e.g. OpenSeesPy not loadable on the interpreter, torch missing)
   - numerical tolerance issue
   - fixture/setup issue
3. Read only the files needed to understand the failure.
4. Propose the **smallest fix**.
   - If localized and low-risk, implement it.
   - If it touches public API, multiple modules, numerical assumptions, or **more than 3 files**, summarize the plan first.
   - Never silently change units, sign conventions, tolerances, or solver conventions to make a test pass.
5. Do not refactor unrelated code.
6. After fixing, run focused tests again; add a regression test if a real bug was fixed.

## Output
- Failure summary
- Root cause (category + specifics)
- Files changed
- Minimal fix applied or proposed
- Tests run
- Remaining risk
