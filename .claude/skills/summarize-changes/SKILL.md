---
description: Summarize uncommitted git changes, explain risk, and draft a commit message.
---

# Summarize changes

Explain the current uncommitted work and draft a commit message. Read-only (no edits, no commit).

## Steps
1. `git status` — see staged/unstaged/untracked.
2. `git diff --stat` — scope of change.
3. `git diff` (and `git diff --staged`) if not too large — read the actual changes; for large diffs, summarize per file.
4. Describe the changes in **engineering terms** (what behavior/structure changed, not just line counts).
   Flag any change to units, sign conventions, tolerances, solver conventions, public APIs, or dependencies.

## Output
1. High-level summary
2. Files changed
3. Behavior changed
4. Tests affected (and whether they were run)
5. Risk points
6. Suggested commit message (format below)

## Commit message format
```
[type]: short summary

Details:
- item 1
- item 2
```
Use the project's required `Co-Authored-By` trailer in the actual commit. Do not commit unless the user asks.
