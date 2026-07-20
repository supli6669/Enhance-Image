---
name: systematic-debugging
description: Use when investigating bugs, crashes, memory leaks, segfaults, UI freezes, or unexpected errors. Enforces systematic root-cause analysis over trial-and-error guessing.
---

# Systematic Debugging Skill

Use this skill whenever a bug, error log, crash, memory leak, or unexpected behavior is reported or observed.

## Procedural Debugging Steps

1. **Minimal Reproduction**:
   - Create a minimal, isolated reproduction script or command that triggers the bug reliably.
   - Capture full error output, stack traces, and relevant system state.

2. **Root Cause Analysis (RCA)**:
   - Trace stack trace line numbers directly into source code.
   - Inspect variable states, tensor shapes, and thread interactions immediately preceding the failure.
   - Identify the exact flaw (e.g. state mutation across threads, shape mismatch, missing DLL, unhandled exception).

3. **Hypothesis & Targeted Fix**:
   - Formulate a precise hypothesis explaining why the bug occurs.
   - Implement the smallest, most targeted fix that addresses the root cause directly without side effects.

4. **Regression Testing & Verification**:
   - Run the reproduction harness to confirm the fix works.
   - Run existing test suites to verify zero regressions.
