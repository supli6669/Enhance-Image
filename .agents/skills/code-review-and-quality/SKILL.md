---
name: code-review-and-quality
description: Use when performing code reviews, code refactoring, code quality audits, or linting code for maintainability, readability, type safety, and correctness.
---

# Code Review & Quality Skill

Use this skill during code reviews, refactoring sessions, or code quality audits.

## Quality Checklist & Guidelines

1. **Correctness & Robustness**:
   - Verify all edge cases are handled (empty inputs, None values, resource exhaustion).
   - Ensure proper error handling and descriptive error messages rather than silent failures.

2. **Performance & Resource Hygiene**:
   - Verify no memory leaks, unclosed file handles, or un-garbage-collected heavy objects (tensors, ONNX sessions).
   - Avoid redundant compute in loops or unnecessary array copies.

3. **Readability & Maintainability**:
   - Keep functions focused and single-purpose (SOLID principles).
   - Add type hints and docstrings for key public APIs.
   - Maintain consistency with repository style and existing abstractions.

4. **Security & Data Integrity**:
   - Ensure input validation and sanitization.
   - Prevent hardcoded secrets or unsafe file path traversals.
