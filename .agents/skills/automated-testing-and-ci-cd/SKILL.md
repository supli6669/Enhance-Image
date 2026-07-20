---
name: automated-testing-and-ci-cd
description: Best practices for writing automated unit tests, integration test harnesses, output dimension regression testing, pytest test suites, and CI/CD workflow automation.
---

# Automated Testing & CI/CD Skill

Use this skill when writing test harnesses, verifying pipeline outputs, or configuring CI/CD testing pipelines.

## Testing Standards & Guidelines

1. **Pipeline Integration Test Harness**:
   - Every major pipeline change must be verified with an automated integration test (`tools/test_pipeline.py`).
   - The test harness should process a sample image end-to-end, verify output dimensions, check non-empty tensor shapes, and assert zero exceptions.

2. **Output Regression & Dimension Validation**:
   - Assert output dimensions match mathematically expected dimensions:
     ```python
     expected_h = int(round(img.shape[0] * scale_factor)) * upscale
     expected_w = int(round(img.shape[1] * scale_factor)) * upscale
     assert enhanced_img.shape[0] == expected_h, f"Height mismatch: {enhanced_img.shape[0]} vs {expected_h}"
     assert enhanced_img.shape[1] == expected_w, f"Width mismatch: {enhanced_img.shape[1]} vs {expected_w}"
     ```

3. **Headless & Environment Isolation**:
   - Ensure tests execute cleanly in headless CPU environments without requiring GUI displays or GPU hardware.
   - Set environment variables (`CUDA_VISIBLE_DEVICES=""`, `PYTHONUNBUFFERED=1`) during test execution.
