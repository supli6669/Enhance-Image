---
name: accidental-data-loss-prevention
description: Safeguards and verification procedures before running destructive file system operations, recursive directory deletions, force git pushes, or model checkpoint overwrites.
---

# Accidental Data Loss Prevention Skill

Use this skill whenever performing operations that could result in irreversible data loss, deleting directories, overwriting model checkpoints, or forcing git branches.

## Data Protection Safeguards

1. **Pre-Deletion Verification**:
   - Before deleting files or directories, verify target paths strictly.
   - Do NOT run recursive `Remove-Item -Recurse` or `rm -rf` on root directories or top-level project folders.

2. **Model Checkpoint Safeguards**:
   - Never overwrite active training checkpoints (`net_g_latest.pth`, `*.state`) without backing up current state.
   - Use temp files when generating runtime options (`cf_runtime_*.yml`) so original config YAML files are never corrupted.

3. **Git Branch & Remote Safety**:
   - Verify active branch before running `git reset` or `git push --force`.
   - Ensure local changes are committed or stashed before checking out or pulling branches.
