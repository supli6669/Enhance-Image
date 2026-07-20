---
name: git-workflow-and-release-management
description: Procedures for Git version control, conventional commit messages, submodules, Hugging Face Spaces synchronization, release tagging, and automated deployment pipelines.
---

# Git Workflow & Release Management Skill

Use this skill when staging commits, managing submodules, configuring remote repositories, or releasing deployments.

## Git Standards & Best Practices

1. **Conventional Commits**:
   - Format commit messages strictly: `<type>: <description>`
     - `feat:` for new features or capabilities.
     - `fix:` for bug fixes and patches.
     - `perf:` for performance optimizations.
     - `refactor:` for code restructures without functional changes.
     - `docs:` for documentation updates.

2. **Large File & Heavy Weight Safeguards**:
   - Never commit heavy PyTorch weights (`*.pth`, `*.onnx`), model checkpoints (`net_g_*.pth`), or raw datasets to Git.
   - Verify `.gitignore` excludes `weights/`, `experiments/`, `.venv/`, `__pycache__/`, and `.lmdb/` before staging files.

3. **Multi-Remote Synchronization (GitHub + Hugging Face Spaces)**:
   - When deploying updates, push to both origin and HF Space remotes:
     ```bash
     git push origin main
     git push hf main --force
     ```

4. **Submodule Management**:
   - For vendored model submodules (`models/CodeFormer`, `models/Real-ESRGAN`), track submodule commits explicitly without nesting untracked binary weights.
