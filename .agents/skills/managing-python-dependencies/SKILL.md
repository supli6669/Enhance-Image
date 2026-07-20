---
name: managing-python-dependencies
description: Rules for Python environment isolation, wheel compilation, dependency resolution, handling BasicSR setup.py version parsing patches, and requirements.txt maintenance.
---

# Managing Python Dependencies Skill

Use this skill when installing packages, resolving build incompatibilities, building standalone wheels, or updating `requirements.txt`.

## Dependency Hygiene Rules

1. **Virtual Environment Isolation**:
   - Always run Python and pip commands using the project's local virtual environment (`.venv\Scripts\python.exe` / `.venv\Scripts\pip.exe`).
   - Never install packages into global system Python.

2. **BasicSR & Python 3.11+ Version Patch**:
   - Standard BasicSR `setup.py` fails on Python 3.11+ due to `KeyError: '__version__'` in `locals()`.
   - Patch `setup.py` to use explicit dictionary execution: `g = {}; exec(open(version_file).read(), g); return g['__version__']`.
   - Set `BASICSR_EXT=False` during installation to force pure Python build without requiring a CUDA C++ compiler.

3. **Requirements Locking & Cleaning**:
   - Keep `requirements.txt` concise and clean. Avoid committing fake version numbers or platform-dependent build paths.
