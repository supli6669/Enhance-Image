---
name: streamlit-thread-state-guidelines
description: Architectural rules and patterns for Streamlit background threading, thread-safe queue IPC, cached function guards, session_state initialization, and progress polling in AI web applications.
---

# Streamlit Multi-Threading & State Guidelines Skill

Use this skill whenever editing `app.py` or implementing background processing, state handling, or progress reporting in Streamlit.

## Critical Architectural Rules

1. **Thread-Safe Queue IPC (Never write `st.session_state` from background threads)**:
   - Streamlit forbids writing to `st.session_state` from spawned `threading.Thread` instances.
   - Use `queue.Queue` as an IPC bridge. Worker threads push `{'type': 'progress'}` or `{'type': 'result'}` dicts into the queue.
   - Main request thread reads from `queue.Queue` during the `time.sleep(0.1)` polling loop and writes to `st.session_state` safely.

2. **Param Reset Polling Guard**:
   - When checking if UI parameters have changed, NEVER reset state during active processing:
     ```python
     if st.session_state.get('last_run_params') != current_params and not st.session_state.get('processing'):
         st.session_state.processing = False
     ```
   - Omitting `and not st.session_state.get('processing')` causes an infinite loop spawning a new thread on every rerun poll cycle.

3. **Function & Helper Caching**:
   - Use `@st.cache_data(ttl=5)` for disk log reading functions (`get_training_status()`) to prevent disk I/O flooding.
   - Do NOT bind `progress_callback` inside `@st.cache_resource` initialization. Pass callbacks per call or use queue-based callbacks.
   - Snapshot Streamlit sidebar widgets into local variables before spawning background threads.

4. **None Guarding Before Processing Results**:
   - Always check `if enhanced_img is None:` before accessing `.shape` or calling OpenCV functions.
