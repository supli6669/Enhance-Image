---
name: building-data-apps
description: Best practices for building interactive data applications, AI image web interfaces, Streamlit side-by-side comparison widgets, stats dashboards, and responsive layout styling.
---

# Building Data & AI Applications Skill

Use this skill when designing or enhancing web applications, interactive dashboards, image comparison tools, or data visualization UIs.

## UI Design & Architecture Guidelines

1. **Rich Modern Aesthetics**:
   - Use vibrant dark themes, glassmorphism card styling, gradient headers, and HSL-tailored color palettes.
   - Avoid generic plain browser defaults; inject custom CSS (`st.markdown("<style>...", unsafe_html=True)`).

2. **Interactive Before/After Comparison**:
   - Provide flexible view modes for image processing apps:
     - 👥 Side-by-side columns (`st.columns(2)`).
     - 🌗 Split screen image overlay.
     - ↔️ Interactive slider (`streamlit-image-comparison`).

3. **Performance Stats & Metadata Dashboard**:
   - Display key image statistics in a clean horizontal badge row: processing duration (seconds), original dimensions, enhanced dimensions, and scaling factor.

4. **Async Progress & Polling**:
   - Show real-time progress bars (`st.progress`) and stage messages ("Detecting Faces", "Upscaling Background", "Blending Faces") during background processing.
