import streamlit as st
import cv2
import numpy as np
import os
import time
import threading
import queue
from io import BytesIO
from pipeline import LocalAIEnhancerPipeline

project_dir = os.path.dirname(os.path.abspath(__file__))

# ── Page Configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Portrait Enhancer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Minimalist Premium Styling ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stApp {
    background: radial-gradient(ellipse 100% 80% at 50% -20%, rgba(124, 58, 237, 0.15) 0%, #090714 80%);
    color: #f3f0ff;
}

section[data-testid="stSidebar"] {
    background: #0d0a1d !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

.brand-header {
    background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%);
    padding: 24px 20px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(124, 58, 237, 0.25);
}

.brand-header h2 {
    color: white;
    font-weight: 800;
    font-size: 1.25rem;
    margin: 0;
    letter-spacing: 0.5px;
}

.brand-header p {
    color: rgba(255, 255, 255, 0.85);
    font-size: 0.78rem;
    margin: 4px 0 0 0;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 6px;
}

.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    text-align: center;
    margin-bottom: 32px;
}

div.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 14px 28px;
    font-weight: 700;
    font-size: 1rem;
    width: 100%;
    transition: all 0.2s ease;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.3);
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(124, 58, 237, 0.5);
}

div.stDownloadButton > button {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 14px 28px;
    font-weight: 700;
    font-size: 1rem;
    width: 100%;
    transition: all 0.2s ease;
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3);
}

div.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(16, 185, 129, 0.5);
}

.metric-badge {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
}

.metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.metric-val {
    font-size: 1.1rem;
    color: #f3f0ff;
    font-weight: 700;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Session State Initializer Guard ─────────────────────────────────────────────
for key, default in [
    ('processing', False),
    ('enhanced_img', None),
    ('processing_error', None),
    ('process_duration', None),
    ('start_time', None),
    ('last_run_params', None),
    ('progress_state', None),
    ('num_faces_detected', 0)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Load Pipeline Resource ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_pipeline():
    return LocalAIEnhancerPipeline()

try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"Failed to initialize AI Pipeline: {e}")
    pipeline = None

# ── Sidebar Controls (Minimalist & Clean) ───────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <h2>✨ Wink Studio</h2>
        <p>AI Portrait & Image Restoration</p>
    </div>
    """, unsafe_allow_html=True)

    # Preset selection
    preset_choice = st.radio(
        "Enhancement Preset",
        ["✨ Wink Studio (Best Quality)", "⚡ Ultra Fast CPU", "🎨 Natural Likeness"],
        index=0,
        help="Select pre-configured quality mode."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Core 3 Controls
    if "Wink Studio" in preset_choice:
        default_w = 0.3
        default_upscale = 2
        default_wink = True
        default_grain = 0.15
        default_color = True
        default_eye = True
        default_detector = "retinaface_mobile0.25"
    elif "Ultra Fast" in preset_choice:
        default_w = 0.5
        default_upscale = 1
        default_wink = False
        default_grain = 0.0
        default_color = False
        default_eye = False
        default_detector = "retinaface_mobile0.25"
    else: # Natural Likeness
        default_w = 0.65
        default_upscale = 2
        default_wink = True
        default_grain = 0.1
        default_color = True
        default_eye = True
        default_detector = "retinaface_mobile0.25"

    w_val = st.slider(
        "AI Detail vs Likeness (w)",
        min_value=0.0,
        max_value=1.0,
        value=default_w,
        step=0.05,
        help="0.0 = Max AI Detail restoration. 1.0 = Keep exact original face likeness."
    )

    upscale_val = st.select_slider(
        "Output Resolution Scale",
        options=[1, 2, 4],
        value=default_upscale,
        format_func=lambda x: f"{x}× Resolution"
    )

    # Advanced Settings (Collapsible to keep UI clean)
    with st.expander("⚙️ Advanced Tuning", expanded=False):
        face_detector = st.selectbox(
            "Detector Model",
            ["retinaface_mobile0.25", "retinaface_resnet50", "YOLOv5n", "YOLOv5l"],
            index=0
        )
        det_thresh = st.slider("Detection Threshold", 0.1, 1.0, 0.5, 0.05)
        wink_mode = st.toggle("Wink Quality Engine", value=default_wink)
        skin_grain = st.slider("Skin Grain Retention", 0.0, 0.5, default_grain, 0.05)
        color_match = st.checkbox("Auto Skin Tone Alignment", value=default_color)
        eye_enhancement = st.checkbox("Eye & Lip Sparkle", value=default_eye)
        bg_upscale = st.toggle("Real-ESRGAN Background Upscale", value=False)
        face_upscale = st.toggle("Real-ESRGAN Face Upscale", value=False)

# ── Main Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Portrait Enhancer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Restore blurry portraits, skin texture & eye detail with studio-level clarity</div>', unsafe_allow_html=True)

# ── File Upload Section ────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload portrait photo (PNG, JPG, WEBP)", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    # Decode Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if input_img is None:
        st.error("Could not decode image file. Please upload a valid portrait image.")
        st.stop()

    current_params = {
        'img_name': uploaded_file.name,
        'w': w_val,
        'upscale': upscale_val,
        'detector': face_detector,
        'thresh': det_thresh,
        'wink': wink_mode,
        'grain': skin_grain,
        'color': color_match,
        'eye': eye_enhancement,
        'bg_up': bg_upscale,
        'face_up': face_upscale
    }

    # Parameters change guard: reset output state if parameters change while idle
    if st.session_state.get('last_run_params') != current_params and not st.session_state.get('processing'):
        st.session_state.enhanced_img = None
        st.session_state.processing_error = None
        st.session_state.process_duration = None

    # Trigger processing thread if output is None
    if st.session_state.enhanced_img is None and st.session_state.get('processing_error') is None:
        if not st.session_state.get('processing'):
            st.session_state.processing = True
            st.session_state.start_time = time.time()
            
            res_queue = queue.Queue()
            st.session_state._result_queue = res_queue

            def local_progress_callback(stage, progress, message):
                res_queue.put({'type': 'progress', 'stage': stage, 'progress': progress, 'message': message})

            if pipeline:
                pipeline.progress_callback = local_progress_callback

            def _worker():
                try:
                    res = pipeline.process_image(
                        input_img,
                        w=w_val,
                        detection_model=face_detector,
                        upscale=upscale_val,
                        blend_softness=0.5,
                        bg_upsampler='realesrgan' if bg_upscale else None,
                        det_threshold=det_thresh,
                        face_upsample=face_upscale,
                        parallel=True,
                        wink_mode=wink_mode,
                        eye_enhancement=eye_enhancement,
                        skin_grain=skin_grain,
                        color_match=color_match
                    )
                    res_queue.put({
                        'type': 'result',
                        'enhanced_img': res,
                        'duration': time.time() - st.session_state.get('start_time', time.time()),
                        'params': current_params
                    })
                except Exception as ex:
                    import traceback
                    traceback.print_exc()
                    res_queue.put({'type': 'error', 'error': str(ex)})

            threading.Thread(target=_worker, daemon=True).start()

    # Poll Queue for updates
    if st.session_state.get('processing'):
        res_queue = st.session_state.get('_result_queue')
        if res_queue:
            while not res_queue.empty():
                msg = res_queue.get_nowait()
                if msg['type'] == 'progress':
                    st.session_state.progress_state = msg
                elif msg['type'] == 'result':
                    st.session_state.enhanced_img = msg['enhanced_img']
                    st.session_state.process_duration = msg['duration']
                    st.session_state.last_run_params = msg['params']
                    st.session_state.processing = False
                    st.session_state.progress_state = None
                    st.rerun()
                elif msg['type'] == 'error':
                    st.session_state.processing_error = msg['error']
                    st.session_state.processing = False
                    st.session_state.progress_state = None
                    st.rerun()

        # Render Progress UI
        p_state = st.session_state.get('progress_state') or {}
        stage_msg = p_state.get('message', 'Processing image with AI...')
        prog_val = p_state.get('progress', 0.1)

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(float(prog_val))
        st.info(f"✨ {stage_msg}")
        time.sleep(0.3)
        st.rerun()

    # Render Errors if any
    if st.session_state.get('processing_error'):
        st.error(f"Processing Error: {st.session_state.processing_error}")
        if st.button("🔄 Try Again"):
            st.session_state.processing_error = None
            st.session_state.processing = False
            st.rerun()

    # Render Results Section
    enhanced_img = st.session_state.get('enhanced_img')
    if enhanced_img is not None:
        st.markdown("<hr>", unsafe_allow_html=True)

        # Image Stats Bar
        in_h, in_w = input_img.shape[:2]
        out_h, out_w = enhanced_img.shape[:2]
        duration = st.session_state.get('process_duration', 0.0)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-badge"><div class="metric-label">Original Size</div><div class="metric-val">{in_w}×{in_h} px</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-badge"><div class="metric-label">Enhanced Size</div><div class="metric-val">{out_w}×{out_h} px</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-badge"><div class="metric-label">Speed (CPU)</div><div class="metric-val">{duration:.2f} s</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Side-by-Side Comparison Display
        c_orig, c_enh = st.columns(2)
        with c_orig:
            st.markdown("##### 📷 Original Image")
            st.image(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with c_enh:
            st.markdown("##### ✨ Wink Enhanced HD")
            st.image(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Download Button
        success, encoded_buf = cv2.imencode('.png', enhanced_img)
        if success:
            st.download_button(
                label="⬇️ Download Enhanced HD Image",
                data=encoded_buf.tobytes(),
                file_name=f"enhanced_{uploaded_file.name}",
                mime="image/png"
            )
else:
    # Empty State Guide
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; border: 2px dashed rgba(255,255,255,0.1); border-radius: 20px; background: rgba(255,255,255,0.01);">
        <p style="font-size: 1.2rem; color: #94a3b8; font-weight: 600;">Drag and drop any portrait photo above to get started</p>
        <p style="font-size: 0.9rem; color: #64748b; margin-top: 8px;">Supports PNG, JPG, JPEG, WEBP. Optimized for fast CPU execution.</p>
    </div>
    """, unsafe_allow_html=True)
