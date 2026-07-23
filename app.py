import streamlit as st
import cv2
import numpy as np
import os
import time
import threading
import queue
import re
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from pipeline import LocalAIEnhancerPipeline

project_dir = os.path.dirname(os.path.abspath(__file__))
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


def load_uploaded_image(uploaded_file):
    """Validate upload limits before handing image bytes to OpenCV."""
    file_data = uploaded_file.getvalue()
    if len(file_data) > MAX_UPLOAD_BYTES:
        raise ValueError("Image is too large. Please upload a file smaller than 15 MB.")

    try:
        with Image.open(BytesIO(file_data)) as image:
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    "Image resolution is too large. Please upload an image up to 16 megapixels."
                )
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Could not decode image file. Please upload a valid portrait image.") from error

    decoded_image = cv2.imdecode(np.frombuffer(file_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded_image is None:
        raise ValueError("Could not decode image file. Please upload a valid portrait image.")
    if decoded_image.shape[0] * decoded_image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError("Image resolution is too large. Please upload an image up to 16 megapixels.")
    return decoded_image


def enhanced_filename(original_name):
    """Produce a download-safe PNG filename without trusting the upload path."""
    base_name = os.path.basename(original_name.replace("\\", "/"))
    stem, _ = os.path.splitext(base_name)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return f"enhanced_{safe_stem or 'portrait'}.png"

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
        sharpen_val = st.slider("🔥 Extra Sharpness Boost", 0.0, 1.0, 0.2, 0.05, help="Multi-scale edge-aware adaptive sharpening")
        color_match = st.checkbox("Auto Skin Tone Alignment", value=default_color)
        
        st.markdown("**🎭 Facial Organ Enhancements**")
        enable_eyes = st.checkbox("👁️ Eye Sparkle & Contrast Boost", value=default_eye)
        enable_lips = st.checkbox("👄 Lip Saturation & Definition", value=True)
        enable_skin = st.checkbox("💆 Real Skin Grain Retention", value=True)

        bg_upscale = st.toggle("Real-ESRGAN Background Upscale", value=False)
        face_upscale = st.toggle("Real-ESRGAN Face Upscale", value=False)

# ── Main Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Portrait Enhancer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Restore blurry portraits, skin texture & eye detail with studio-level clarity</div>', unsafe_allow_html=True)

# ── File Upload Section ────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload portrait photo (PNG, JPG, WEBP)", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    try:
        input_img = load_uploaded_image(uploaded_file)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    current_params = {
        'img_name': uploaded_file.name,
        'w': w_val,
        'upscale': upscale_val,
        'detector': face_detector,
        'thresh': det_thresh,
        'wink': wink_mode,
        'grain': skin_grain,
        'sharpen': sharpen_val,
        'color': color_match,
        'eye': enable_eyes,
        'lip': enable_lips,
        'skin': enable_skin,
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
            if pipeline is None:
                st.session_state.processing_error = "AI pipeline is unavailable. Please try again later."
                st.rerun()

            st.session_state.processing = True
            request_start_time = time.time()
            st.session_state.start_time = request_start_time
            
            res_queue = queue.Queue()
            st.session_state._result_queue = res_queue

            def local_progress_callback(stage, progress, message):
                res_queue.put({'type': 'progress', 'stage': stage, 'progress': progress, 'message': message})

            process_args = {
                'w': w_val,
                'detection_model': face_detector,
                'upscale': upscale_val,
                'blend_softness': 0.5,
                'bg_upsampler': 'realesrgan' if bg_upscale else None,
                'det_threshold': det_thresh,
                'sharpen_amount': sharpen_val,
                'face_upsample': face_upscale,
                'parallel': True,
                'wink_mode': wink_mode,
                'eye_enhancement': enable_eyes,
                'skin_grain': skin_grain,
                'color_match': color_match,
                'enable_eyes': enable_eyes,
                'enable_lips': enable_lips,
                'enable_skin': enable_skin,
                'progress_callback': local_progress_callback,
            }

            def _worker(
                request_image=input_img.copy(),
                request_params=current_params.copy(),
                request_args=process_args.copy(),
                request_queue=res_queue,
                request_started_at=request_start_time,
            ):
                try:
                    res = pipeline.process_image(
                        request_image,
                        **request_args,
                    )

                    request_queue.put({
                        'type': 'result',
                        'enhanced_img': res,
                        'duration': time.time() - request_started_at,
                        'params': request_params
                    })
                except Exception as ex:
                    import traceback
                    traceback.print_exc()
                    request_queue.put({'type': 'error', 'error': str(ex)})

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

        # AI Quality Score Report Card
        if pipeline and hasattr(pipeline, 'wink_enhancer'):
            q_report = pipeline.wink_enhancer.calculate_quality_report(input_img, enhanced_img)
            st.markdown("#### 📊 AI Quality Score Report")
            q1, q2, q3, q4 = st.columns(4)
            with q1:
                st.markdown(f'<div class="metric-badge"><div class="metric-label">Sharpness Gain</div><div class="metric-val" style="color: #34d399;">+{q_report["sharpness_gain_pct"]}%</div></div>', unsafe_allow_html=True)
            with q2:
                st.markdown(f'<div class="metric-badge"><div class="metric-label">Original Sharpness</div><div class="metric-val">{q_report["orig_sharpness"]}</div></div>', unsafe_allow_html=True)
            with q3:
                st.markdown(f'<div class="metric-badge"><div class="metric-label">Enhanced Sharpness</div><div class="metric-val">{q_report["enh_sharpness"]}</div></div>', unsafe_allow_html=True)
            with q4:
                st.markdown(f'<div class="metric-badge"><div class="metric-label">Skin Tone Match</div><div class="metric-val" style="color: #60a5fa;">{q_report["tone_fidelity_pct"]}%</div></div>', unsafe_allow_html=True)

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
                file_name=enhanced_filename(uploaded_file.name),
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
