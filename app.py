import streamlit as st
import cv2
import numpy as np
import os
import time
import threading
import queue
import re
import glob
from datetime import datetime, timezone
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from pipeline import LocalAIEnhancerPipeline

project_dir = os.path.dirname(os.path.abspath(__file__))
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


def load_uploaded_image(uploaded_file):
    """Validate upload limits and decode image bytes with auto EXIF orientation."""
    file_data = uploaded_file.getvalue()
    if len(file_data) > MAX_UPLOAD_BYTES:
        raise ValueError("Image is too large. Please upload a file smaller than 15 MB.")

    try:
        from PIL import ImageOps
        pil_img = Image.open(BytesIO(file_data))
        pil_img = ImageOps.exif_transpose(pil_img)
        
        # Convert RGBA/Palette/Grayscale/CMYK to standard 3-channel RGB
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
            
        width, height = pil_img.size
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError("Image resolution is too large. Please upload an image up to 16 megapixels.")
            
        rgb_arr = np.array(pil_img)
        decoded_image = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        return decoded_image
    except ValueError:
        raise
    except Exception as error:
        # Fallback to OpenCV decode
        decoded_image = cv2.imdecode(np.frombuffer(file_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded_image is None:
            raise ValueError("Could not decode image file. Please upload a valid JPG, PNG, or WEBP portrait.") from error
        if decoded_image.shape[0] * decoded_image.shape[1] > MAX_IMAGE_PIXELS:
            raise ValueError("Image resolution is too large. Please upload an image up to 16 megapixels.")
        return decoded_image


def enhanced_filename(original_name):
    """Produce a download-safe PNG filename without trusting the upload path."""
    base_name = os.path.basename(original_name.replace("\\", "/"))
    stem, _ = os.path.splitext(base_name)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return f"enhanced_{safe_stem or 'portrait'}.png"


TRAIN_LOG_GLOB = os.path.join(project_dir, "models", "CodeFormer", "experiments", "*_CodeFormer_stage3_custom", "train_*.log")
TRAIN_STATE_GLOB = os.path.join(
    project_dir, "models", "CodeFormer", "experiments", "*_CodeFormer_stage3_custom", "training_states", "*.state"
)
TRAIN_TOTAL_ITERATIONS = 20_000


@st.cache_data(ttl=10, show_spinner=False)
def get_training_status(log_glob, state_glob):
    """Read only the recent training log; safe to call during a live CPU run."""
    records = []
    log_mtime = None
    try:
        log_paths = glob.glob(log_glob)
        if not log_paths:
            return records, log_mtime, (None, None)
        log_path = max(log_paths, key=os.path.getmtime)
        log_mtime = os.path.getmtime(log_path)
        with open(log_path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 350_000))
            text = handle.read().decode("utf-8", errors="replace")
        pattern = re.compile(
            r"iter:\s*([\d,]+).*?eta:\s*(.*?),\s*time.*?"
            r"l_g_pix:\s*([\deE+\-.]+).*?l_g_percep:\s*([\deE+\-.]+).*?"
            r"l_g_identity:\s*([\deE+\-.]+)"
        )
        for match in pattern.finditer(text):
            records.append({
                "iteration": int(match.group(1).replace(",", "")),
                "eta": match.group(2).strip(),
                "pixel_loss": float(match.group(3)),
                "perceptual_loss": float(match.group(4)),
                "identity_loss": float(match.group(5)),
            })
    except OSError:
        pass

    checkpoints = []
    for path in glob.glob(state_glob):
        try:
            checkpoints.append((int(os.path.splitext(os.path.basename(path))[0]), os.path.getmtime(path)))
        except (OSError, ValueError):
            continue
    latest_checkpoint = max(checkpoints, default=(None, None), key=lambda value: value[0])
    return records[-100:], log_mtime, latest_checkpoint


def render_training_dashboard():
    records, log_mtime, latest_checkpoint = get_training_status(TRAIN_LOG_GLOB, TRAIN_STATE_GLOB)
    latest = records[-1] if records else None
    is_running = bool(log_mtime and (datetime.now(timezone.utc).timestamp() - log_mtime < 180))

    st.markdown("<div class='training-panel'><div class='training-kicker'>LIVE TRAINING CONTROL ROOM</div><h2>CodeFormer CPU Training</h2><p>Read-only monitor — viewing this dashboard does not pause or compete with training.</p></div>", unsafe_allow_html=True)
    if st.button("↻ Refresh training data", key="refresh_training"):
        get_training_status.clear()
        st.rerun()

    if not latest:
        st.info("No training metrics found yet. Start training or refresh after the first iteration.")
        return

    progress = min(latest["iteration"] / TRAIN_TOTAL_ITERATIONS, 1.0)
    st.progress(progress, text=f"Iteration {latest['iteration']:,} / {TRAIN_TOTAL_ITERATIONS:,} ({progress:.1%})")
    c1, c2, c3, c4 = st.columns(4)
    status_label = "● RUNNING" if is_running else "● PAUSED / STOPPED"
    status_color = "#34d399" if is_running else "#fbbf24"
    with c1:
        st.markdown(f"<div class='metric-badge'><div class='metric-label'>Status</div><div class='metric-val' style='color:{status_color}'>{status_label}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-badge'><div class='metric-label'>ETA</div><div class='metric-val'>{latest['eta']}</div></div>", unsafe_allow_html=True)
    with c3:
        checkpoint = latest_checkpoint[0] if latest_checkpoint[0] is not None else "—"
        st.markdown(f"<div class='metric-badge'><div class='metric-label'>Last Checkpoint</div><div class='metric-val'>{checkpoint}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-badge'><div class='metric-label'>Identity Loss</div><div class='metric-val'>{latest['identity_loss']:.4f}</div></div>", unsafe_allow_html=True)

    chart_data = [{"iteration": row["iteration"], "Pixel loss": row["pixel_loss"], "Perceptual loss": row["perceptual_loss"], "Identity loss": row["identity_loss"]} for row in records]
    st.markdown("#### Loss trends (latest 100 iterations)")
    st.line_chart(chart_data, x="iteration", y=["Pixel loss", "Perceptual loss", "Identity loss"], color=["#a78bfa", "#f472b6", "#34d399"], height=260)

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

.training-panel {
    margin: 14px 0 20px;
    padding: 22px 24px;
    border: 1px solid rgba(167, 139, 250, 0.3);
    border-radius: 18px;
    background: linear-gradient(120deg, rgba(124, 58, 237, 0.20), rgba(219, 39, 119, 0.10));
}
.training-panel h2 { margin: 4px 0; color: #fff; font-size: 1.45rem; }
.training-panel p { margin: 0; color: #c4b5fd; }
.training-kicker { color: #f9a8d4; font-size: .72rem; font-weight: 800; letter-spacing: .12em; }
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
    ('num_faces_detected', 0),
    ('custom_presets', {}),
    ('comparison_gif', None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Load Pipeline Resource ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_pipeline():
    return LocalAIEnhancerPipeline()

# The dashboard must not initialise the heavy model while CPU training is live.
pipeline = None

APP_VERSION = "v2.9.0 (Build 2026.09.02)"

# ── Sidebar Controls (Minimalist & Clean) ───────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="brand-header">
        <h2>✨ Wink Studio</h2>
        <p>AI Portrait & Image Restoration</p>
        <div style="margin-top: 6px;">
            <span style="background: rgba(99, 102, 241, 0.18); color: #818cf8; font-size: 0.76rem; padding: 3px 8px; border-radius: 9999px; font-weight: 600; border: 1px solid rgba(99, 102, 241, 0.3);">{APP_VERSION}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Base preset options + any user created presets
    base_presets = [
        "💎 Pure Quality & Sharpness (Mọi Loại Ảnh - 100% Trung Thực)",
        "🌿 Natural Likeness (Chân Dung Tự Nhiên)",
        "✨ Wink Studio (Chân Dung Nghệ Thuật)",
        "⚡ Ultra Fast CPU",
        "📜 Old Photo Restoration",
        "🎮 Game / Anime Character"
    ]
    all_preset_options = base_presets + [f"⭐ {k}" for k in st.session_state.custom_presets.keys()]

    preset_choice = st.selectbox(
        "Enhancement Preset",
        all_preset_options,
        index=0,
        help="Select pre-configured quality mode or your own custom saved presets."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Determine default values from preset
    if preset_choice.startswith("⭐ "):
        custom_key = preset_choice[2:]
        cp = st.session_state.custom_presets.get(custom_key, {})
        default_w = cp.get('w', 0.8)
        default_upscale = cp.get('upscale', 2)
        default_wink = cp.get('wink', False)
        default_grain = cp.get('grain', 0.20)
        default_sharpen = cp.get('sharpen', 0.25)
        default_color = cp.get('color', True)
        default_eye = cp.get('eye', False)
        default_lip = cp.get('lip', False)
        default_skin = cp.get('skin', False)
        default_teeth = cp.get('teeth', False)
        default_tone_glow = cp.get('tone_glow', False)
        default_dark_circles = cp.get('dark_circles', False)
        default_catchlight = cp.get('catchlight', False)
        default_hair = cp.get('hair', False)
        default_relighting = cp.get('relighting', False)
        default_anti_glare = cp.get('anti_glare', False)
        default_makeup = cp.get('makeup', False)
        default_blush = cp.get('blush', 0.0)
        default_eyebrow = cp.get('eyebrow', 0.0)
        default_crystal_skin = cp.get('crystal_skin', False)
        default_crystal_skin_val = cp.get('crystal_skin_val', 0.0)
        default_glossy_lips = cp.get('glossy_lips', False)
        default_lip_gloss = cp.get('lip_gloss', 0.0)
        default_lip_vibrance = cp.get('lip_vibrance', 0.0)
        default_doll_eye = cp.get('doll_eye', False)
        default_doll_eye_depth = cp.get('doll_eye_depth', 0.0)
        default_golden_hour = cp.get('golden_hour', False)
        default_golden_warmth = cp.get('golden_warmth', 0.0)
        default_super_clarity = cp.get('super_clarity', True)
        default_clarity_val = cp.get('clarity_val', 0.45)
        default_deblur = cp.get('deblur', True)
        default_deblur_val = cp.get('deblur_val', 0.35)
        default_dehaze = cp.get('dehaze', True)
        default_dehaze_val = cp.get('dehaze_val', 0.25)
        default_bokeh = cp.get('bokeh', 0.0)
        default_lut = cp.get('lut', 'None')
        default_chromatic = cp.get('chromatic', False)
        default_detector = cp.get('detector', 'retinaface_mobile0.25')
        pipeline_preset_mode = 'Custom'
    elif "Pure Quality & Sharpness" in preset_choice:
        default_w = 0.85
        default_upscale = 2
        default_wink = False
        default_grain = 0.20
        default_sharpen = 0.25
        default_color = True
        default_eye = False
        default_lip = False
        default_skin = False
        default_teeth = False
        default_tone_glow = False
        default_dark_circles = False
        default_catchlight = False
        default_hair = False
        default_relighting = False
        default_anti_glare = False
        default_makeup = False
        default_blush = 0.0
        default_eyebrow = 0.0
        default_crystal_skin = False
        default_crystal_skin_val = 0.0
        default_glossy_lips = False
        default_lip_gloss = 0.0
        default_lip_vibrance = 0.0
        default_doll_eye = False
        default_doll_eye_depth = 0.0
        default_golden_hour = False
        default_golden_warmth = 0.0
        default_super_clarity = True
        default_clarity_val = 0.45
        default_deblur = True
        default_deblur_val = 0.35
        default_dehaze = True
        default_dehaze_val = 0.25
        default_bokeh = 0.0
        default_lut = "None"
        default_chromatic = False
        default_detector = "retinaface_mobile0.25"
        pipeline_preset_mode = 'Pure Quality'
    elif "Wink Studio" in preset_choice:
        default_w = 0.3
        default_upscale = 2
        default_wink = True
        default_grain = 0.15
        default_sharpen = 0.2
        default_color = True
        default_eye = True
        default_lip = True
        default_skin = True
        default_teeth = True
        default_tone_glow = True
        default_dark_circles = True
        default_catchlight = True
        default_hair = True
        default_relighting = True
        default_anti_glare = True
        default_makeup = True
        default_blush = 0.30
        default_eyebrow = 0.35
        default_crystal_skin = True
        default_crystal_skin_val = 0.45
        default_glossy_lips = True
        default_lip_gloss = 0.40
        default_lip_vibrance = 0.25
        default_doll_eye = True
        default_doll_eye_depth = 0.45
        default_golden_hour = False
        default_golden_warmth = 0.25
        default_super_clarity = True
        default_clarity_val = 0.40
        default_deblur = False
        default_deblur_val = 0.35
        default_dehaze = True
        default_dehaze_val = 0.25
        default_bokeh = 0.0
        default_lut = "None"
        default_chromatic = False
        default_detector = "retinaface_mobile0.25"
        pipeline_preset_mode = 'Modern Portrait'
    elif "Ultra Fast" in preset_choice:
        default_w = 0.5
        default_upscale = 1
        default_wink = False
        default_grain = 0.0
        default_sharpen = 0.0
        default_color = False
        default_eye = False
        default_lip = False
        default_skin = False
        default_teeth = False
        default_tone_glow = False
        default_dark_circles = False
        default_catchlight = False
        default_hair = False
        default_relighting = False
        default_anti_glare = False
        default_makeup = False
        default_blush = 0.0
        default_eyebrow = 0.0
        default_crystal_skin = False
        default_crystal_skin_val = 0.0
        default_glossy_lips = False
        default_lip_gloss = 0.0
        default_lip_vibrance = 0.0
        default_doll_eye = False
        default_doll_eye_depth = 0.0
        default_golden_hour = False
        default_golden_warmth = 0.0
        default_super_clarity = False
        default_clarity_val = 0.0
        default_deblur = False
        default_deblur_val = 0.0
        default_dehaze = False
        default_dehaze_val = 0.0
        default_bokeh = 0.0
        default_lut = "None"
        default_chromatic = False
        default_detector = "retinaface_mobile0.25"
        pipeline_preset_mode = 'Custom'
    elif "Old Photo" in preset_choice:
        default_w = 0.85
        default_upscale = 2
        default_wink = True
        default_grain = 0.05
        default_sharpen = 0.15
        default_color = True
        default_eye = True
        default_lip = True
        default_skin = True
        default_teeth = True
        default_tone_glow = True
        default_dark_circles = True
        default_catchlight = True
        default_hair = True
        default_relighting = True
        default_anti_glare = True
        default_makeup = True
        default_blush = 0.20
        default_eyebrow = 0.40
        default_crystal_skin = True
        default_crystal_skin_val = 0.35
        default_glossy_lips = True
        default_lip_gloss = 0.30
        default_lip_vibrance = 0.20
        default_doll_eye = True
        default_doll_eye_depth = 0.40
        default_golden_hour = True
        default_golden_warmth = 0.30
        default_super_clarity = True
        default_clarity_val = 0.50
        default_deblur = True
        default_deblur_val = 0.40
        default_dehaze = True
        default_dehaze_val = 0.35
        default_bokeh = 0.0
        default_lut = "Kodak Portra 400 (Warm Gold)"
        default_chromatic = True
        default_detector = "retinaface_mobile0.25"
        pipeline_preset_mode = 'Old Photo Restoration'
    elif "Game / Anime" in preset_choice:
        default_w = 0.3
        default_upscale = 2
        default_wink = True
        default_grain = 0.0
        default_sharpen = 0.1
        default_color = False
        default_eye = False
        default_lip = False
        default_skin = False
        default_teeth = False
        default_tone_glow = False
        default_dark_circles = False
        default_catchlight = True
        default_hair = True
        default_relighting = True
        default_anti_glare = False
        default_makeup = False
        default_blush = 0.0
        default_eyebrow = 0.0
        default_crystal_skin = True
        default_crystal_skin_val = 0.55
        default_glossy_lips = True
        default_lip_gloss = 0.50
        default_lip_vibrance = 0.35
        default_doll_eye = True
        default_doll_eye_depth = 0.60
        default_golden_hour = False
        default_golden_warmth = 0.0
        default_super_clarity = True
        default_clarity_val = 0.30
        default_deblur = False
        default_deblur_val = 0.0
        default_dehaze = False
        default_dehaze_val = 0.0
        default_bokeh = 0.0
        default_lut = "Teal & Orange / Cyberpunk"
        default_chromatic = False
        default_detector = "retinaface_mobile0.25"
        pipeline_preset_mode = 'Game / Anime Character'
    else: # Natural Likeness
        default_w = 0.65
        default_upscale = 2
        default_wink = True
        default_grain = 0.1
        default_sharpen = 0.15
        default_color = True
        default_eye = True
        default_lip = True
        default_skin = True
        default_teeth = True
        default_tone_glow = True
        default_dark_circles = True
        default_catchlight = True
        default_hair = True
        default_relighting = False
        default_anti_glare = True
        default_makeup = False
        default_blush = 0.0
        default_eyebrow = 0.20
        default_crystal_skin = True
        default_crystal_skin_val = 0.35
        default_glossy_lips = False
        default_lip_gloss = 0.0
        default_lip_vibrance = 0.0
        default_doll_eye = True
        default_doll_eye_depth = 0.30
        default_golden_hour = False
        default_golden_warmth = 0.0
        default_super_clarity = True
        default_clarity_val = 0.35
        default_deblur = False
        default_deblur_val = 0.0
        default_dehaze = True
        default_dehaze_val = 0.20
        default_bokeh = 0.0
        default_lut = "None"
        default_chromatic = False
        default_detector = "retinaface_mobile0.25"
        pipeline_preset_mode = 'Custom'

    # Model Version Switcher
    avail_models = {}
    try:
        if pipeline is not None:
            avail_models = pipeline.get_available_models()
    except Exception:
        pass

    model_options = ["Auto (Recommended)"] + list(avail_models.keys())
    selected_model_ver = st.selectbox(
        "🧠 AI Model Engine",
        model_options,
        index=0,
        help="Select neural network weights (INT8 Fast CPU, ArcFace Cloud Fine-Tuned, or Baseline)."
    )

    # Universal / Background Upscaler Switcher
    avail_upscalers = {}
    try:
        if pipeline is not None:
            avail_upscalers = pipeline.get_available_upscalers()
    except Exception:
        pass

    upscaler_options = list(avail_upscalers.keys()) if avail_upscalers else ["Auto (Real-ESRGAN / Lanczos)"]
    selected_upscaler = st.selectbox(
        "🌐 Universal Super-Resolution Engine",
        upscaler_options,
        index=0,
        help="Select neural network for full-image super-resolution (landscapes, anime, products, textures, background)."
    )

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
        options=[1, 2, 4, 8],
        value=default_upscale,
        format_func=lambda x: f"{x}× Resolution" + (" (8K Ultra-HD)" if x == 8 else "")
    )

    # Advanced Settings (Collapsible to keep UI clean)
    with st.expander("⚙️ Advanced Tuning & Presets", expanded=False):
        face_detector = st.selectbox(
            "Detector Model",
            ["retinaface_mobile0.25", "retinaface_resnet50", "YOLOv5n", "YOLOv5l"],
            index=0
        )
        det_thresh = st.slider("Detection Threshold", 0.1, 1.0, 0.5, 0.05)
        wink_mode = st.toggle("Wink Quality Engine", value=default_wink)
        skin_grain = st.slider("Skin Grain Retention", 0.0, 0.5, default_grain, 0.05)
        sharpen_val = st.slider("🔥 Extra Sharpness Boost", 0.0, 1.0, default_sharpen, 0.05, help="Multi-scale edge-aware adaptive sharpening")
        color_match = st.checkbox("Auto Skin Tone Alignment", value=default_color)

        st.markdown("**🔬 Razor-Sharp & Super-Clarity Engine**")
        enable_super_clarity = st.checkbox("🔬 Laplacian Multi-Scale Super-Clarity", value=default_super_clarity, help="Tăng nét vi chi tiết đa tầng (lỗ chân lông, sợi mi, kẽ tóc) không quầng sáng")
        clarity_val = st.slider("Micro-Texture Boost", 0.0, 1.0, default_clarity_val, 0.05) if enable_super_clarity else 0.0
        enable_deblur = st.checkbox("🌊 Optical De-Blur (Khử Nhòe Rung Tay & Out Nét)", value=default_deblur, help="Tái tạo viền nét sắc nhọn cho ảnh mờ out nét")
        deblur_val = st.slider("De-Blur Strength", 0.0, 1.0, default_deblur_val, 0.05) if enable_deblur else 0.0
        enable_dehaze = st.checkbox("✨ Crystal De-Haze & Deep Contrast (Khử Màng Sương Mờ)", value=default_dehaze, help="Khử lớp màng mờ xám giúp ảnh trong veo và tương phản sâu")
        dehaze_val = st.slider("De-Haze Strength", 0.0, 1.0, default_dehaze_val, 0.05) if enable_dehaze else 0.0

        st.markdown("**🌟 Ultimate Glamour & Aesthetics**")
        enable_crystal_skin = st.checkbox("💎 Poreless Crystal Skin (Mịn Da Pha Lê)", value=default_crystal_skin, help="Mịn da tự nhiên, xóa mụn tàn nhang mà vẫn giữ trọn lỗ chân lông thật")
        crystal_skin_val = st.slider("Skin Smooth Level", 0.0, 1.0, default_crystal_skin_val, 0.05) if enable_crystal_skin else 0.0
        enable_glossy_lips = st.checkbox("👄 3D Glassy Gloss Lips (Môi Căng Mọng Nước)", value=default_glossy_lips, help="Tạo điểm sáng 3D thủy tinh và tăng sắc môi rạng rỡ")
        lip_gloss_val = st.slider("Lip Glass Shine", 0.0, 1.0, default_lip_gloss, 0.05) if enable_glossy_lips else 0.0
        lip_vibrance_val = st.slider("Lip Color Vibrance", 0.0, 1.0, default_lip_vibrance, 0.05) if enable_glossy_lips else 0.0
        enable_doll_eye = st.checkbox("👁️ Doll-Eye & Limbal Ring (Mắt Búp Bê Có Hồn)", value=default_doll_eye, help="Tăng chiều sâu viền đen con ngươi và làm sáng mắt trong veo")
        doll_eye_val = st.slider("Eye Soul Depth", 0.0, 1.0, default_doll_eye_depth, 0.05) if enable_doll_eye else 0.0
        enable_golden_hour = st.checkbox("🌅 Sun-Kissed Golden Hour Glow (Nắng Hoàng Hôn Ấm Áp)", value=default_golden_hour, help="Hiệu ứng ánh sáng studio ấm áp và ánh hào quang mơ màng")
        golden_warmth_val = st.slider("Golden Warmth", 0.0, 1.0, default_golden_warmth, 0.05) if enable_golden_hour else 0.0
        
        st.markdown("**🎭 Facial Organ Enhancements**")
        enable_eyes = st.checkbox("👁️ Eye Sparkle & Sclera Glow", value=default_eye)
        enable_lips = st.checkbox("👄 Lip Saturation & Definition", value=default_lip)
        enable_skin = st.checkbox("💆 Real Skin Grain Retention", value=default_skin)
        enable_teeth = st.checkbox("🦷 Natural Teeth Whitening", value=default_teeth)
        enable_tone_glow = st.checkbox("✨ Studio Skin Glow & White Balance", value=default_tone_glow)
        enable_dark_circles = st.checkbox("🌿 Under-Eye Dark Circles & Blemish Concealer", value=default_dark_circles)

        st.markdown("**✨ Studio Glamour & Lighting**")
        enable_catchlight = st.checkbox("👁️ Catchlight Studio Glow (Mắt Long Lanh)", value=default_catchlight, help="Tạo đốm sáng phản chiếu softbox/ringlight trong con ngươi mắt")
        enable_hair = st.checkbox("💇 Hair Strand Super-Clarity & Gloss (Tóc Bóng Mượt)", value=default_hair, help="Tách rõ từng lọn tóc và tăng độ bóng mượt")
        enable_relighting = st.checkbox("✨ 3D Studio Relighting & Highlighter (Đánh Đèn Studio)", value=default_relighting, help="Bắt sáng sống mũi T-Zone và tạo viền sáng ven tóc Rim Light")

        st.markdown("**💄 Studio Beauty & Makeup**")
        enable_anti_glare = st.checkbox("🧽 AI Anti-Glare & Matte Skin (Khử Cháy Sáng & Dầu)", value=default_anti_glare, help="Phục hồi vùng da bóng dầu/cháy sáng flash thành tone da mịn tự nhiên")
        enable_makeup = st.checkbox("💄 Natural Studio Makeup (Trang Điểm Má Hồng & Chân Mày)", value=default_makeup, help="Đánh má hồng tự nhiên và định hình nét chân mày")
        blush_val = st.slider("Rosy Cheek Blush", 0.0, 1.0, default_blush, 0.05) if enable_makeup else 0.0
        eyebrow_val = st.slider("Eyebrow Definition", 0.0, 1.0, default_eyebrow, 0.05) if enable_makeup else 0.0

        st.markdown("**🎨 Studio Color & Optics**")
        lut_options = ["None", "Kodak Portra 400 (Warm Gold)", "Fuji Pro 400H (Pastel Jade)", "Teal & Orange / Cyberpunk", "Leica Monochrome (B&W)"]
        lut_idx = lut_options.index(default_lut) if default_lut in lut_options else 0
        color_lut_val = st.selectbox("Cinematic Film LUT", lut_options, index=lut_idx)
        lut_intensity = st.slider("LUT Intensity", 0.0, 1.0, 1.0, 0.05) if color_lut_val != "None" else 1.0
        bokeh_val = st.slider("📷 Studio Portrait Bokeh (f/1.4 Blur)", 0.0, 1.0, default_bokeh, 0.05, help="Simulate wide-aperture shallow depth of field background blur")

        chromatic_fix = st.toggle("🌈 Chromatic Aberration Correction", value=default_chromatic, help="Radial channel realignment for old lenses and color fringing")
        bg_upscale = st.toggle("Real-ESRGAN Background Upscale", value=False)
        face_upscale = st.toggle("Real-ESRGAN Face Upscale", value=False)

        st.markdown("---")
        st.markdown("**💾 Custom Presets Manager**")
        new_preset_name = st.text_input("New Preset Name", placeholder="e.g. Vintage Studio")
        if st.button("Save Current as Preset") and new_preset_name.strip():
            p_name = new_preset_name.strip()
            st.session_state.custom_presets[p_name] = {
                'w': w_val,
                'upscale': upscale_val,
                'detector': face_detector,
                'wink': wink_mode,
                'grain': skin_grain,
                'sharpen': sharpen_val,
                'color': color_match,
                'super_clarity': enable_super_clarity,
                'clarity_val': clarity_val,
                'deblur': enable_deblur,
                'deblur_val': deblur_val,
                'dehaze': enable_dehaze,
                'dehaze_val': dehaze_val,
                'crystal_skin': enable_crystal_skin,
                'crystal_skin_val': crystal_skin_val,
                'glossy_lips': enable_glossy_lips,
                'lip_gloss': lip_gloss_val,
                'lip_vibrance': lip_vibrance_val,
                'doll_eye': enable_doll_eye,
                'doll_eye_depth': doll_eye_val,
                'golden_hour': enable_golden_hour,
                'golden_warmth': golden_warmth_val,
                'eye': enable_eyes,
                'lip': enable_lips,
                'skin': enable_skin,
                'teeth': enable_teeth,
                'tone_glow': enable_tone_glow,
                'dark_circles': enable_dark_circles,
                'catchlight': enable_catchlight,
                'hair': enable_hair,
                'relighting': enable_relighting,
                'anti_glare': enable_anti_glare,
                'makeup': enable_makeup,
                'blush': blush_val,
                'eyebrow': eyebrow_val,
                'lut': color_lut_val,
                'lut_intensity': lut_intensity,
                'bokeh': bokeh_val,
                'chromatic': chromatic_fix
            }
            st.success(f"Saved custom preset: '{p_name}'")
            st.rerun()

        if st.session_state.custom_presets:
            preset_to_del = st.selectbox("Delete Preset", list(st.session_state.custom_presets.keys()))
            if st.button("Delete Selected Preset"):
                st.session_state.custom_presets.pop(preset_to_del, None)
                st.rerun()

# ── Main Header ─────────────────────────────────────────────────────────────────
st.markdown(f'''
<div style="display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; margin-bottom: 0.2rem;">
    <div class="hero-title" style="margin-bottom: 0;">AI Portrait Enhancer</div>
    <div style="background: rgba(99, 102, 241, 0.15); color: #818cf8; font-size: 0.82rem; padding: 4px 12px; border-radius: 9999px; font-weight: 600; border: 1px solid rgba(99, 102, 241, 0.35);">
        ✨ {APP_VERSION}
    </div>
</div>
''', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Restore blurry portraits, skin texture & eye detail with studio-level clarity</div>', unsafe_allow_html=True)

with st.expander("📈 Training Dashboard", expanded=False):
    render_training_dashboard()

tab_photo, tab_video, tab_batch, tab_benchmark = st.tabs([
    "📸 Portrait Enhancement",
    "🎥 Video AI Restoration",
    "📁 Batch Processing & Report",
    "📊 Benchmark & Quality Explorer"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: SINGLE PORTRAIT RESTORATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_photo:
    uploaded_file = st.file_uploader("Upload portrait photo (PNG, JPG, WEBP)", type=["png", "jpg", "jpeg", "webp"], key="photo_uploader")
    if uploaded_file is None:
        camera_file = st.camera_input("Or capture a live portrait photo with webcam")
        if camera_file is not None:
            uploaded_file = camera_file

    if uploaded_file is not None:
        try:
            input_img = load_uploaded_image(uploaded_file)
        except ValueError as error:
            st.error(str(error))
            st.stop()

        if pipeline is None:
            try:
                pipeline = get_pipeline()
            except Exception as error:
                st.error(f"Failed to initialize AI Pipeline: {error}")
                st.stop()

        current_params = {
            'img_name': getattr(uploaded_file, 'name', 'webcam_capture.png'),
            'w': w_val,
            'upscale': upscale_val,
            'detector': face_detector,
            'thresh': det_thresh,
            'wink': wink_mode,
            'grain': skin_grain,
            'sharpen': sharpen_val,
            'color': color_match,
            'super_clarity': enable_super_clarity,
            'clarity_val': clarity_val,
            'deblur': enable_deblur,
            'deblur_val': deblur_val,
            'dehaze': enable_dehaze,
            'dehaze_val': dehaze_val,
            'crystal_skin': enable_crystal_skin,
            'crystal_skin_val': crystal_skin_val,
            'glossy_lips': enable_glossy_lips,
            'lip_gloss': lip_gloss_val,
            'lip_vibrance': lip_vibrance_val,
            'doll_eye': enable_doll_eye,
            'doll_eye_depth': doll_eye_val,
            'golden_hour': enable_golden_hour,
            'golden_warmth': golden_warmth_val,
            'eye': enable_eyes,
            'lip': enable_lips,
            'skin': enable_skin,
            'teeth': enable_teeth,
            'tone_glow': enable_tone_glow,
            'dark_circles': enable_dark_circles,
            'catchlight': enable_catchlight,
            'hair': enable_hair,
            'relighting': enable_relighting,
            'anti_glare': enable_anti_glare,
            'makeup': enable_makeup,
            'blush': blush_val,
            'eyebrow': eyebrow_val,
            'lut': color_lut_val,
            'lut_intensity': lut_intensity,
            'bokeh': bokeh_val,
            'chromatic': chromatic_fix,
            'bg_up': bg_upscale,
            'face_up': face_upscale,
            'model_ver': selected_model_ver,
            'upscaler_choice': selected_upscaler
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

                chosen_upscaler_path = avail_upscalers.get(selected_upscaler)
                is_lanczos = chosen_upscaler_path == "lanczos"
                process_args = {
                    'w': w_val,
                    'detection_model': face_detector,
                    'upscale': upscale_val,
                    'blend_softness': 0.5,
                    'bg_upsampler': None if is_lanczos else ('realesrgan' if bg_upscale else None),
                    'bg_upsampler_model': None if is_lanczos else chosen_upscaler_path,
                    'det_threshold': det_thresh,
                    'sharpen_amount': sharpen_val,
                    'face_upsample': face_upscale,
                    'parallel': True,
                    'preset_mode': pipeline_preset_mode,
                    'model_version': selected_model_ver,
                    'wink_mode': wink_mode,
                    'eye_enhancement': enable_eyes,
                    'skin_grain': skin_grain,
                    'color_match': color_match,
                    'enable_super_clarity': enable_super_clarity,
                    'clarity_strength': clarity_val,
                    'enable_deblur': enable_deblur,
                    'deblur_strength': deblur_val,
                    'enable_dehaze': enable_dehaze,
                    'dehaze_strength': dehaze_val,
                    'enable_crystal_skin': enable_crystal_skin,
                    'crystal_skin_strength': crystal_skin_val,
                    'enable_glossy_lips': enable_glossy_lips,
                    'lip_gloss': lip_gloss_val,
                    'lip_vibrance': lip_vibrance_val,
                    'enable_doll_eye': enable_doll_eye,
                    'doll_eye_depth': doll_eye_val,
                    'enable_golden_hour': enable_golden_hour,
                    'golden_warmth': golden_warmth_val,
                    'golden_bloom': 0.20 if enable_golden_hour else 0.0,
                    'enable_eyes': enable_eyes,
                    'enable_lips': enable_lips,
                    'enable_skin': enable_skin,
                    'enable_teeth': enable_teeth,
                    'enable_tone_glow': enable_tone_glow,
                    'enable_dark_circles': enable_dark_circles,
                    'enable_catchlight': enable_catchlight,
                    'catchlight_strength': 0.55,
                    'enable_hair': enable_hair,
                    'hair_clarity': 0.35,
                    'hair_sheen': 0.25,
                    'enable_relighting': enable_relighting,
                    'relighting_rim': 0.25,
                    'relighting_tzone': 0.20,
                    'enable_anti_glare': enable_anti_glare,
                    'anti_glare_strength': 0.50,
                    'enable_makeup': enable_makeup,
                    'blush_strength': blush_val,
                    'eyebrow_boost': eyebrow_val,
                    'color_lut': color_lut_val,
                    'lut_intensity': lut_intensity,
                    'bokeh_strength': bokeh_val,
                    'chromatic_aberration': chromatic_fix,
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

            # Comparison Display (Interactive Split Slider, Side-by-Side, or 400% Zoom Loupe)
            view_col1, view_col2 = st.columns([1, 1])
            with view_col1:
                st.markdown("#### ✨ Visual Comparison")
            with view_col2:
                comparison_mode = st.radio(
                    "View Mode",
                    ["🎚️ Interactive Split Slider", "🔲 Side-by-Side", "🔍 400% Zoom Loupe"],
                    horizontal=True,
                    label_visibility="collapsed"
                )

            if comparison_mode == "🎚️ Interactive Split Slider" and pipeline and hasattr(pipeline, 'wink_enhancer'):
                st.caption("💡 **Hướng dẫn:** Bên Trái (🔴 Ảnh Gốc) ◀ ⬌ ▶ Bên Phải (✨ Ảnh AI Nâng Cấp). Nhấp hoặc kéo thanh trượt để so sánh độ nét!")
                slider_html = pipeline.wink_enhancer.generate_comparison_slider_html(input_img, enhanced_img, slider_id="portrait-split-slider")
                comp_height = int(min(850, max(420, (in_h / in_w) * 750 + 30))) if in_w > 0 else 550
                st.components.v1.html(slider_html, height=comp_height)
            elif comparison_mode == "🔍 400% Zoom Loupe" and pipeline and hasattr(pipeline, 'wink_enhancer'):
                st.caption("💡 **Hướng dẫn:** Rê chuột hoặc chạm vào ảnh để kích hoạt kính lúp soi chi tiết 400% song song cả 2 ảnh!")
                loupe_html = pipeline.wink_enhancer.generate_zoom_inspector_html(input_img, enhanced_img, widget_id="portrait-loupe-widget")
                comp_height = int(min(850, max(420, (in_h / in_w) * 750 + 60))) if in_w > 0 else 580
                st.components.v1.html(loupe_html, height=comp_height)
            else:
                c_orig, c_enh = st.columns(2)
                with c_orig:
                    st.markdown("##### 📷 Original Image")
                    st.image(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), use_column_width=True)

                with c_enh:
                    st.markdown("##### ✨ Wink Enhanced HD")
                    st.image(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB), use_column_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Download & Export Section
            d1, d2 = st.columns(2)
            with d1:
                success, encoded_buf = cv2.imencode('.png', enhanced_img)
                if success:
                    st.download_button(
                        label="⬇️ Download Enhanced HD Image (PNG)",
                        data=encoded_buf.tobytes(),
                        file_name=enhanced_filename(uploaded_file.name),
                        mime="image/png"
                    )
            with d2:
                if st.button("🎬 Generate Before/After Comparison GIF"):
                    with st.spinner("Generating smooth comparison animation..."):
                        gif_bytes = pipeline.wink_enhancer.create_comparison_animation(input_img, enhanced_img)
                        st.session_state.comparison_gif = gif_bytes
                        st.rerun()

                if st.session_state.get('comparison_gif'):
                    gif_name = f"comparison_{os.path.splitext(uploaded_file.name)[0]}.gif"
                    st.download_button(
                        label="⬇️ Download Comparison GIF Animation",
                        data=st.session_state.comparison_gif,
                        file_name=gif_name,
                        mime="image/gif"
                    )

    else:
        # Empty State Guide
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; border: 2px dashed rgba(255,255,255,0.1); border-radius: 20px; background: rgba(255,255,255,0.01);">
            <p style="font-size: 1.2rem; color: #94a3b8; font-weight: 600;">Drag and drop any portrait photo above to get started</p>
            <p style="font-size: 0.9rem; color: #64748b; margin-top: 8px;">Supports PNG, JPG, JPEG, WEBP. Optimized for fast CPU execution.</p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: VIDEO AI RESTORATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown("### 🎥 AI Video Portrait Restoration")
    st.markdown("Enhance facial detail and clarity in portrait videos frame-by-frame on CPU.")

    v_file = st.file_uploader("Upload video file (MP4, MOV, AVI)", type=["mp4", "mov", "avi"], key="video_uploader")
    if v_file is not None:
        v_col1, v_col2 = st.columns([1, 1])
        with v_col1:
            st.markdown("##### 📹 Original Video")
            st.video(v_file)

        v_stride = st.select_slider("Frame Sampling (Stride)", options=[1, 2, 3], value=1, help="1 = Restore every frame (highest quality). 2 = Restore every 2nd frame (2x faster on CPU).")
        max_f = st.number_input("Max Frames to Process (0 = Entire Video)", min_value=0, max_value=10000, value=60, step=30)

        if st.button("✨ Enhance Video"):
            import tempfile
            with st.spinner("Processing video frames with AI..."):
                t_dir = tempfile.mkdtemp()
                in_path = os.path.join(t_dir, v_file.name)
                with open(in_path, "wb") as f:
                    f.write(v_file.getvalue())

                out_path = os.path.join(t_dir, f"enhanced_{v_file.name}")
                if pipeline is None:
                    pipeline = get_pipeline()

                v_prog = st.progress(0.0)
                v_msg = st.empty()

                def v_callback(stage, pct, msg):
                    v_prog.progress(float(pct))
                    v_msg.info(f"✨ {msg}")

                v_stats = pipeline.process_video(
                    input_video_path=in_path,
                    output_video_path=out_path,
                    w=w_val,
                    detection_model=face_detector,
                    upscale=upscale_val,
                    frame_stride=v_stride,
                    max_frames=max_f if max_f > 0 else None,
                    progress_callback=v_callback
                )

                st.success(f"Video enhancement complete! Processed {v_stats['total_frames']} frames in {v_stats['duration_sec']:.2f}s ({v_stats['avg_fps']:.1f} FPS).")
                
                if os.path.exists(out_path):
                    with open(out_path, "rb") as vf:
                        v_bytes = vf.read()
                    st.download_button(
                        label="⬇️ Download Enhanced Video",
                        data=v_bytes,
                        file_name=f"enhanced_{v_file.name}",
                        mime="video/mp4"
                    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: BATCH PROCESSING & REPORT CARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### 📁 Batch Portrait Restoration & Quality Report Card")
    st.markdown("Upload multiple portraits for automatic batch enhancement with ZIP export and executive HTML scorecard.")

    batch_uploads = st.file_uploader(
        "Upload multiple portrait photos (up to 50)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if batch_uploads:
        st.info(f"Loaded {len(batch_uploads)} portrait(s) for batch enhancement.")
        if st.button("✨ Enhance All Portraits (Batch)"):
            if pipeline is None:
                pipeline = get_pipeline()

            batch_dict = {}
            for uf in batch_uploads:
                try:
                    img = load_uploaded_image(uf)
                    batch_dict[uf.name] = img
                except Exception as ex:
                    st.warning(f"Could not load {uf.name}: {ex}")

            if batch_dict:
                b_prog = st.progress(0.0)
                b_msg = st.empty()

                def b_callback(stage, pct, msg):
                    b_prog.progress(float(pct))
                    b_msg.info(f"✨ {msg}")

                batch_res = pipeline.process_batch_images(
                    batch_dict,
                    w=w_val,
                    detection_model=face_detector,
                    upscale=upscale_val,
                    blend_softness=0.5,
                    bg_upsampler='realesrgan' if bg_upscale else None,
                    det_threshold=det_thresh,
                    sharpen_amount=sharpen_val,
                    face_upsample=face_upscale,
                    parallel=True,
                    preset_mode=pipeline_preset_mode,
                    wink_mode=wink_mode,
                    eye_enhancement=enable_eyes,
                    skin_grain=skin_grain,
                    color_match=color_match,
                    enable_eyes=enable_eyes,
                    enable_lips=enable_lips,
                    enable_skin=enable_skin,
                    enable_teeth=enable_teeth,
                    enable_tone_glow=enable_tone_glow,
                    chromatic_aberration=chromatic_fix,
                    progress_callback=b_callback
                )

                st.session_state['last_batch_data'] = batch_res
                st.success(f"Batch completed! Enhanced {batch_res['total_count']} portraits in {batch_res['total_duration']:.2f}s ({batch_res['avg_time_per_image']:.2f}s/photo).")

    last_batch = st.session_state.get('last_batch_data')
    if last_batch:
        st.markdown("<hr>", unsafe_allow_html=True)
        import io
        import zipfile

        # Generate ZIP in-memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for name, item in last_batch["items"].items():
                _, encoded_img = cv2.imencode('.png', item["enhanced"])
                zip_file.writestr(f"enhanced_{name}", encoded_img.tobytes())

        # Generate HTML report
        html_report_str = pipeline.generate_html_report(last_batch) if pipeline else ""

        col_z1, col_z2 = st.columns(2)
        with col_z1:
            st.download_button(
                label="⬇️ Download All Enhanced Portraits (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="enhanced_portraits_batch.zip",
                mime="application/zip"
            )
        with col_z2:
            if html_report_str:
                st.download_button(
                    label="📄 Download Quality Scorecard Report (HTML)",
                    data=html_report_str.encode('utf-8'),
                    file_name="restoration_quality_report.html",
                    mime="text/html"
                )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: BENCHMARK & QUALITY EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_benchmark:
    st.markdown("### 📊 Benchmark & Restoration Quality Explorer")
    st.markdown("Explore quantitative quality scores, PSNR/SSIM metrics, and ArcFace facial identity similarity.")

    import json
    report_path = os.path.join(project_dir, "benchmarks", "baseline_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as rf:
            rep_data = json.load(rf)

        m_psnr = rep_data.get("mean_psnr", 0.0)
        m_ssim = rep_data.get("mean_ssim", 0.0)
        m_lpips = rep_data.get("mean_lpips", 0.0)
        m_arcface = rep_data.get("mean_arcface_similarity", 0.0)
        n_samples = rep_data.get("total_samples", 0)

        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            st.markdown(f'<div class="metric-badge"><div class="metric-label">Holdout Samples</div><div class="metric-val">{n_samples}</div></div>', unsafe_allow_html=True)
        with b2:
            st.markdown(f'<div class="metric-badge"><div class="metric-label">Mean PSNR</div><div class="metric-val" style="color: #34d399;">{m_psnr:.2f} dB</div></div>', unsafe_allow_html=True)
        with b3:
            st.markdown(f'<div class="metric-badge"><div class="metric-label">Mean SSIM</div><div class="metric-val" style="color: #60a5fa;">{m_ssim:.4f}</div></div>', unsafe_allow_html=True)
        with b4:
            st.markdown(f'<div class="metric-badge"><div class="metric-label">Mean LPIPS</div><div class="metric-val" style="color: #f472b6;">{m_lpips:.4f}</div></div>', unsafe_allow_html=True)
        with b5:
            st.markdown(f'<div class="metric-badge"><div class="metric-label">ArcFace Identity</div><div class="metric-val" style="color: #a78bfa;">{m_arcface:.4f}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if "category_breakdown" in rep_data:
            st.markdown("#### 📂 Metric Breakdown by Degradation Category")
            cat_list = []
            for c_name, c_metrics in rep_data["category_breakdown"].items():
                cat_list.append({
                    "Category": c_name,
                    "Count": c_metrics.get("count", 0),
                    "PSNR (dB)": round(c_metrics.get("mean_psnr", 0), 2),
                    "SSIM": round(c_metrics.get("mean_ssim", 0), 4),
                    "LPIPS": round(c_metrics.get("mean_lpips", 0), 4),
                    "ArcFace Sim": round(c_metrics.get("mean_arcface_similarity", 0), 4)
                })
            st.dataframe(cat_list, use_container_width=True)
    else:
        st.info("No saved benchmark baseline report found. Run `python tools/evaluate_restoration.py` to generate quantitative metrics on the 500-sample benchmark.")

