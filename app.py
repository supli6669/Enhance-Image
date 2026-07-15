import streamlit as st
import cv2
import numpy as np
import os
import time
import threading
from io import BytesIO
from pipeline import LocalAIEnhancerPipeline

project_dir = os.path.dirname(os.path.abspath(__file__))

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Enhancer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }

.stApp {
    background:
        radial-gradient(ellipse 80% 60% at 20% -10%, rgba(120,40,255,0.22) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 110%, rgba(240,60,150,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 100% 80% at 50% 50%, #080612 0%, #06040f 100%);
    color: #ede9fe;
    min-height: 100vh;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0818 0%, #100d20 100%) !important;
    border-right: 1px solid rgba(139,92,246,0.15) !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 0; }

/* Mobile optimization */
@media (max-width: 768px) {
    section[data-testid="stSidebar"] {
        position: fixed !important;
        z-index: 999 !important;
        transform: translateX(-100%) !important;
        transition: transform 0.3s ease !important;
    }
    section[data-testid="stSidebar"].mobile-open {
        transform: translateX(0) !important;
    }
    .stApp {
        padding-left: 0 !important;
    }
    div.stButton > button {
        padding: 16px 20px !important;
        font-size: 1rem !important;
    }
    .sidebar-brand h2 {
        font-size: 0.9rem !important;
    }
    .hero-title {
        font-size: 2.2rem !important;
    }
    .hero-sub {
        font-size: 0.9rem !important;
    }
}

.sidebar-brand {
    background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%);
    padding: 20px 16px; margin: 0 -1rem 24px -1rem;
    text-align: center; border-bottom: 1px solid rgba(255,255,255,0.06);
}
.sidebar-brand h2 {
    margin: 0; font-size: 1.1rem; font-weight: 700;
    color: white; letter-spacing: 1px; text-transform: uppercase;
}
.sidebar-brand span { font-size: 1.6rem; }

.sidebar-section {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #6d28d9; margin: 20px 0 8px 0; padding-left: 2px;
}

.hero-wrap {
    position: relative; text-align: center;
    padding: 54px 20px 44px; margin-bottom: 8px; overflow: hidden;
}
.hero-glow {
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -60%); width: 600px; height: 300px;
    background: radial-gradient(ellipse, rgba(139,92,246,0.28) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    display: inline-block;
    background: rgba(139,92,246,0.12); border: 1px solid rgba(139,92,246,0.35);
    color: #c4b5fd; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase;
    padding: 5px 16px; border-radius: 100px; margin-bottom: 18px;
}
.hero-title {
    font-size: 3.6rem; font-weight: 900; line-height: 1.1;
    background: linear-gradient(135deg, #e9d5ff 0%, #a78bfa 35%, #f472b6 70%, #fb923c 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 14px; letter-spacing: -1px;
}
.hero-sub {
    font-size: 1.05rem; color: #7c6f9c; font-weight: 400;
    max-width: 520px; margin: 0 auto; line-height: 1.6;
}

section[data-testid="stFileUploader"] {
    background: rgba(139,92,246,0.03);
    border: 2px dashed rgba(139,92,246,0.3);
    border-radius: 16px; padding: 10px; transition: border-color 0.3s;
}
section[data-testid="stFileUploader"]:hover { border-color: rgba(139,92,246,0.55); }

div.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%);
    color: white !important; border: none; border-radius: 10px;
    padding: 12px 28px; font-weight: 700; font-size: 0.95rem;
    letter-spacing: 0.3px; transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35); width: 100%;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(124,58,237,0.5), 0 0 30px rgba(124,58,237,0.15);
}
div.stButton > button:active { transform: translateY(0); }

div.stDownloadButton > button {
    background: linear-gradient(135deg, #059669 0%, #0d9488 100%);
    color: white !important; border: none; border-radius: 10px;
    padding: 12px 28px; font-weight: 700; font-size: 0.92rem;
    transition: all 0.25s ease; box-shadow: 0 4px 20px rgba(5,150,105,0.3); width: 100%;
}
div.stDownloadButton > button:hover {
    transform: translateY(-2px); box-shadow: 0 8px 28px rgba(5,150,105,0.45);
}

.glass-card {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.065);
    border-radius: 20px; padding: 28px 32px; backdrop-filter: blur(16px);
    margin-bottom: 20px; position: relative; overflow: hidden;
}
.glass-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(167,139,250,0.4), transparent);
}

.stats-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 20px 0; }
.stat-pill {
    display: flex; align-items: center; gap: 8px; padding: 8px 16px;
    background: rgba(109,40,217,0.1); border: 1px solid rgba(109,40,217,0.25);
    border-radius: 100px; font-size: 0.85rem; font-weight: 600; color: #c4b5fd;
}

.section-header { display: flex; align-items: center; gap: 10px; margin: 28px 0 16px 0; }
.section-header .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: linear-gradient(135deg, #7c3aed, #db2777);
    box-shadow: 0 0 10px rgba(124,58,237,0.7); flex-shrink: 0;
}
.section-header h3 { font-size: 1.05rem; font-weight: 700; color: #ddd6fe; margin: 0; }

.img-label {
    text-align: center; padding: 6px 0 10px;
    font-size: 0.82rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
}
.img-label.before { color: #6b7280; }
.img-label.after  { color: #a78bfa; }

.step-list { counter-reset: steps; list-style: none; padding: 0; margin: 0; }
.step-list li {
    counter-increment: steps; display: flex; align-items: flex-start;
    gap: 14px; margin-bottom: 18px; color: #c4b5fd; line-height: 1.6; font-size: 0.95rem;
}
.step-list li::before {
    content: counter(steps); flex-shrink: 0; width: 28px; height: 28px;
    border-radius: 50%; background: linear-gradient(135deg, #7c3aed, #db2777);
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 0.8rem; color: white;
    box-shadow: 0 0 14px rgba(124,58,237,0.5);
}

.device-badge {
    display: flex; align-items: center; gap: 8px; padding: 10px 14px;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; margin-top: 24px;
}
.device-badge .label { font-size: 0.7rem; letter-spacing: 1.5px; text-transform: uppercase; color: #6b7280; }
.device-badge .value { font-size: 0.9rem; font-weight: 700; }
.device-badge .value.online  { color: #34d399; }
.device-badge .value.offline { color: #f87171; }

.tip-box {
    margin-top: 20px; padding: 14px 16px;
    background: rgba(167,139,250,0.05); border: 1px solid rgba(167,139,250,0.15);
    border-left: 3px solid #7c3aed; border-radius: 8px;
}
.tip-title { color: #c084fc; font-weight: 700; font-size: 0.78rem; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 6px; }
.tip-body  { color: #9ca3af; font-size: 0.82rem; line-height: 1.6; }

hr { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 24px 0; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Pipeline ───────────────────────────────────────────────────────────────────
# Progress state management
if 'progress_state' not in st.session_state:
    st.session_state.progress_state = {
        'stage': None,
        'progress': 0.0,
        'message': '',
        'active': False,
        'cancelled': False
    }

# Presets management
if 'presets' not in st.session_state:
    st.session_state.presets = {
        'Portrait': {'fidelity': 0.5, 'blend': 0.5, 'threshold': 0.5, 'upscale': 2, 'sharpen': 0.0},
        'Landscape': {'fidelity': 0.7, 'blend': 0.3, 'threshold': 0.6, 'upscale': 2, 'sharpen': 0.2},
        'Anime': {'fidelity': 0.3, 'blend': 0.4, 'threshold': 0.4, 'upscale': 2, 'sharpen': 0.0},
        'Vintage': {'fidelity': 0.6, 'blend': 0.6, 'threshold': 0.5, 'upscale': 2, 'sharpen': 0.1}
    }

# History management
if 'history' not in st.session_state:
    st.session_state.history = []

# Dark mode management
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

def apply_theme():
    """Apply dark/light theme based on user preference."""
    if not st.session_state.dark_mode:
        st.markdown("""
        <style>
        .stApp {
            background-color: #f5f5f5 !important;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.9) !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #e8e8e8 0%, #d0d0d0 100%) !important;
        }
        .sidebar-brand {
            background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%) !important;
        }
        .hero-title {
            color: #1a202c !important;
        }
        .hero-sub {
            color: #4a5568 !important;
        }
        </style>
        """, unsafe_allow_html=True)

def progress_callback(stage, progress, message):
    """Callback function to update progress state from pipeline."""
    if not st.session_state.progress_state.get('cancelled', False):
        st.session_state.progress_state = {
            'stage': stage,
            'progress': progress,
            'message': message,
            'active': True,
            'cancelled': False
        }

@st.cache_resource(show_spinner=False, hash_funcs={LocalAIEnhancerPipeline: lambda _: None})
def get_pipeline():
    try:
        return LocalAIEnhancerPipeline(progress_callback=progress_callback)
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None

pipeline = get_pipeline()

# Apply theme
apply_theme()

# Keyboard shortcuts
st.markdown("""
<script>
document.addEventListener('keydown', function(e) {
    // Ctrl+U: Focus on upload (simulated by scrolling to upload section)
    if (e.ctrlKey && e.key === 'u') {
        e.preventDefault();
        const uploadSection = document.querySelector('[data-testid="stFileUploader"]');
        if (uploadSection) {
            uploadSection.scrollIntoView({ behavior: 'smooth' });
        }
    }
    // Ctrl+R: Run processing (if image is loaded)
    if (e.ctrlKey && e.key === 'r') {
        e.preventDefault();
        const processButton = document.querySelector('button[kind="primary"]');
        if (processButton) {
            processButton.click();
        }
    }
    // Ctrl+S: Save settings (show toast)
    if (e.ctrlKey && e.key === 's') {
        e.preventDefault();
        // Settings are automatically saved to session state
        alert('Settings saved!');
    }
    // Esc: Cancel processing
    if (e.key === 'Escape') {
        const cancelButton = document.querySelector('button:contains("Cancel")');
        if (cancelButton) {
            cancelButton.click();
        }
    }
});
</script>
""", unsafe_allow_html=True)

def get_training_status():
    import re
    import datetime
    import glob
    exp_dir = os.path.join(project_dir, "models", "CodeFormer", "experiments")
    if not os.path.exists(exp_dir):
        return None
        
    dirs = [d for d in os.listdir(exp_dir) if d.endswith("_CodeFormer_stage3_custom")]
    if not dirs:
        return None
        
    latest_dir = sorted(dirs)[-1]
    log_pattern = os.path.join(exp_dir, latest_dir, "train_*.log")
    log_files = glob.glob(log_pattern)
    if not log_files:
        return None
        
    latest_log = log_files[0]
    try:
        with open(latest_log, "r", encoding="utf-8") as f:
            lines = f.readlines()[-30:]
            
        for line in reversed(lines):
            if "iter:" in line and "epoch:" in line:
                match = re.search(r"epoch:\s*(\d+),\s*iter:\s*([\d,]+)", line)
                eta_match = re.search(r"eta:\s*([\d:]+)", line)
                loss_match = re.search(r"cross_entropy_loss:\s*([\d.e+-]+)", line)
                
                if match:
                    epoch = int(match.group(1))
                    iteration = int(match.group(2).replace(",", ""))
                    eta = eta_match.group(1) if eta_match else "Unknown"
                    loss = float(loss_match.group(1)) if loss_match else 0.0
                    
                    # Extract timestamp at start of line: "2026-07-15 13:25:52"
                    ts_str = line.split(",")[0][:19]
                    try:
                        log_time = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                        now = datetime.datetime.now()
                        # If the last update was within 10 minutes, training is active
                        is_active = abs((now - log_time).total_seconds()) < 600
                    except Exception:
                        is_active = True
                        
                    return {
                        "active": is_active,
                        "epoch": epoch,
                        "iter": iteration,
                        "eta": eta,
                        "loss": loss,
                        "last_update": ts_str
                    }
    except Exception:
        pass
    return None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Dark mode toggle
    dark_mode_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="dark_mode_toggle")
    if dark_mode_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode_toggle
        st.rerun()
    
    st.markdown("""
    <div class="sidebar-brand">
        <span>✨</span>
        <h2>AI Enhancer</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Export/Import settings
    st.markdown("<div class='sidebar-section'>⚙️ Settings</div>", unsafe_allow_html=True)
    
    col_export, col_import = st.columns(2)
    with col_export:
        if st.button("📤 Export"):
            import json
            settings = {
                'fidelity_weight': st.session_state.get('fidelity_weight', 0.5),
                'blend_softness': st.session_state.get('blend_softness', 0.5),
                'det_threshold': st.session_state.get('det_threshold', 0.5),
                'upscale_factor': st.session_state.get('upscale_factor', 2),
                'sharpen_amount': st.session_state.get('sharpen_amount', 0.0),
                'dark_mode': st.session_state.dark_mode
            }
            st.download_button(
                "Download JSON",
                json.dumps(settings, indent=2),
                "ai_enhancer_settings.json",
                "application/json",
                key="download_settings"
            )
    
    with col_import:
        uploaded_settings = st.file_uploader("📥 Import", type=['json'], key="import_settings")
        if uploaded_settings:
            try:
                import json
                settings = json.loads(uploaded_settings.read())
                st.session_state.fidelity_weight = settings.get('fidelity_weight', 0.5)
                st.session_state.blend_softness = settings.get('blend_softness', 0.5)
                st.session_state.det_threshold = settings.get('det_threshold', 0.5)
                st.session_state.upscale_factor = settings.get('upscale_factor', 2)
                st.session_state.sharpen_amount = settings.get('sharpen_amount', 0.0)
                st.session_state.dark_mode = settings.get('dark_mode', True)
                st.success("Settings imported successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to import settings: {e}")

    # ── Training Dashboard ──────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'>📊 Training Dashboard</div>", unsafe_allow_html=True)
    status = get_training_status()
    if status:
        badge_color = "online" if status["active"] else "offline"
        badge_text = "🟢 Active" if status["active"] else "⚪ Idle"
        
        st.markdown(f"""
        <div class="glass-card" style="padding: 16px 20px; margin-bottom: 12px; border-radius: 12px; background: rgba(139,92,246,0.03);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:0.75rem; font-weight:700; color:#8b5cf6;">STATUS</span>
                <span class="value {badge_color}" style="font-size:0.85rem; font-weight:800;">{badge_text}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:0.75rem; color:#6b7280;">Iteration</span>
                <span style="font-size:0.8rem; font-weight:700; color:#ddd6fe;">{status['iter']} / 20000</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:0.75rem; color:#6b7280;">Epoch</span>
                <span style="font-size:0.8rem; font-weight:700; color:#ddd6fe;">{status['epoch']}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:0.75rem; color:#6b7280;">Loss</span>
                <span style="font-size:0.8rem; font-weight:700; color:#db2777;">{status['loss']:.4f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:0.75rem; color:#6b7280;">ETA</span>
                <span style="font-size:0.8rem; font-weight:700; color:#34d399;">{status['eta']}</span>
            </div>
            <div style="font-size:0.6rem; color:#4b5563; text-align:right; margin-top:8px;">
                Updated: {status['last_update']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No active training logs detected.")

    st.markdown("<div class='sidebar-section'>🎯 Face Restoration</div>", unsafe_allow_html=True)
    enable_face_restoration = st.toggle("Enable Face Restoration", value=True,
        help="Detect and restore faces. Disable if the original face is already sharp and you only want to upscale the background.")
    
    # Preset selection with help
    preset_cols = st.columns([2, 1])
    with preset_cols[0]:
        selected_preset = st.selectbox("Quick Preset", ["Custom", "Portrait", "Landscape", "Anime", "Vintage"],
            help="Pre-configured settings optimized for different image types")
    with preset_cols[1]:
        if st.button("Apply", key="apply_preset", help="Apply the selected preset to all parameters"):
            if selected_preset != "Custom":
                preset = st.session_state.presets[selected_preset]
                st.session_state.fidelity_weight = preset['fidelity']
                st.session_state.blend_softness = preset['blend']
                st.session_state.det_threshold = preset['threshold']
                st.session_state.upscale_factor = preset['upscale']
                st.session_state.sharpen_amount = preset['sharpen']
                st.success(f"Applied {selected_preset} preset!")
                st.rerun()
    
    fidelity_weight = st.slider("Fidelity Weight (w)", 0.0, 1.0, st.session_state.get('fidelity_weight', 0.3), 0.05,
        help="0.0 = Max AI detail (more enhancement, less original likeness). 1.0 = Max fidelity (keeps original face structure). Recommended: 0.3-0.7 for portraits.", key="fidelity_weight")
    blend_softness  = st.slider("Mask Blending Softness", 0.0, 1.0, st.session_state.get('blend_softness', 0.5), 0.05,
        help="Controls the edge softness when blending restored faces. Higher = smoother, more natural blend. Lower = sharper, more noticeable transition.", key="blend_softness")
    det_threshold   = st.slider("Detection Threshold", 0.1, 1.0, st.session_state.get('det_threshold', 0.5), 0.05,
        help="Face detection confidence threshold. Lower values detect more faces including blurry ones. Higher values reduce false detections on non-face objects.", key="det_threshold")
    face_upscale_toggle = st.toggle("Real-ESRGAN Face Upscale", value=False,
        help="Use Real-ESRGAN to upscale the restored face. ⚠️ Extremely slow on CPU (10x slower). Keep disabled for virtually identical quality with much faster processing.")

    st.markdown("<div class='sidebar-section'>🖼️ Background Upscaling</div>", unsafe_allow_html=True)
    face_detector    = st.selectbox("Face Detector",
        ["retinaface_mobile0.25", "retinaface_resnet50", "YOLOv5l", "YOLOv5n"],
        help="MobileNet = extremely fast (recommended for CPU). RetinaFace = most accurate (slower). YOLOv5 = good balance of speed and accuracy.")
    upscale_factor   = st.select_slider("Upscale Factor", [1, 2, 3, 4], value=st.session_state.get('upscale_factor', 2),
        help="Output resolution multiplier. 2× = double resolution. Higher values increase processing time exponentially.")
    bg_upscale_toggle = st.toggle("Real-ESRGAN Background Upscale", value=False,
        help="Upscale the entire image background (not just faces). ⚠️ Slow on CPU. Use only if you need to upscale the whole image.")

    st.markdown("<div class='sidebar-section'>🔬 Post-Processing</div>", unsafe_allow_html=True)
    sharpen_amount = st.slider("Sharpness Boost", 0.0, 1.0, st.session_state.get('sharpen_amount', 0.0), 0.05,
        help="Unsharp mask filter to enhance edge sharpness. 0.0 = disabled. Use sparingly (0.1-0.3) to avoid over-sharpening artifacts.", key="sharpen_amount")

    st.markdown("""
    <div class="tip-box">
        <div class="tip-title">💡 Pro Tip</div>
        <div class="tip-body">
            Lower <b>Fidelity Weight</b> (0.3–0.5) for richer AI-generated detail.<br>
            Lower <b>Detection Threshold</b> if faces in blurry images go undetected.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if pipeline is not None:
        dstr = "NVIDIA GPU (CUDA)" if "cuda" in str(pipeline.device) else "CPU"
        st.markdown(f"""
        <div class="device-badge">
            <div><div class="label">Active Device</div>
            <div class="value online">🟢 {dstr}</div></div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="device-badge">
            <div><div class="label">Active Device</div>
            <div class="value offline">🔴 Offline / Load Failed</div></div>
        </div>""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
# Mobile menu toggle
if st.button("☰ Menu", key="mobile_menu"):
    st.markdown("""
    <script>
    document.querySelector('[data-testid="stSidebar"]').classList.toggle('mobile-open');
    </script>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="hero-wrap">
    <div class="hero-glow"></div>
    <div class="hero-tag">✦ Self-Hosted · Local AI · Zero Cloud</div>
    <div class="hero-title">AI Image Enhancer</div>
    <div class="hero-sub">
        Restore blurry portraits and upscale images with CodeFormer + Real-ESRGAN —
        running entirely on your machine.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Upload & Tabs ──────────────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["✨ Single Image", "📦 Batch Processing"])

with tab_single:
    col_upload, col_sample = st.columns([3, 1], gap="large")
    uploaded_file = None
    use_sample    = False

    with col_upload:
        uploaded_file = st.file_uploader(
            "Drop your image here, or click to browse",
            type=["png", "jpg", "jpeg", "webp"],
            key="single_uploader"
        )
    with col_sample:
        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
        if st.button("🖼️ Use Sample Portrait"):
            use_sample = True

    sample_image_path = os.path.join(project_dir, "models", "CodeFormer", "inputs", "whole_imgs", "00.jpg")

    # ── Pipeline Execution (Single) ─────────────────────────────────────────────
    if uploaded_file is not None or use_sample:
        if use_sample:
            if not os.path.exists(sample_image_path):
                st.error(f"Sample image not found: `{sample_image_path}`")
                st.stop()
            img      = cv2.imread(sample_image_path)
            img_name = "sample_portrait.jpg"
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img        = cv2.imdecode(file_bytes, 1)
            img_name   = uploaded_file.name

        if img is None:
            st.error("Could not decode image. Please upload a valid PNG / JPG / WEBP file.")
            st.stop()

        st.markdown("<hr>", unsafe_allow_html=True)

        start_time = time.time()
        # Async processing using a background thread
        if not st.session_state.get('processing'):
            # Initialize state
            st.session_state.processing = True
            st.session_state.progress_state = {'stage': None, 'progress': 0.0, 'message': '', 'active': False}
            
            # Progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            stage_text = st.empty()
            
            # Cancel button
            cancel_col, _ = st.columns([1, 3])
            with cancel_col:
                cancel_button = st.button("❌ Cancel", type="secondary")
            
            if cancel_button:
                st.session_state.progress_state['cancelled'] = True
                st.warning("⚠️ Processing cancelled by user")
                st.stop()
            
            result_container = {}
            def _run():
                max_retries = 2
                retry_count = 0
                last_error = None
                
                while retry_count <= max_retries:
                    try:
                        result = pipeline.process_image(
                            img,
                            w=fidelity_weight,
                            detection_model=face_detector,
                            upscale=upscale_factor,
                            blend_softness=blend_softness,
                            bg_upsampler='realesrgan' if bg_upscale_toggle else None,
                            det_threshold=det_threshold,
                            sharpen_amount=sharpen_amount,
                            face_upsample=face_upscale_toggle,
                            parallel=True,
                            batch_size=4,
                            face_restore=enable_face_restoration
                        )
                        result_container['enhanced_img'] = result
                        break
                    except Exception as e:
                        last_error = e
                        retry_count += 1
                        if retry_count <= max_retries:
                            st.session_state.progress_state['message'] = f"Retry {retry_count}/{max_retries}: {str(e)[:50]}..."
                            time.sleep(1)
                        else:
                            result_container['error'] = str(e)
                finally:
                    st.session_state.processing = False
                    st.session_state.progress_state['active'] = False
            
            threading.Thread(target=_run, daemon=True).start()
            
            # Wait for result (polling with progress updates)
            stage_names = {
                'initialization': '🔧 Initializing',
                'detection': '👁️ Detecting Faces',
                'background': '🖼️ Upscaling Background',
                'restoration': '✨ Restoring Faces',
                'blending': '🎨 Blending Faces',
                'complete': '✅ Complete'
            }
            
            while st.session_state.processing:
                time.sleep(0.1)
                progress_data = st.session_state.progress_state
                
                # Check for cancellation
                if progress_data.get('cancelled', False):
                    st.session_state.processing = False
                    st.warning("⚠️ Processing cancelled by user")
                    progress_bar.empty()
                    status_text.empty()
                    stage_text.empty()
                    st.stop()
                
                if progress_data['active']:
                    stage_name = stage_names.get(progress_data['stage'], progress_data['stage'])
                    stage_text.text(f"**{stage_name}**")
                    status_text.text(progress_data['message'])
                    progress_bar.progress(progress_data['progress'])
                else:
                    status_text.text('⏳ Starting...')
            
            enhanced_img = result_container.get('enhanced_img')
            process_duration = time.time() - start_time
            
            # Clear progress UI
            progress_bar.empty()
            status_text.empty()
            stage_text.empty()
            
            # Check for errors
            if 'error' in result_container:
                st.error(f"❌ Processing failed after retries: {result_container['error']}")
                st.info("💡 Try reducing the upscale factor or disabling some features.")
                st.stop()
            
            # Save to history
            history_item = {
                'name': img_name,
                'timestamp': time.time(),
                'params': {
                    'fidelity': fidelity_weight,
                    'blend': blend_softness,
                    'threshold': det_threshold,
                    'upscale': upscale_factor,
                    'sharpen': sharpen_amount
                },
                'original_shape': img.shape[:2],
                'enhanced_shape': enhanced_img.shape[:2]
            }
            st.session_state.history.insert(0, history_item)
            # Keep only last 10 items
            if len(st.session_state.history) > 10:
                st.session_state.history = st.session_state.history[:10]
        else:
            st.warning('Processing is already running. Please wait.')
            st.stop()

        h_orig, w_orig = img.shape[:2]
        h_enh,  w_enh  = enhanced_img.shape[:2]
        mid_x          = w_enh // 2
        img_resized    = cv2.resize(img, (w_enh, h_enh), interpolation=cv2.INTER_LANCZOS4)

        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-pill">⏱️ {process_duration:.2f}s</div>
            <div class="stat-pill">📐 Original: {w_orig}×{h_orig}</div>
            <div class="stat-pill">🚀 Enhanced: {w_enh}×{h_enh}</div>
            <div class="stat-pill">⬆️ {upscale_factor}× upscale</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-header">
            <div class="dot"></div><h3>Comparison View</h3>
        </div>""", unsafe_allow_html=True)

        view_mode = st.radio("Mode", ["↔️ Interactive Slider", "👥 Side-by-Side", "🌗 Split Screen"],
                             horizontal=True, label_visibility="collapsed")

        if "Interactive Slider" in view_mode:
            from streamlit_image_comparison import image_comparison
            image_comparison(
                img1=cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB),
                img2=cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB),
                label1="Original",
                label2="AI Enhanced",
                show_labels=True,
                make_responsive=True
            )
        elif "Side-by-Side" in view_mode:
            col_b, col_a = st.columns(2, gap="medium")
            with col_b:
                st.markdown("<div class='img-label before'>◀ Before (Original)</div>", unsafe_allow_html=True)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_a:
                st.markdown("<div class='img-label after'>After (AI Enhanced) ▶</div>", unsafe_allow_html=True)
                st.image(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.markdown("<div class='img-label after'>◀ Original &nbsp;|&nbsp; Enhanced ▶</div>",
                        unsafe_allow_html=True)
            split_img = np.copy(enhanced_img)
            split_img[:, :mid_x] = img_resized[:, :mid_x]
            cv2.line(split_img, (mid_x, 0), (mid_x, h_enh), (255, 255, 255), max(2, w_enh // 300))
            st.image(cv2.cvtColor(split_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("""
        <div class="section-header">
            <div class="dot"></div><h3>Download Results</h3>
        </div>""", unsafe_allow_html=True)

        col_dl1, col_dl2 = st.columns(2, gap="medium")
        with col_dl1:
            ok, buf = cv2.imencode(".png", enhanced_img)
            if ok:
                st.download_button("📥 Download Enhanced Image",
                    BytesIO(buf).getvalue(),
                    f"enhanced_{os.path.splitext(img_name)[0]}.png",
                    "image/png", use_container_width=True)
        with col_dl2:
            split_img = np.copy(enhanced_img)
            split_img[:, :mid_x] = img_resized[:, :mid_x]
            cv2.line(split_img, (mid_x, 0), (mid_x, h_enh), (255, 255, 255), max(2, w_enh // 300))
            ok2, buf2 = cv2.imencode(".png", split_img)
            if ok2:
                st.download_button("🌗 Download Split Comparison",
                    BytesIO(buf2).getvalue(),
                    f"comparison_{os.path.splitext(img_name)[0]}.png",
                    "image/png", use_container_width=True)

        # History section
        if st.session_state.history:
            st.markdown("""
            <div class="section-header">
                <div class="dot"></div><h3>Recent History</h3>
            </div>""", unsafe_allow_html=True)
            
            for idx, item in enumerate(st.session_state.history[:5]):  # Show last 5
                with st.expander(f"📸 {item['name']} - {time.strftime('%H:%M:%S', time.localtime(item['timestamp']))}"):
                    cols = st.columns(4)
                    cols[0].metric("Fidelity", f"{item['params']['fidelity']:.2f}")
                    cols[1].metric("Blend", f"{item['params']['blend']:.2f}")
                    cols[2].metric("Upscale", f"{item['params']['upscale']}×")
                    cols[3].metric("Sharpen", f"{item['params']['sharpen']:.2f}")
                    
                    if st.button(f"🔄 Restore Settings", key=f"restore_{idx}"):
                        st.session_state.fidelity_weight = item['params']['fidelity']
                        st.session_state.blend_softness = item['params']['blend']
                        st.session_state.det_threshold = item['params']['threshold']
                        st.session_state.upscale_factor = item['params']['upscale']
                        st.session_state.sharpen_amount = item['params']['sharpen']
                        st.success("Settings restored!")
                        st.rerun()

    else:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <div class="section-header" style="margin-top:0">
                <div class="dot"></div><h3>How to Get Started</h3>
            </div>
            <ol class="step-list">
                <li><span>Upload a low-quality, blurry, or old portrait photo above. Supports PNG, JPG, and WEBP.</span></li>
                <li><span>Or click <b>"Use Sample Portrait"</b> to run the pipeline on a built-in demo — no upload needed.</span></li>
                <li><span>Fine-tune the AI with sidebar sliders. <b>Fidelity Weight</b> controls creativity vs. likeness. <b>Mask Softness</b> controls face blending.</span></li>
                <li><span>Review the before/after comparison and download your crystal-clear PNG result.</span></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

with tab_batch:
    uploaded_files = st.file_uploader(
        "Upload multiple images to process",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="batch_uploader"
    )
    if uploaded_files:
        if st.button("🚀 Process Batch", key="batch_button"):
            import zipfile
            from io import BytesIO
            
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                # Progress bar and status
                batch_progress = st.progress(0)
                status_text = st.empty()
                
                total_files = len(uploaded_files)
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing ({idx+1}/{total_files}): {file.name}")
                    
                    # decode image
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    img_b = cv2.imdecode(file_bytes, 1)
                    
                    if img_b is None:
                        st.warning(f"Could not decode {file.name}, skipping.")
                        continue
                        
                    # run pipeline
                    result = pipeline.process_image(
                        img_b,
                        w=fidelity_weight,
                        detection_model=face_detector,
                        upscale=upscale_factor,
                        blend_softness=blend_softness,
                        bg_upsampler='realesrgan' if bg_upscale_toggle else None,
                        det_threshold=det_threshold,
                        sharpen_amount=sharpen_amount,
                        face_upsample=face_upscale_toggle,
                        parallel=True,
                        batch_size=4,
                        face_restore=enable_face_restoration
                    )
                    
                    # encode result as png
                    ok, buf = cv2.imencode(".png", result)
                    if ok:
                        # add to zip
                        zip_file.writestr(f"enhanced_{file.name}", BytesIO(buf).getvalue())
                        
                    batch_progress.progress((idx + 1) / total_files)
                    
                status_text.text("✅ Batch processing complete!")
                
            st.download_button(
                "📥 Download Enhanced Zip",
                zip_buffer.getvalue(),
                "enhanced_images.zip",
                "application/zip",
                use_container_width=True
            )

st.markdown("""
<div style="display:flex; gap:12px; flex-wrap:wrap; justify-content:center; margin-top:8px;">
    <div class="stat-pill">🤖 CodeFormer Face Restoration</div>
    <div class="stat-pill">🖼️ Real-ESRGAN Super-Resolution</div>
    <div class="stat-pill">🔒 100% Local · No Cloud</div>
    <div class="stat-pill">📦 Custom-Trained Weights</div>
</div>
""", unsafe_allow_html=True)
