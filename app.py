import streamlit as st
import cv2
import numpy as np
import os
import time
from io import BytesIO
from pipeline import LocalAIEnhancerPipeline

project_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Page Configuration and Styling
st.set_page_config(
    page_title="Custom AI Face Enhancer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for premium dark aesthetics, glassmorphism, and custom typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Background Gradient */
    .stApp {
        background: radial-gradient(circle at top right, #1f1235 0%, #0d0b14 100%);
        color: #f1ecf7;
    }
    
    /* Header Gradient Text */
    .header-title {
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    
    .header-subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Cards and Glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0b090f !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Button Customization */
    div.stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
    }
    
    /* Download Button */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        width: 100%;
    }
    
    div.stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
    }
    
    /* File Uploader styling */
    section[data-testid="stFileUploader"] {
        border: 2px dashed rgba(167, 139, 250, 0.3);
        background: rgba(255, 255, 255, 0.01);
        border-radius: 12px;
        padding: 20px;
    }
    
    /* Stat Badge */
    .stat-badge {
        display: inline-block;
        padding: 6px 12px;
        background: rgba(124, 58, 237, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.2);
        color: #c084fc;
        border-radius: 8px;
        font-weight: 600;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Cache the Pipeline Initialization
@st.cache_resource(show_spinner=True)
def get_pipeline():
    """Initializes and caches the CodeFormer restoration pipeline."""
    try:
        return LocalAIEnhancerPipeline()
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None

pipeline = get_pipeline()

# 3. Sidebar Configuration
st.sidebar.markdown("<h2 style='text-align: center; color: #a78bfa;'>✨ AI Parameters</h2>", unsafe_allow_html=True)
st.sidebar.write("Configure details for the local AI models.")

# Advanced tuning sliders
fidelity_weight = st.sidebar.slider(
    "Fidelity Weight (w)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="0.0: Max quality (generates realistic details). 1.0: Max fidelity (closely resembles input)."
)

blend_softness = st.sidebar.slider(
    "Mask Blending Softness",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Control how smoothly the restored face is blended back onto the background. High values remove harsh crop boundaries."
)

face_detector = st.sidebar.selectbox(
    "Face Detector Model",
    options=["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],
    index=0,
    help="RetinaFace is highly accurate but slower. YOLOv5 is faster for smaller/crowded faces."
)

upscale_factor = st.sidebar.slider(
    "Background Upscale Factor",
    min_value=1,
    max_value=4,
    value=2,
    step=1,
    help="Scale factor for the output background image."
)

bg_upscale_toggle = st.sidebar.checkbox(
    "Real-ESRGAN Background Upscale",
    value=True,
    help="Enable AI-based super-resolution for the background using Real-ESRGAN. If unchecked, standard bilinear resizing is used."
)

det_threshold = st.sidebar.slider(
    "Face Detection Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Confidence threshold for face detection. Lower values find more/blurry faces, higher values reduce false detections."
)

sharpen_amount = st.sidebar.slider(
    "Post-Processing Sharpness",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    help="Enhance edge sharpness using a post-processing unsharp mask filter. 0.0 is disabled, higher values give an Ultra HD look."
)

# Tip card in sidebar
st.sidebar.markdown("""
<div style='margin-top: 20px; padding: 12px; background: rgba(167, 139, 250, 0.05); border-radius: 8px; border: 1px solid rgba(167, 139, 250, 0.15);'>
    <span style='color: #c084fc; font-weight: 600; font-size: 0.85rem;'>💡 KHÔNG THẤY KHÁC BIỆT?</span><br/>
    <span style='color: #9ca3af; font-size: 0.8rem; line-height: 1.4;'>
        - Hãy <b>giảm Fidelity Weight (w)</b> xuống (ví dụ: 0.3 - 0.5) để AI tự tạo thêm chi tiết nét hơn.<br/>
        - Đảm bảo <b>Face Detection Threshold</b> đủ thấp để AI nhận diện được mặt trong ảnh mờ.
    </span>
</div>
""", unsafe_allow_html=True)

# Display active hardware device
if pipeline is not None:
    device_str = "NVIDIA GPU (CUDA)" if "cuda" in str(pipeline.device) else "CPU"
    st.sidebar.markdown(f"""
    <div style='margin-top: 30px; padding: 12px; background: rgba(255, 255, 255, 0.05); border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);'>
        <span style='color: #9ca3af; font-size: 0.85rem;'>ACTIVE DEVICE:</span><br/>
        <strong style='color: #10b981; font-size: 0.95rem;'>🟢 {device_str}</strong>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style='margin-top: 30px; padding: 12px; background: rgba(255, 255, 255, 0.05); border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);'>
        <span style='color: #9ca3af; font-size: 0.85rem;'>ACTIVE DEVICE:</span><br/>
        <strong style='color: #ef4444; font-size: 0.95rem;'>🔴 Offline / Load Failed</strong>
    </div>
    """, unsafe_allow_html=True)

# 4. Main Page Header
st.markdown("<h1 class='header-title'>Custom AI Face Enhancer</h1>", unsafe_allow_html=True)
st.markdown("<p class='header-subtitle'>Self-Hosted Local AI Image Restoration Pipeline</p>", unsafe_allow_html=True)

# 5. Core Interface
col_upload, col_sample = st.columns([2, 1])

uploaded_file = None
use_sample = False

with col_upload:
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "webp"])

with col_sample:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("Use Sample Portrait Image"):
        use_sample = True

# Process sample image path if selected
sample_image_path = os.path.join(project_dir, "models", "CodeFormer", "inputs", "whole_imgs", "00.jpg")

# 6. Pipeline Execution
if uploaded_file is not None or use_sample:
    # Load input image
    if use_sample:
        if not os.path.exists(sample_image_path):
            st.error(f"Sample image not found at {sample_image_path}")
            st.stop()
        img = cv2.imread(sample_image_path)
        img_name = "sample_portrait.jpg"
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_name = uploaded_file.name

    if img is None:
        st.error("Failed to decode image. Please upload a valid image file.")
        st.stop()
        
    st.markdown("### ⚡ Enhancement Status")
    
    # Run pipeline with timing
    start_time = time.time()
    with st.spinner("Executing Local AI Restoration Pipeline..."):
        if pipeline is None:
            st.error("AI pipeline is not loaded. Cannot process image.")
            st.stop()
            
        # Process the image through the custom pipeline
        try:
            enhanced_img = pipeline.process_image(
                img,
                w=fidelity_weight,
                detection_model=face_detector,
                upscale=upscale_factor,
                blend_softness=blend_softness,
                bg_upsampler='realesrgan' if bg_upscale_toggle else None,
                det_threshold=det_threshold,
                sharpen_amount=sharpen_amount
            )
            process_duration = time.time() - start_time
        except Exception as e:
            st.error(f"An error occurred during pipeline execution: {e}")
            st.stop()
            
    # Display statistics
    h_orig, w_orig, _ = img.shape
    h_enh, w_enh, _ = enhanced_img.shape
    
    st.markdown(f"""
    <div style='margin-bottom: 20px;'>
        <span class='stat-badge'>⏱️ Time: {process_duration:.2f}s</span>
        <span class='stat-badge'>📐 Original Size: {w_orig}x{h_orig}</span>
        <span class='stat-badge'>🚀 Enhanced Size: {w_enh}x{h_enh}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # 7. Visual Comparison View Mode
    view_mode = st.radio(
        "Select Comparison Mode:",
        ["👥 Side-by-Side (2 Columns)", "🌗 Split-Screen (50/50 Split)"],
        horizontal=True
    )
    
    # Pre-render images
    h_enh, w_enh, _ = enhanced_img.shape
    img_resized = cv2.resize(img, (w_enh, h_enh), interpolation=cv2.INTER_LANCZOS4)
    mid_x = w_enh // 2
    
    if "Side-by-Side" in view_mode:
        col_before, col_after = st.columns(2)
        with col_before:
            st.markdown("<h4 style='text-align: center; color: #9ca3af;'>Before (Original)</h4>", unsafe_allow_html=True)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
            
        with col_after:
            st.markdown("<h4 style='text-align: center; color: #a78bfa;'>After (AI Restored)</h4>", unsafe_allow_html=True)
            enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
            st.image(enhanced_rgb, use_container_width=True)
    else:
        st.markdown("<h4 style='text-align: center; color: #a78bfa;'>🌗 Split-Screen Comparison (Left: Before | Right: After)</h4>", unsafe_allow_html=True)
        # Create split image
        split_img = np.copy(enhanced_img)
        split_img[:, :mid_x] = img_resized[:, :mid_x]
        # Draw vertical line separating them
        cv2.line(split_img, (mid_x, 0), (mid_x, h_enh), (255, 255, 255), max(2, w_enh // 300))
        
        split_rgb = cv2.cvtColor(split_img, cv2.COLOR_BGR2RGB)
        st.image(split_rgb, use_container_width=True)
        
    # 8. Download Results
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        is_success, buffer = cv2.imencode(".png", enhanced_img)
        if is_success:
            png_bytes = BytesIO(buffer).getvalue()
            st.download_button(
                label="📥 Download Enhanced Image Only",
                data=png_bytes,
                file_name=f"enhanced_{os.path.splitext(img_name)[0]}.png",
                mime="image/png",
                use_container_width=True
            )
            
    with col_dl2:
        # Create split comparison image for download
        split_img_dl = np.copy(enhanced_img)
        split_img_dl[:, :mid_x] = img_resized[:, :mid_x]
        cv2.line(split_img_dl, (mid_x, 0), (mid_x, h_enh), (255, 255, 255), max(2, w_enh // 300))
        
        is_success_split, buffer_split = cv2.imencode(".png", split_img_dl)
        if is_success_split:
            split_png_bytes = BytesIO(buffer_split).getvalue()
            st.download_button(
                label="🌗 Download Split Comparison Image",
                data=split_png_bytes,
                file_name=f"split_comparison_{os.path.splitext(img_name)[0]}.png",
                mime="image/png",
                use_container_width=True
            )
else:
    # 9. Clean Welcome/Intro layout
    st.markdown("""
    <div class='glass-card' style='margin-top: 40px;'>
        <h3 style='color: #a78bfa;'>How to Use:</h3>
        <ol style='color: #d1d5db; line-height: 1.8;'>
            <li>Upload a low-quality, blurry, or old portrait photo using the box above.</li>
            <li>Alternatively, click <strong>"Use Sample Portrait Image"</strong> to run a demo with a built-in portrait immediately.</li>
            <li>Use the sliders in the left sidebar to fine-tune the AI:
                <ul>
                    <li><strong>Fidelity Weight</strong> balances natural quality (hallucination) vs resemblance to the original.</li>
                    <li><strong>Mask Blending Softness</strong> controls how smoothly the cropped face borders dissolve into the background.</li>
                    <li><strong>Face Detector Model</strong> adapts detection speed and accuracy for single or multiple faces.</li>
                </ul>
            </li>
            <li>Once processed, review the side-by-side comparison and download the upscaled, crystal-clear PNG image.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
