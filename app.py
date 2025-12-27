import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import cv2

# --- Page Config ---
st.set_page_config(
    page_title="SkinAI - Dermatology Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Modern Design System ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        * {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #ffffff;
        }

        .stApp {
            background: radial-gradient(circle at 0% 0%, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        }

        /* Hero Section */
        .hero-container {
            text-align: center;
            padding: 4rem 2rem;
            background: rgba(255, 255, 255, 0.03);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 3rem;
        }

        .hero-title {
            font-size: 4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            letter-spacing: -2px;
        }

        .hero-subtitle {
            font-size: 1.25rem;
            color: #a0aec0 !important;
            max-width: 800px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* Applying the glass effect directly to column children */
        [data-testid="stColumn"] > div {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 32px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            margin-bottom: 2rem;
        }

        .section-header {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            gap: 12px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        /* File Uploader visibility improvement */
        .stFileUploader section {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 2px dashed rgba(255, 255, 255, 0.2) !important;
            border-radius: 20px !important;
            padding: 2rem !important;
        }
        
        /* 'Drag and drop' & 'Limit' text visibility */
        .stFileUploader [data-testid='stFileUploaderText'],
        .stFileUploader [data-testid='stFileUploaderDropzoneInstructions'] {
            color: #ffffff !important;
            opacity: 1 !important;
            font-weight: 500 !important;
        }
        
        .stFileUploader [data-testid='stFileUploaderText']::before {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        /* File Name and Size Visibility */
        .stFileUploader [data-testid="stFileUploaderFileName"], 
        .stFileUploader [data-testid="stFileUploaderFileData"],
        .stFileUploader span {
            color: #ffffff !important;
            opacity: 1 !important;
            font-weight: 600 !important;
        }
        
        /* 'Browse files' button text visibility fix */
        .stFileUploader button {
            background: #ffffff !important;
            color: #0f172a !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            border: none !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.2s ease !important;
        }

        /* Modern Button */
        div.stButton > button {
            background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
            color: white !important;
            border: none;
            padding: 1rem 2rem;
            border-radius: 16px;
            font-weight: 700;
            font-size: 1.2rem;
            width: 100%;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 10px 20px -5px rgba(58, 123, 213, 0.4);
            margin-top: 1rem;
        }

        footer {
            margin-top: 6rem;
            padding: 4rem 2rem;
            text-align: center;
            background: rgba(0, 0, 0, 0.2);
            color: #718096 !important;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">SkinAI Diagnostics</h1>
        <p class="hero-subtitle">Utilizing neural networks to distinguish between skin lesions and documents. Fast, secure, and precise.</p>
    </div>
""", unsafe_allow_html=True)

# --- Load Model ---
MODEL_PATH = "skin_cancer_model.h5"

@st.cache_resource
def load_skin_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
    return None

model = load_skin_model()

def is_valid_skin_image(image):
    """
    Improved heuristic to distinguish skin dermoscopy from documents/text.
    Dermoscopy (ISIC) is often zoom-heavy, colorful, and has low high-frequency text patterns.
    """
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge/Text Detection (Documents have sharp, dense high-frequency edges)
    edges = cv2.Canny(gray, 75, 200)
    edge_density = np.sum(edges > 0) / edges.size
    
    # 2. Laplacian Variance (Focus/Texture check)
    # Natural skin has a specific variance. Text documents have extremely high localized variance.
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 3. Pure White Background (Document rejection)
    # Scale to 0-1 and check percentage of pure white pixels
    white_mask = gray > 245
    white_ratio = np.sum(white_mask) / gray.size

    # REFINED LOGIC: 
    # If it has VERY high edge density AND high localized variance, it's text.
    # If it's mostly a white sheet of paper (>60%), it's a document.
    
    if white_ratio > 0.70:
        return False, "Image appears to be a white document or empty backdrop."
    if edge_density > 0.25 and var > 500:
        return False, "High text-like density detected. Please upload an actual skin photo."
        
    return True, "Valid"

# --- Main Layout ---
c1, c2 = st.columns([1, 1], gap="large")

with c1:
    st.markdown('<div class="section-header">📷 Image Acquisition</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a dermoscopic lesion image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Specimen', use_container_width=True)
    else:
        st.markdown('<div style="text-align:center; padding: 4rem 0; opacity: 0.4;">🧬<p>Waiting for image...</p></div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="section-header">🧠 AI Intelligence</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None and model is not None:
        if st.button("Perform Diagnostic Scan"):
            valid, msg = is_valid_skin_image(image)
            
            if not valid:
                st.error("❌ Upload skin pic")
                st.info(f"System Check: {msg}")
            else:
                with st.spinner("Analyzing neural features..."):
                    try:
                        img = image.resize((224, 224))
                        img_array = np.array(img)
                        if img_array.shape[-1] == 4: img_array = img_array[..., :3]
                        
                        img_array = np.expand_dims(img_array, axis=0)
                        processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.astype(np.float32))
                        
                        preds = model.predict(processed_img)
                        score = preds[0]
                        idx = np.argmax(score) if len(score) > 1 else (1 if score[0] > 0.5 else 0)
                        res = ["Benign", "Malignant"][idx]
                        conf = (score[idx] if len(score) > 1 else (score[0] if score[0] > 0.5 else 1-score[0])) * 100
                        
                        st.markdown(f'<div style="text-align:center; padding:1rem; border:2px solid {"#00ff88" if res=="Benign" else "#ff4b2b"}; border-radius:16px;">'
                                    f'<h2 style="color:{"#00ff88" if res=="Benign" else "#ff4b2b"} !important;">{res.upper()}</h2>'
                                    f'<p style="font-size:1.2rem;">Confidence: {conf:.1f}%</p></div>', unsafe_allow_html=True)
                        st.progress(float(conf/100))
                        
                        st.markdown('<div style="background:rgba(255,255,255,0.05); padding:1rem; border-radius:10px; margin-top:1rem;">'
                                    '<strong>Medical Notice:</strong> This AI assists in screening. '
                                    'Consult a dermatologist for clinical diagnosis.</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Analysis Error: {e}")
    else:
        st.markdown('<div style="text-align:center; padding: 4rem 0; opacity: 0.4;"><p>Upload image to scan.</p></div>', unsafe_allow_html=True)

st.markdown("""
    <footer>
        <p style="color: #718096 !important;">© 2025 SkinAI Neural Solutions | Premium Diagnostic Vision</p>
    </footer>
""", unsafe_allow_html=True)
