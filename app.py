import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

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

        /* File Name and Size Visibility (CRITICAL FIX) */
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
            color: #0f172a !important; /* Dark text on light button */
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

        div.stButton > button:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 30px -10px rgba(58, 123, 213, 0.6);
            filter: brightness(1.1);
        }

        /* Results Display */
        .result-container {
            text-align: center;
            padding: 1rem;
        }

        .status-badge {
            display: inline-block;
            padding: 0.75rem 2rem;
            border-radius: 16px;
            font-weight: 800;
            font-size: 2rem;
            margin-bottom: 1.5rem;
            letter-spacing: 1px;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        .benign-badge {
            background: rgba(0, 255, 136, 0.15);
            color: #00ff88;
            border: 2px solid #00ff88;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
        }

        .malignant-badge {
            background: rgba(255, 75, 43, 0.15);
            color: #ff4b2b;
            border: 2px solid #ff4b2b;
            box-shadow: 0 0 20px rgba(255, 75, 43, 0.2);
        }

        .confidence-label {
            color: #a0aec0 !important;
            font-size: 1.1rem;
            font-weight: 500;
            margin-top: 2rem;
        }

        .confidence-value {
            color: #ffffff;
            font-size: 3.5rem;
            font-weight: 800;
            margin: 0.5rem 0;
        }

        .info-box {
            background: rgba(58, 123, 213, 0.12);
            border-radius: 16px;
            padding: 1.5rem;
            border-left: 6px solid #3a7bd5;
            margin-top: 2rem;
            font-size: 1rem;
            line-height: 1.5;
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
        <p class="hero-subtitle">Utilizing state-of-the-art neural networks to assist in early detection and classification of skin lesions. Professional analysis in seconds.</p>
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

# --- Main Columns ---
c1, c2 = st.columns([1, 1], gap="large")

with c1:
    st.markdown('<div class="section-header">📷 Image Acquisition</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a dermoscopic lesion image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Specimen', use_container_width=True)
    else:
        st.markdown("""
            <div style="text-align:center; padding: 4rem 0; opacity: 0.4;">
                <div style="font-size: 5rem; margin-bottom: 1rem;">🧬</div>
                <p>Waiting for image input...</p>
            </div>
        """, unsafe_allow_html=True)

with c2:
    st.markdown('<div class="section-header">🧠 AI Intelligence</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None and model is not None:
        if st.button("Perform Diagnostic Scan"):
            with st.spinner("Analyzing spectral patterns..."):
                try:
                    # Preprocess - Using 224x224 as expected by the model
                    img = image.resize((224, 224))
                    img_array = np.array(img)
                    if img_array.shape[-1] == 4:
                         img_array = img_array[..., :3]
                    
                    img_array = np.expand_dims(img_array, axis=0)
                    # MobileNetV2 preprocess (scaling pixels to [-1, 1])
                    processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.astype(np.float32))
                    
                    # Inference
                    preds = model.predict(processed_img)
                    score = preds[0]
                    
                    # Logic for classification result
                    if len(score) == 1: # Sigmoid
                        val = score[0]
                        res = "Malignant" if val > 0.5 else "Benign"
                        conf = val * 100 if val > 0.5 else (1 - val) * 100
                    else: # Softmax
                        idx = np.argmax(score)
                        classes = ["Benign", "Malignant"]
                        res = classes[idx]
                        conf = score[idx] * 100
                    
                    # Visual Output
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    badge = "benign-badge" if res == "Benign" else "malignant-badge"
                    st.markdown(f'<div class="status-badge {badge}">{res.upper()}</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="confidence-label">AI CONFIDENCE</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence-value">{conf:.1f}%</div>', unsafe_allow_html=True)
                    st.progress(float(conf/100))
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("""
                        <div class="info-box">
                            <strong>Medical Notice:</strong> This analysis is powered by deep learning and is for 
                            informational purposes only. It should not replace a professional diagnosis. 
                            If Malignant is indicated, or if you have concerns, please visit a doctor immediately.
                        </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
    elif model is None:
        st.error("System Failure: Model weights not initialized.")
    else:
        st.markdown("""
            <div style="text-align:center; padding: 4rem 0; opacity: 0.4;">
                <p>Complete acquisition step to unlock analysis results.</p>
            </div>
        """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <footer>
        <p style="color: #718096 !important;">© 2025 SkinAI Neural Solutions | Advanced Healthcare Vision</p>
        <p style="opacity: 0.6; margin-top: 1rem; color: #718096 !important;">Optimized for Dermoscopic Imagery • High Precision Diagnostics</p>
    </footer>
""", unsafe_allow_html=True)
