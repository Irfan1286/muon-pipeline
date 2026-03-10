import streamlit as st  # type: ignore
import pandas as pd
import time
import os
import pickle
import sys

# Ensure we can import from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.feature_extractor import extract_all_features    # type: ignore

# --- Setup Dashboard Configuration ---
st.set_page_config(
    page_title="Muon Physics Pipeline - Main",
    page_icon="⚛️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #09090E, #1A1A2E); color: #E2E8F0; }
    h1, h2, h3 { color: #4F46E5 !important; font-family: 'Inter', sans-serif; font-weight: 700; }
    .main-title { background: -webkit-linear-gradient(45deg, #4F46E5, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem !important; padding-bottom: 20px; text-align: center; }
    .glass-container { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 24px; margin-bottom: 24px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); text-align: center; }
    .stFileUploader > div > div { background: rgba(79, 70, 229, 0.05); border: 1px dashed rgba(79, 70, 229, 0.5); border-radius: 12px; transition: all 0.3s ease; }
    .stFileUploader > div > div:hover { background: rgba(79, 70, 229, 0.1); border-color: #4F46E5; }
    .stButton > button { background: linear-gradient(90deg, #4F46E5, #7C3AED); color: white; border: none; border-radius: 8px; padding: 10px 24px; font-weight: 600; transition: transform 0.2s, box-shadow 0.2s; width: 100%;}
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4); color: white; }
    .status-safe { color: #10B981; font-size: 2.5rem; font-weight: 900; background: rgba(16, 185, 129, 0.1); padding: 20px; border-radius: 12px; border: 2px solid #10B981; margin-top: 20px;}
    .status-lead { color: #EF4444; font-size: 2.5rem; font-weight: 900; background: rgba(239, 68, 68, 0.1); padding: 20px; border-radius: 12px; border: 2px solid #EF4444; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api", "model.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

def main():
    st.markdown('<h1 class="main-title">Scanner Output</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 1.1rem; margin-bottom: 30px;'>Upload your scan data to determine block safety.</p>", unsafe_allow_html=True)
    
    model = load_model()
    if model is None:
        st.warning("⚠️ Model not found. Please train the model via the api script first.")
        
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Muon Tracking CSV", type=['csv'])
    
    if uploaded_file is not None:
        if st.button("Run Analysis", use_container_width=True):
            if model is None:
                st.error("Cannot run analysis without trained model.")
                return
            with st.spinner("Analyzing Coordinate Streams..."):
                time.sleep(1) # Artificial delay for effect
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Extract Features using util
                    features_df = extract_all_features(df, batch_size=10000)
                    
                    # Compute predictions
                    X = features_df[['Variance', 'Kurtosis', 'Tail_Ratio']]
                    predictions = model.predict(X)
                    
                    if 1 in predictions:
                        st.markdown('<div class="status-lead">LEAD DETECTED</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-safe">BLOCK IS SAFE</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
