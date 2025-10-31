import streamlit as st
import torch
import os
import subprocess
from transformers import BertTokenizer, BertForSequenceClassification

# ===========================
# CONFIG
# ===========================
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1AK8V47qGumkdIWO8EWt-ZkPqMvMUACQp?usp=drive_link"
MODEL_DIR = "model"

# ===========================
# Helper: Safe gdown import
# ===========================
def ensure_gdown():
    try:
        import gdown
    except ModuleNotFoundError:
        st.write("Installing gdown...")
        subprocess.run(["pip", "install", "gdown"], check=True)
        import gdown
    return gdown

# ===========================
# Download model if missing
# ===========================
if not os.path.exists(MODEL_DIR):
    st.write("📦 Downloading fine-tuned BERT model from Google Drive...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    gdown = ensure_gdown()
    gdown.download_folder(url=DRIVE_FOLDER_URL, output=MODEL_DIR, quiet=False, use_cookies=False)
    st.success("✅ Model downloaded successfully!")

# ===========================
# Load model & tokenizer
# ===========================
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer

model, tokenizer = load_model()

# These must match your training labels
emotion_labels = ['anger', 'joy', 'neutral', 'sadness']

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="Emotion Detection with BERT", page_icon="💬")
st.title("💬 Emotion Detection using Fine-Tuned BERT")
st.markdown("Detect **Joy**, **Sadness**, **Anger**, or **Neutral** from text using your trained BERT model.")

text_input = st.text_area("📝 Enter a sentence to analyze emotion:", height=120)

if st.button("🔍 Predict Emotion"):
    if not text_input.strip():
        st.warning("Please enter a sentence first!")
    else:
        with st.spinner("Analyzing emotion..."):
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                logits = model(**inputs).logits
                pred_id = torch.argmax(logits, dim=1).item()
                confidence = torch.nn.functional.softmax(logits, dim=1)[0][pred_id].item()

            st.success(f"**Predicted Emotion:** {emotion_labels[pred_id].capitalize()}")
            st.progress(confidence)
            st.write(f"Confidence: **{confidence*100:.2f}%**")

st.markdown("---")
st.caption("🚀 Fine-tuned BERT Emotion Classifier | Built with Streamlit")
