import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gdown
import os

# =====================================================
# 🔹 CONFIGURATION
# =====================================================
st.set_page_config(page_title="BERT Emotion Detection", page_icon="💬")

# Google Drive folder ID (your shared folder link)
# Example: https://drive.google.com/drive/folders/1AK8V47qGumkdIWO8EWt-ZkPqMvMUACQp?usp=drive_link
DRIVE_FOLDER_ID = "1AK8V47qGumkdIWO8EWt-ZkPqMvMUACQp"
MODEL_DIR = "bert_emotion_model"

# =====================================================
# 🔹 DOWNLOAD MODEL FROM GOOGLE DRIVE (if not already)
# =====================================================
if not os.path.exists(MODEL_DIR):
    st.write("📦 Downloading model from Google Drive...")
    gdown.download_folder(f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}", quiet=False, use_cookies=False)
else:
    st.write("✅ Model found locally.")

# =====================================================
# 🔹 LOAD MODEL AND TOKENIZER
# =====================================================
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# Define label mapping (edit this based on your dataset)
id2label = {
    0: "joy",
    1: "sadness",
    2: "anger",
    3: "neutral"
}

# =====================================================
# 🔹 STREAMLIT UI
# =====================================================
st.title("💬 BERT Emotion Detection")
st.markdown("### Predict emotions from text using your fine-tuned BERT model!")

user_input = st.text_area("Enter your sentence here:", placeholder="e.g., I’m feeling so happy today!")

if st.button("🔍 Predict Emotion"):
    if not user_input.strip():
        st.warning("Please enter a sentence first!")
    else:
        encoding = tokenizer(
            user_input,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = np.argmax(probs)
        
        predicted_emotion = id2label[pred]
        st.success(f"🎯 **Predicted Emotion:** {predicted_emotion.upper()}")

        # Pie chart
        fig, ax = plt.subplots()
        ax.pie(probs, labels=[id2label[i] for i in range(len(probs))], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        # Confidence table
        st.markdown("### Confidence Scores")
        df = pd.DataFrame({
            "Emotion": [id2label[i] for i in range(len(probs))],
            "Probability": [round(p, 3) for p in probs]
        })
        st.dataframe(df)

st.markdown("---")
st.caption("Deployed via Streamlit + BERT | Model from Google Drive")
