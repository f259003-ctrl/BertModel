import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import requests
import zipfile
import os

# Constants
MODEL_DIR = "bert_emotion_model"
DRIVE_URL = "https://drive.google.com/uc?id=157VPrMzOPF9S6d25Wrkd2n_vMnY2--xS&export=download"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_DIR):
        # Download ZIP file
        response = requests.get(DRIVE_URL)
        with open("model.zip", "wb") as f:
            f.write(response.content)

        # Extract ZIP file
        if zipfile.is_zipfile("model.zip"):
            with zipfile.ZipFile("model.zip", "r") as zip_ref:
                zip_ref.extractall(MODEL_DIR)
        else:
            raise ValueError("Downloaded file is not a valid ZIP archive.")

    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

# Load model
tokenizer, model = download_and_load_model()

# Streamlit UI
st.title("🧠 Emotion Detection with BERT")
st.write("Enter a sentence below to detect its emotional tone.")

text_input = st.text_area("Your sentence:")

if st.button("Detect Emotion"):
    if text_input.strip():
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
            label = model.config.id2label[pred]
        st.success(f"Predicted Emotion: **{label}**")
    else:
        st.warning("Please enter a sentence to analyze.")
