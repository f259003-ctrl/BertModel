import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import requests
import zipfile
import os

# Load model from Google Drive
MODEL_DIR = "bert_emotion_model"
DRIVE_URL = "https://drive.google.com/uc?id=1AK8V47qGumkdIWO8EWt-ZkPqMvMUACQp&export=download"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_DIR):
        with open("model.zip", "wb") as f:
            f.write(requests.get(DRIVE_URL).content)
        with zipfile.ZipFile("model.zip", "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    return tokenizer, model

tokenizer, model = download_and_load_model()
model.eval()

# UI
st.title("Emotion Detection with BERT")
text_input = st.text_area("Enter a sentence to analyze emotion:")

if st.button("Detect Emotion"):
    if text_input:
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
            label = model.config.id2label[pred]
        st.success(f"Predicted Emotion: **{label}**")
    else:
        st.warning("Please enter some text.")
