import streamlit as st
import torch
import os
import gdown
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# ======================
# CONFIGURATION
# ======================
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1AK8V47qGumkdIWO8EWt-ZkPqMvMUACQp?usp=drive_link"
MODEL_DIR = "bert_emotion_model"

# ======================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# ======================
if not os.path.exists(MODEL_DIR):
    st.write("Downloading fine-tuned BERT model from Google Drive...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    gdown.download_folder(id=None, url=DRIVE_FOLDER_URL, output=MODEL_DIR, quiet=False, use_cookies=False)
    st.success("Model downloaded successfully!")

# ======================
# LOAD MODEL + TOKENIZER
# ======================
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer

model, tokenizer = load_model()
model.eval()

# Define emotion labels (should match your training label order)
emotion_labels = ['anger', 'joy', 'neutral', 'sadness']

# ================
