import gdown
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def load_model_from_drive():
    # Google Drive folder link
    drive_folder = "https://drive.google.com/drive/folders/1AK8V47qGumkdIWO8EWt-ZkPqMvMUACQp?usp=drive_link"

    # The folder where model will be stored locally
    model_dir = "bert_emotion_model"

    # If model directory doesn’t exist, download from drive
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print("🔽 Downloading model files from Google Drive...")
        # You can use gdown to download each file manually if you have file IDs.
        # For now, assume model files are manually added to this folder.
        # You can also zip model folder and use gdown.download() for one-click download.

    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, device
