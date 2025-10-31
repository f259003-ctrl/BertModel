import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

# ===============================
# Load Model and Tokenizer
# ===============================
MODEL_PATH = "/content/bert_emotion_model"  # Change this path if needed
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

id2label = model.config.id2label

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Emotion Detection with BERT", page_icon="💬")
st.title("💬 Emotion Detection using Fine-Tuned BERT")
st.markdown("Enter a sentence below to detect its emotion!")

text_input = st.text_area("📝 Input Text", placeholder="Type something like: I'm feeling so happy today!")

if st.button("🔍 Detect Emotion"):
    if not text_input.strip():
        st.warning("Please enter a sentence first!")
    else:
        # Tokenize input
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=1).item()
            emotion = id2label[pred_id]
            confidence = probs[0][pred_id].item()

        # ===============================
        # Display results
        # ===============================
        st.success(f"**Predicted Emotion:** {emotion}")
        st.write(f"Confidence: **{confidence*100:.2f}%**")

        # Emotion probability chart
        probs_dict = {id2label[i]: float(probs[0][i]) for i in range(len(probs[0]))}
        st.bar_chart(probs_dict)

st.markdown("---")
st.caption("Developed with using BERT + Streamlit")
