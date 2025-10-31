import streamlit as st
import numpy as np
import pandas as pd
from model_loader import load_model_from_drive
from utils import predict_emotion

# Load model and tokenizer
@st.cache_resource
def load_model():
    return load_model_from_drive()

model, tokenizer, device = load_model()

# Define label names (must match your dataset order)
emotion_labels = ["joy", "sadness", "anger", "neutral"]

st.set_page_config(page_title="BERT Emotion Detection", page_icon="🤖", layout="wide")

st.title("🤖 BERT Emotion Detection App")
st.markdown("Enter a sentence below to predict its emotion using a fine-tuned BERT model.")

user_input = st.text_area("📝 Enter your text here:", height=150)

if st.button("Predict Emotion"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            label_id, probs = predict_emotion(user_input, model, tokenizer, device)
            predicted_emotion = emotion_labels[label_id]

        st.success(f"🎯 **Predicted Emotion:** {predicted_emotion.upper()}")

        # Show probability pie chart
        prob_df = pd.DataFrame({
            "Emotion": emotion_labels,
            "Probability": np.round(probs, 3)
        }).sort_values("Probability", ascending=False)

        st.subheader("Confidence Scores")
        st.dataframe(prob_df)

        st.subheader("📊 Emotion Probability Chart")
        st.bar_chart(prob_df.set_index("Emotion"))
    else:
        st.warning("Please enter a sentence to predict.")
