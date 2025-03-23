import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Set title
st.title("A7: Training Distillation vs LoRA")
st.markdown("<h2>Toxic Comment Classifier</h2>", unsafe_allow_html=True)
st.markdown("Enter a sentence below, and the model will predict whether it is **toxic**, **normal**, or **offensive**.")

# ✅ Load tokenizer and model from the saved student_model_odd directory
MODEL_PATH = "../saved_models/student_model_odd"  # Update to match your saved folder

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()
model.eval()

# ✅ Define label mapping from IDs to text (used during training)
id2label = {
    0: "Hate Speech",
    1: "Normal",
    2: "Offensive"
}

# Input area
user_input = st.text_area("Enter your text here:", height=120)

# Predict on button click
if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # Tokenize and predict
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_label = torch.argmax(probs, dim=1).item()

        # Display result
        st.subheader("Prediction")
        # st.write(f"**Predicted Label:** {id2label[pred_label]}")
        if pred_label == 0:
            st.error(f"Hate Speech detected!")
        elif pred_label == 2:
            st.warning(f"Offensive content detected.")
        elif pred_label == 1:
            st.info(f"This text is Normal.")

        st.subheader("Confidence Scores")
        for idx, score in enumerate(probs[0]):
            st.write(f"{id2label[idx]}: **{score:.2%}**")
