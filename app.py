import streamlit as st
import joblib
from PIL import Image
import os

# App Title
st.set_page_config(page_title="Language Detection App")
st.title("🌍 Language Detection App")
st.write("Enter a sentence below, and I will detect which language it's written in!")

# Load Model and Vectorizer
try:
    model = joblib.load('language_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except FileNotFoundError:
    st.error("❌ Model or vectorizer not found. Please make sure 'language_model.pkl' and 'vectorizer.pkl' are in the same folder.")
    st.stop()

# Display Confusion Matrix
if os.path.exists('confusion_matrix.png'):
    image = Image.open('confusion_matrix.png')
    st.image(image, caption='Confusion Matrix', use_column_width=True)
else:
    st.warning("Confusion matrix image not found.")

# Text Input
user_input = st.text_area("📝 Type text here:")

# Prediction
if st.button("🔍 Detect Language"):
    if user_input.strip():
        try:
            text_vector = vectorizer.transform([user_input])
            prediction = model.predict(text_vector)
            st.success(f"✅ Detected Language: **{prediction[0]}**")
        except Exception as e:
            st.error(f"❌ Error while predicting: {e}")
    else:
        st.warning("⚠️ Please enter some text above.")

