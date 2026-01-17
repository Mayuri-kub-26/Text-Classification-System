import streamlit as st
import joblib
import os
import matplotlib.pyplot as plt
from PIL import Image
from src.preprocess import preprocess_text

# Paths
MODEL_PATH = os.path.join('models', 'spam_classifier.pkl')
CM_PATH = os.path.join('models', 'confusion_matrix.png')

# Page Config
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stTextArea>div>div>textarea {
        background-color: #ffffff;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return None

def main():
    st.title("Text Classification System")
    st.write("Enter a message below to check if it's **Spam** or **Not Spam**(Ham)")

    # Load Model
    model = load_model()
    
    if model is None:
        st.error("Model not found! Please run `python src/train.py` first to train the model.")
        return

    # User Input
    user_input = st.text_area("Message Content", height=150, placeholder="Type your message here...")

    if st.button("Analyze Message"):
        if user_input.strip():
            # Preprocess
            processed_text = preprocess_text(user_input)
            
            # Predict
            prediction = model.predict([processed_text])[0]
            probability = model.predict_proba([processed_text]).max()
            
            # Display Result
            if prediction == 'spam':
                st.error(f"**SPAM DETECTED** (Confidence: {probability:.2%})")
            else:
                st.success(f"**NOT SPAM**/Ham (Confidence: {probability:.2%})")
        else:
            st.warning("Please enter a message to analyze.")

    st.markdown("---")
    st.header("Model Performance")
    
    if os.path.exists(CM_PATH):
        image = Image.open(CM_PATH)
        st.image(image, caption='Confusion Matrix', use_container_width=True)
    else:
        st.info("Confusion matrix image not available.")
    
    st.markdown("### Model Details")
    st.info("""
    - **Algorithm**: Naive Bayes (MultinomialNB)
    - **Feature Extraction**: TF-IDF Vectorizer
    - **Preprocessing**: Lowercasing, Tokenization, Stopword Removal, Stemming
    """)

if __name__ == "__main__":
    main()
