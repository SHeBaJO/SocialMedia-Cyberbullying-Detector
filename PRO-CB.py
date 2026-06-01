import streamlit as st
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from transformers import pipeline

# ==========================================
# DOWNLOAD NLTK
# ==========================================

nltk.download('stopwords')
nltk.download('wordnet')

# ==========================================
# TEXT PREPROCESSING
# ==========================================

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = str(text)

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Tokenization
    words = text.split()

    # Remove stopwords
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ==========================================
# LOAD PRETRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    classifier = pipeline(
        "text-classification",
        model="unitary/toxic-bert"
    )

    return classifier

classifier = load_model()

# ==========================================
# STREAMLIT UI
# ==========================================

st.set_page_config(
    page_title="Cyberbullying Detection",
    layout="centered"
)

st.title("Cyberbullying Detection using Toxic-BERT")

st.write(
    "Detect toxic and cyberbullying text using a pretrained Hugging Face model."
)

# ==========================================
# USER INPUT
# ==========================================

user_text = st.text_area(
    "Enter Text",
    height=150
)

# ==========================================
# PREDICTION
# ==========================================

if st.button("Analyze Text"):

    if user_text.strip() == "":

        st.warning("Please enter text")

    else:

        # Clean text
        cleaned_text = clean_text(user_text)

        # Prediction
        result = classifier(cleaned_text)

        label = result[0]['label']
        score = result[0]['score']

        # ==================================
        # DISPLAY RESULT
        # ==================================

        st.subheader("Prediction Result")

        st.write(f"Label: {label}")

        st.write(f"Confidence Score: {score:.2f}")

        # ==================================
        # ALERTS
        # ==================================

        toxic_labels = [
            "toxic",
            "insult",
            "identity_hate",
            "threat",
            "obscene"
        ]

        if label.lower() in toxic_labels:

            st.error("Cyberbullying / Toxic Content Detected")

        else:

            st.success("Safe Text")