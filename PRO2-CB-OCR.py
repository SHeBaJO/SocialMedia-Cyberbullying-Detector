import streamlit as st
import re
import nltk
import easyocr
import numpy as np
from PIL import Image

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
# LOAD TOXIC-BERT MODEL
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
# LOAD EASYOCR
# ==========================================

@st.cache_resource
def load_ocr():

    reader = easyocr.Reader(['en'])

    return reader

reader = load_ocr()

# ==========================================
# STREAMLIT PAGE
# ==========================================

st.set_page_config(
    page_title="Cyberbullying Detection",
    layout="centered"
)

st.title("Cyberbullying Detection using Toxic-BERT + EasyOCR")

st.write(
    "Detect toxic and cyberbullying text from typed text or uploaded images."
)

# ==========================================
# TEXT INPUT
# ==========================================

st.subheader("1. Manual Text Input")

user_text = st.text_area(
    "Enter Text",
    height=150
)

# ==========================================
# IMAGE INPUT
# ==========================================

st.subheader("2. Upload Image for OCR")

uploaded_image = st.file_uploader(
    "Upload Image",
    type=['png', 'jpg', 'jpeg']
)

ocr_text = ""

# ==========================================
# OCR PROCESSING
# ==========================================

if uploaded_image is not None:

    image = Image.open(uploaded_image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_np = np.array(image)

    with st.spinner("Extracting text using EasyOCR..."):

        result = reader.readtext(image_np)

        extracted_texts = [text[1] for text in result]

        ocr_text = " ".join(extracted_texts)

    st.subheader("Extracted OCR Text")

    st.write(ocr_text)

# ==========================================
# SELECT INPUT SOURCE
# ==========================================

final_text = ""

if ocr_text.strip() != "":
    final_text = ocr_text
else:
    final_text = user_text

# ==========================================
# PREDICTION
# ==========================================

if st.button("Analyze Text"):

    if final_text.strip() == "":

        st.warning("Please enter text or upload image")

    else:

        # Clean text
        cleaned_text = clean_text(final_text)

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