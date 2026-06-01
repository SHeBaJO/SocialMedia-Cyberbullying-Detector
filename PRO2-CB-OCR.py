import easyocr
import numpy as np
import streamlit as st
from PIL import Image

from detector import analyze_text, format_scores, load_classifier

@st.cache_resource
def load_model():
    return load_classifier()

classifier = load_model()

@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"])

reader = load_ocr()

st.set_page_config(
    page_title="Cyberbullying Detection",
    layout="centered"
)

st.title("Cyberbullying Detection using Toxic-BERT + EasyOCR")

st.write(
    "Detect toxic or cyberbullying text from typed text or uploaded images."
)

st.subheader("1. Manual Text Input")

user_text = st.text_area(
    "Enter Text",
    height=150
)

st.subheader("2. Upload Image for OCR")

uploaded_image = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

ocr_text = ""

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_np = np.array(image)

    with st.spinner("Extracting text using EasyOCR..."):
        ocr_result = reader.readtext(image_np)
        extracted_texts = [item[1] for item in ocr_result]
        ocr_text = " ".join(extracted_texts)

    st.subheader("Extracted OCR Text")
    st.write(ocr_text if ocr_text else "No readable text found in this image.")

input_source = st.radio(
    "Analyze Source",
    ["OCR text if available", "Manual text"],
    horizontal=True
)

threshold = st.slider(
    "Detection Threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.50,
    step=0.05
)

if input_source == "OCR text if available" and ocr_text.strip():
    final_text = ocr_text
else:
    final_text = user_text

if st.button("Analyze Text"):
    if final_text.strip() == "":
        st.warning("Please enter text or upload image")
        st.stop()

    with st.spinner("Analyzing text..."):
        result = analyze_text(final_text, classifier, threshold)

    if result.is_cyberbullying:
        st.error("Cyberbullying / Toxic Content Detected")
    else:
        st.success("Safe Text")

    col1, col2, col3 = st.columns(3)
    col1.metric("Top Toxic Label", result.top_label)
    col2.metric("Confidence", f"{result.top_score * 100:.2f}%")
    col3.metric("Severity", result.severity)

    if result.cleaned_text:
        st.subheader("Cleaned Text")
        st.write(result.cleaned_text)

    if result.matched_labels:
        st.subheader("Matched Toxic Categories")
        st.dataframe(format_scores(result.matched_labels), use_container_width=True)

    with st.expander("All Model Scores"):
        st.dataframe(format_scores(result.all_scores), use_container_width=True)
