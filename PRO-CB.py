import streamlit as st

from detector import analyze_text, format_scores, load_classifier

@st.cache_resource
def load_model():
    return load_classifier()

classifier = load_model()

st.set_page_config(
    page_title="Cyberbullying Detection",
    layout="centered"
)

st.title("Cyberbullying Detection using Toxic-BERT")

st.write(
    "Detect toxic or cyberbullying text with scored Toxic-BERT categories."
)

user_text = st.text_area(
    "Enter Text",
    height=150
)

threshold = st.slider(
    "Detection Threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.50,
    step=0.05
)

if st.button("Analyze Text"):
    if user_text.strip() == "":
        st.warning("Please enter text")
    else:
        st.stop()

    with st.spinner("Analyzing text..."):
        result = analyze_text(user_text, classifier, threshold)

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
