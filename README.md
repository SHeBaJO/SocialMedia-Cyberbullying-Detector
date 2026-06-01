# Social Media Cyberbullying Detector

A Streamlit application that detects toxic or cyberbullying content in social media text. The project includes a text-only detector and an OCR-enabled detector for images containing text.

## Project Logic

The app follows a simple NLP pipeline:

1. Accept user input as typed text or, in the OCR version, an uploaded image.
2. Extract text from images with EasyOCR when an image is uploaded.
3. Clean the text by lowercasing it, removing URLs, mentions, non-letter characters, English stopwords, and lemmatizing words with NLTK.
4. Run the cleaned text through the pretrained Hugging Face `unitary/toxic-bert` text-classification model as a multi-label classifier.
5. Score every toxic category, including `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.
6. Compare category scores with a user-selected threshold.
7. Mark content as unsafe when one or more toxic categories crosses the threshold.
8. Display the top toxic label, confidence score, severity, matched categories, cleaned text, and all model scores.

## Files

- `PRO-CB.py` - Streamlit app for manual text input.
- `PRO2-CB-OCR.py` - Streamlit app for manual text input and image OCR.
- `detector.py` - Shared text cleaning, model loading, scoring, thresholding, and severity logic.
- `requirements.txt` - Python dependencies for running the apps.

## Installation

```bash
pip install -r requirements.txt
```

The first run may download NLTK data and pretrained model files.

## Run

Text-only version:

```bash
streamlit run PRO-CB.py
```

Text + OCR version:

```bash
streamlit run PRO2-CB-OCR.py
```

## Notes

- No API key is required by this project.
- Do not commit private datasets, credentials, tokens, `.env` files, or downloaded model/cache folders.
- Model predictions are automated estimates and should be reviewed before taking moderation action.
