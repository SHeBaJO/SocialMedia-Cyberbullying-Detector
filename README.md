# Social Media Cyberbullying Detector

A Streamlit application that detects toxic or cyberbullying content in social media text. The project includes a text-only detector and an OCR-enabled detector for images containing text.

## Project Logic

The app follows a simple NLP pipeline:

1. Accept user input as typed text or, in the OCR version, an uploaded image.
2. Extract text from images with EasyOCR when an image is uploaded.
3. Clean the text by lowercasing it, removing URLs, removing non-letter characters, removing English stopwords, and lemmatizing words with NLTK.
4. Run the cleaned text through the pretrained Hugging Face `unitary/toxic-bert` text-classification model.
5. Display the predicted label and confidence score.
6. Mark the content as unsafe when the predicted label matches toxic categories such as `toxic`, `insult`, `identity_hate`, `threat`, or `obscene`.

## Files

- `PRO-CB.py` - Streamlit app for manual text input.
- `PRO2-CB-OCR.py` - Streamlit app for manual text input and image OCR.
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
