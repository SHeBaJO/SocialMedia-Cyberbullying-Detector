# Social Media Cyberbullying Detector

A Streamlit application that detects toxic or cyberbullying content in social media text. The project includes a text-only detector and an OCR-enabled detector for images containing text.

## Project Logic

This repository contains two Streamlit projects that use the same shared detection logic from `detector.py`.

## Project 1: Text Cyberbullying Detector

File: `PRO-CB.py`

This project detects cyberbullying or toxic content from manually typed text.

Logic flow:

1. The user enters a social media comment, message, post, or any other text.
2. The user selects a detection threshold with the slider.
3. The app sends the input text to the shared detection logic in `detector.py`.
4. The text is cleaned by:
   - converting it to lowercase,
   - removing URLs,
   - removing user mentions,
   - removing special characters and numbers,
   - removing English stopwords,
   - lemmatizing words with NLTK.
5. The cleaned text is passed to the pretrained Hugging Face `unitary/toxic-bert` model.
6. The model returns scores for multiple toxic categories instead of only one final label.
7. The app checks toxic category scores such as `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.
8. If one or more category scores are greater than or equal to the selected threshold, the text is marked as cyberbullying or toxic.
9. The app calculates severity from the top toxic score:
   - `High` for scores from `0.85` and above,
   - `Medium` for scores from `0.65` to `0.84`,
   - `Low` for scores from `0.50` to `0.64`,
   - `Safe` when no toxic category crosses the threshold.
10. The result page displays:
    - whether the text is safe or toxic,
    - the top toxic label,
    - confidence percentage,
    - severity,
    - cleaned text,
    - matched toxic categories,
    - all model scores.

## Project 2: OCR Cyberbullying Detector

File: `PRO2-CB-OCR.py`

This project detects cyberbullying or toxic content from both typed text and images that contain text.

Logic flow:

1. The user can enter text manually.
2. The user can also upload an image in `png`, `jpg`, or `jpeg` format.
3. If an image is uploaded, the app opens the image with Pillow and converts it into a NumPy array.
4. EasyOCR reads the uploaded image and extracts any visible English text.
5. The extracted OCR text is displayed in the app so the user can review what was detected from the image.
6. The user chooses which source to analyze:
   - OCR text if available,
   - manual text.
7. The selected text is sent to the same shared detection logic in `detector.py`.
8. The text is cleaned using the same preprocessing pipeline as the text-only project.
9. The cleaned text is analyzed with the pretrained `unitary/toxic-bert` model.
10. The model produces scores for all toxic categories.
11. The app compares each toxic category score with the selected detection threshold.
12. If any toxic category crosses the threshold, the image text or manual text is marked as cyberbullying or toxic.
13. The app displays the final decision, top toxic label, confidence score, severity, cleaned text, matched toxic categories, and all model scores.

## Shared Detection Logic

File: `detector.py`

Both projects use the same backend logic so the prediction behavior remains consistent.

Main responsibilities:

1. Download required NLTK resources if they are missing.
2. Clean and normalize user text.
3. Load the `unitary/toxic-bert` model as a multi-label classifier.
4. Normalize model output scores.
5. Compare toxic category scores against the selected threshold.
6. Decide whether the content is safe or cyberbullying/toxic.
7. Assign severity based on the highest toxic score.
8. Format scores for display in Streamlit tables.

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
