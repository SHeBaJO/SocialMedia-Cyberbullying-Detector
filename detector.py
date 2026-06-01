import re
from dataclasses import dataclass
from typing import Dict, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline


TOXIC_LABELS = {
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
}


@dataclass
class DetectionResult:
    original_text: str
    cleaned_text: str
    is_cyberbullying: bool
    severity: str
    top_label: str
    top_score: float
    matched_labels: List[Dict[str, float]]
    all_scores: List[Dict[str, float]]


def ensure_nltk_data() -> None:
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }

    for resource_path, package_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)


ensure_nltk_data()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()

    cleaned_words = [
        LEMMATIZER.lemmatize(word)
        for word in words
        if word not in STOP_WORDS and len(word) > 1
    ]

    return " ".join(cleaned_words)


def load_classifier():
    return pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        return_all_scores=True,
        function_to_apply="sigmoid",
    )


def _normalize_scores(raw_scores) -> List[Dict[str, float]]:
    if raw_scores and isinstance(raw_scores[0], list):
        raw_scores = raw_scores[0]

    normalized = []
    for item in raw_scores:
        normalized.append(
            {
                "label": str(item["label"]).lower(),
                "score": float(item["score"]),
            }
        )

    return sorted(normalized, key=lambda item: item["score"], reverse=True)


def _severity(score: float) -> str:
    if score >= 0.85:
        return "High"
    if score >= 0.65:
        return "Medium"
    if score >= 0.50:
        return "Low"
    return "Safe"


def analyze_text(text: str, classifier, threshold: float = 0.50) -> DetectionResult:
    original_text = str(text).strip()
    cleaned_text = clean_text(original_text)
    model_text = cleaned_text if cleaned_text else original_text

    raw_scores = classifier(model_text)
    all_scores = _normalize_scores(raw_scores)
    toxic_scores = [item for item in all_scores if item["label"] in TOXIC_LABELS]
    matched_labels = [item for item in toxic_scores if item["score"] >= threshold]

    top = toxic_scores[0] if toxic_scores else all_scores[0]
    top_score = top["score"]
    is_cyberbullying = len(matched_labels) > 0

    return DetectionResult(
        original_text=original_text,
        cleaned_text=cleaned_text,
        is_cyberbullying=is_cyberbullying,
        severity=_severity(top_score) if is_cyberbullying else "Safe",
        top_label=top["label"],
        top_score=top_score,
        matched_labels=matched_labels,
        all_scores=all_scores,
    )


def format_scores(scores: List[Dict[str, float]]) -> List[Dict[str, object]]:
    return [
        {
            "Label": item["label"],
            "Score": round(item["score"], 4),
            "Percent": f"{item['score'] * 100:.2f}%",
        }
        for item in scores
    ]
