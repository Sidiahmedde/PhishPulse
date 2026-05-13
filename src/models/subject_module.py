import re
import numpy as np
from scipy.sparse import hstack
import joblib
from pathlib import Path

# Load pre-trained artifacts (you will save these after training)
_HERE = Path(__file__).resolve().parent
vectorizer = joblib.load(_HERE / "subject_vectorizer.pkl")
phish_model = joblib.load(_HERE / "subject_model.pkl")


def clean_subject(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'^(re|fw|fwd)\s*:\s*', '', text)
    text = re.sub(r'http\S+|www\S+', ' URL_TOKEN ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL_TOKEN ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s!?$]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_custom_features(subject):
    subject_str = str(subject)
    subject_lower = subject_str.lower()

    urgent_words = ["urgent", "action", "verify", "account", "password", "immediately"]
    num_urgent_words = sum(1 for word in urgent_words if word in subject_lower)
    num_capitals = sum(1 for c in subject_str if c.isupper())
    num_exclamations = subject_str.count('!')
    length = len(subject_str)

    return [num_urgent_words, num_capitals, num_exclamations, length]


# 🔥 THIS is your public interface
def subject_module(subject: str) -> float:
    clean = clean_subject(subject)
    tfidf_vec = vectorizer.transform([clean])
    custom_feat = np.array([extract_custom_features(subject)])

    combined = hstack([tfidf_vec, custom_feat])

    probs = phish_model.predict_proba(combined)

    return probs[0][1]  # phishing probability
