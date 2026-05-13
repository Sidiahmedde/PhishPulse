import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score
from scipy.sparse import hstack
import numpy as np
from sklearn.utils import resample
import os
import joblib


# --- 1. Load pretraining datasets ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print(os.getcwd())

enron_df = pd.read_csv("data/processed/enron_subjects_pretraining.csv")
spam_df = pd.read_csv("data/processed/spam_assassin_subjects_only.csv")

spam_only = spam_df[spam_df['target'] == 1]
legit_sa = spam_df[spam_df['target'] == 0]

# --- 2. Cleaning function ---
def clean_subject(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'^(re|fw|fwd)\s*:\s*', '', text)
    text = re.sub(r'http\S+|www\S+', ' URL_TOKEN ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL_TOKEN ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s!?$]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- 3. Custom features ---
def extract_custom_features(subject):
    if pd.isna(subject):
        subject = ""
    subject_str = str(subject)
    subject_lower = subject_str.lower()
    urgent_words = ["urgent", "action", "verify", "account", "password", "immediately"]
    num_urgent_words = sum(1 for word in urgent_words if word in subject_lower)
    num_capitals = sum(1 for c in subject_str if c.isupper())
    num_exclamations = subject_str.count('!')
    length = len(subject_str)
    return [num_urgent_words, num_capitals, num_exclamations, length]

# --- 4. Preprocessing / combine for pretraining ---
print("Cleaning pretraining datasets...")
enron_df['subject_clean'] = enron_df['subject'].apply(clean_subject)
spam_only['subject_clean'] = spam_only['subject'].apply(clean_subject)
legit_sa['subject_clean'] = legit_sa['subject'].apply(clean_subject)

# Label pretraining data: spam=1, legit=0
enron_df['label'] = 0
spam_only['label'] = 1
legit_sa['label'] = 0

# Optional: oversample spam to balance pretraining
n_samples = min(len(enron_df), len(spam_only)*10)  # simple heuristic
spam_oversampled = resample(spam_only, replace=True, n_samples=n_samples, random_state=42)

# Downsample Enron to match
enron_downsampled = enron_df.sample(n=n_samples, random_state=42)

# Combine pretraining data
pretrain_df = pd.concat([enron_downsampled, spam_oversampled, legit_sa]).sample(frac=1, random_state=42).reset_index(drop=True)

# Extract features
X_pretrain_text = pretrain_df['subject_clean']
y_pretrain = pretrain_df['label']
custom_pretrain = pretrain_df['subject'].apply(extract_custom_features).tolist()

# --- 5. Vectorization ---
print("Vectorizing pretraining data...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_pretrain_vec = vectorizer.fit_transform(X_pretrain_text)
X_pretrain_combined = hstack([X_pretrain_vec, np.array(custom_pretrain)])

# --- 6. Split pretraining train/val ---
X_train_pre, X_val_pre, y_train_pre, y_val_pre = train_test_split(
    X_pretrain_combined, y_pretrain, test_size=0.1, random_state=42, stratify=y_pretrain
)

# --- 7. Train pretraining model ---
print("Training pretraining model (spam vs legit)...")
pretrain_model = LogisticRegression(max_iter=1000, class_weight='balanced')
pretrain_model.fit(X_train_pre, y_train_pre)

# Evaluate pretraining
y_pred_pre = pretrain_model.predict(X_val_pre)
print("=== Pretraining Evaluation ===")
print("Accuracy:", accuracy_score(y_val_pre, y_pred_pre))
print("Precision:", precision_score(y_val_pre, y_pred_pre))
print("F1 Score:", f1_score(y_val_pre, y_pred_pre))
print(classification_report(y_val_pre, y_pred_pre))

# --- 8. Fine-tuning on phishing dataset ---
print("Loading phishing dataset...")
phish_df = pd.read_csv("data/raw/CEAS_08.csv")
  # columns: subject, label
phish_df = phish_df[["subject", "label"]].dropna(how="all")
phish_df['subject_clean'] = phish_df['subject'].apply(clean_subject)

X_phish_text = phish_df['subject_clean']
y_phish = phish_df['label']
custom_phish = phish_df['subject'].apply(extract_custom_features).tolist()

X_phish_vec = vectorizer.transform(X_phish_text)
X_phish_combined = hstack([X_phish_vec, np.array(custom_phish)])

# Split phishing train/test
X_train_phish, X_test_phish, y_train_phish, y_test_phish = train_test_split(
    X_phish_combined, y_phish, test_size=0.2, random_state=42, stratify=y_phish
)

# Fine-tune Logistic Regression (starting from scratch)
print("Fine-tuning on phishing dataset...")
phish_model = LogisticRegression(max_iter=1000, class_weight='balanced')
phish_model.fit(X_train_phish, y_train_phish)
joblib.dump(phish_model, "subject_model.pkl")
joblib.dump(vectorizer, "subject_vectorizer.pkl")

# Evaluate phishing model
y_pred_phish = phish_model.predict(X_test_phish)
print("=== Fine-tuning Evaluation (Phishing vs Legit) ===")
print("Accuracy:", accuracy_score(y_test_phish, y_pred_phish))
print("Precision:", precision_score(y_test_phish, y_pred_phish))
print("F1 Score:", f1_score(y_test_phish, y_pred_phish))
print(classification_report(y_test_phish, y_pred_phish))

# --- 9. Prediction function ---
def predict_email_subject(subject):
    clean = clean_subject(subject)
    tfidf_vec = vectorizer.transform([clean])
    custom_feat = np.array([extract_custom_features(subject)])
    combined = hstack([tfidf_vec, custom_feat])
    pred = phish_model.predict(combined)
    probs = phish_model.predict_proba(combined)
    confidence = probs[0][pred[0]]  # probability of the predicted class

    label = "Phishing" if pred[0] == 1 else "Legitimate"

    return label, confidence

# --- 10. Command-line input ---
if __name__ == "__main__":
    while True:
        new_subject = input("\nEnter an email subject to check (or 'quit' to exit): ")
        if new_subject.lower() == "quit":
            print("Exiting phishing detector.")
            break
        label, confidence = predict_email_subject(new_subject)
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.4f}")