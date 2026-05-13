import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import hstack
import numpy as np
from sklearn.metrics import precision_score, f1_score
import os
from sklearn.utils import resample

enron_df = pd.read_csv("data/processed/enron_subjects_pretraining.csv")  # must have column 'subject'
spam_df = pd.read_csv("data/processed/spam_assassin_subjects.csv")
spam_only = spam_df[spam_df['target'] == 1]
legit_sa = spam_df[spam_df['target'] == 0]

def clean_subject(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()

    # Remove email prefixes
    text = re.sub(r'^(re|fw|fwd)\s*:\s*', '', text)

    # Normalize URLs and emails (instead of removing)
    text = re.sub(r'http\S+|www\S+', ' URL_TOKEN ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL_TOKEN ', text)

    # Optional: keep or remove numbers
    text = re.sub(r'\d+', '', text)

    # Keep useful symbols
    text = re.sub(r'[^\w\s!?$]', '', text)

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

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

# --- 2. Load dataset ---
print("loading dataset")
print(os.getcwd())
data = pd.read_csv("data/raw/CEAS_08.csv")  # CSV: columns = subject, label
data = data[["subject", "label"]]

data = data.dropna(how="all")
data.to_csv("data/processed/email_subject_features.csv", index=False)

data["subject_clean"] = data["subject"].apply(clean_subject)
custom_features = data["subject"].apply(extract_custom_features).tolist()

X_text = data["subject_clean"]
y = data["label"]

# --- 3. Split train/test ---
print("training")
X_train_text, X_test_text, y_train, y_test, custom_train, custom_test = train_test_split(
    X_text, y, custom_features, test_size=0.2, random_state=42, stratify=y
)

# --- 4. TF-IDF vectorization ---
print("TF-IDF vectorization")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)

# --- 5. Combine TF-IDF with custom features ---
print("combine features")
X_train_combined = hstack([X_train_vec, np.array(custom_train)])
X_test_combined = hstack([X_test_vec, np.array(custom_test)])

# --- 6. Train Logistic Regression ---
print("Logistic regression")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_combined, y_train)

# --- 7. Evaluate ---
print("Evaluate:")
y_pred = model.predict(X_test_combined)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Model Precision:", precision_score(y_test, y_pred))
print("Model F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- 8. Prediction function ---
def predict_email_subject(subject):
    print("Predicting:")
    clean = clean_subject(subject)
    tfidf_vec = vectorizer.transform([clean])
    custom_feat = np.array([extract_custom_features(subject)])
    combined = hstack([tfidf_vec, custom_feat])
    pred = model.predict(combined)
    return "Phishing" if pred[0] == 1 else "Legitimate"

# --- 9. Command-line input ---
if __name__ == "__main__":
    while True:
        new_subject = input("\nEnter an email subject to check (or 'quit' to exit): ")
        if new_subject.lower() == "quit":
            print("Exiting phishing detector.")
            break
        print("Starting: ")
        result = predict_email_subject(new_subject)
        print("Prediction:", result)
