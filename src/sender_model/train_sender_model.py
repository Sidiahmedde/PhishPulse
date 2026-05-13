import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("phishing_email.csv")

# Features and labels
X = df["text_combined"]
y = df["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words="english")
X_vector = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "sender_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved successfully.")