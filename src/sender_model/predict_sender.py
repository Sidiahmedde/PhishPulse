import joblib

# Load the trained model
model = joblib.load("src/sender_model/sender_model.pkl")
vectorizer = joblib.load("src/sender_model/vectorizer.pkl")

while True:

    text = input("Enter email text (or type quit): ")

    if text.lower() == "quit":
        break

    # Convert text to numbers
    X = vectorizer.transform([text])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    print("\nText:", text)
    print("Phishing probability:", round(probability,4))

    if prediction == 1:
        print("Prediction: PHISHING\n")
    else:
        print("Prediction: LEGIT\n")