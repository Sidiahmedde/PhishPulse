# PhishPulse

PhishPulse is an AI-powered phishing email detection platform that analyzes email content and produces an explainable phishing risk score using multiple detection modules.

## Overview

PhishPulse evaluates an email using several security-focused signals, including sender behavior, subject-line language, message body content, and URL indicators. These module-level scores are combined through a fusion model to produce a final phishing or legitimate classification.

The project includes a Flask-based web interface where users can compose or test emails and immediately view the model's prediction and confidence scores.

## Features

- Web-based phishing detection interface
- Sender analysis
- Subject-line analysis
- Body-content analysis
- URL-based feature extraction
- Fusion-based final prediction
- Explainable module-level confidence scores
- Local Flask deployment

## Technology Stack

- Python
- Flask
- scikit-learn
- pandas
- NumPy
- joblib
- HTML/CSS/JavaScript

## Project Structure

```text
PhishPulse/
├── src/
│   ├── models/
│   ├── fusion/
│   ├── preprocess/
│   └── sender_model/
├── static/
├── templates/
├── dataset/
├── graphs/
├── reports/
├── tests/
├── requirements.txt
└── README.md