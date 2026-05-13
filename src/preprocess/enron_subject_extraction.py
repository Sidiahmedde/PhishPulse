import pandas as pd
import os

# Load Kaggle CSV
print('loading')
print(os.getcwd())
print(os.path.exists("data/raw/enron_emails.csv"))
enron_df = pd.read_csv("data/raw/enron_emails.csv")  # replace with your file name

# Function to extract the subject line from a raw email string
def extract_subject(message_text):
    for line in message_text.splitlines():
        if line.startswith("Subject:"):
            # Remove the "Subject:" prefix and strip spaces
            subject = line[len("Subject:"):].strip()
            return subject
    return ""  # return empty string if no Subject line

# Apply function to the message column
enron_df['subject'] = enron_df['message'].apply(extract_subject)

# Keep only non-empty subjects
enron_subjects = enron_df[enron_df['subject'] != ""][['subject']].copy()
enron_subjects['label'] = 0

# Save to CSV for pretraining
enron_subjects.to_csv("enron_subjects_pretraining.csv", index=False)
print(f"Saved {len(enron_subjects)} Enron subjects for pretraining.")