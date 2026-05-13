import pandas as pd
import os
import re

# ---------- CONFIG ----------
DATA_PATH = "data/raw/spam_assassin.csv"
OUTPUT_PATH = "spam_assassin_subjects_only.csv"

# ---------- SUBJECT EXTRACTOR FOR FLATTENED TEXT ----------
def extract_subject(email_text):
    try:
        text = str(email_text)

        # Look for "Subject:" and capture everything until the next field (like From:, Date:, etc.)
        match = re.search(r"Subject:\s*(.*?)(?:\s+[A-Z][a-zA-Z\-]*:|$)", text)

        if match:
            return match.group(1).strip()
        return ""
    except:
        return ""

# ---------- LOAD ----------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found.")

df = pd.read_csv(DATA_PATH)

# ---------- EXTRACT ----------
df['subject'] = df['text'].apply(extract_subject)

# ---------- KEEP ONLY WHAT YOU NEED ----------
df = df[['target', 'subject']]

# ---------- SAVE ----------
df.to_csv(OUTPUT_PATH, index=False)

print("Done!")
print(df.head())