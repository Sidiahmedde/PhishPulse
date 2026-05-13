import pandas as pd
import re
import json

# =========================
# Load dataset
# =========================
df = pd.read_csv("../../data/raw/CEAS_08.csv")

# =========================
# Normalize column names
# =========================
df = df.drop(columns=['receiver', 'date'])
df.columns = [col.strip().lower() for col in df.columns]

# Fix common misspellings
column_mapping = {
    "sener": "sender",
    "subejct": "subject",
    "from": "sender"
}
df = df.rename(columns=column_mapping)

# =========================
# Ensure required columns
# =========================
required_cols = ["subject", "body", "urls", "sender"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# =========================
# Handle missing values
# =========================
df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")
df["urls"] = df["urls"].fillna("")
df["sender"] = df["sender"].fillna("unknown")

# Remove completely empty rows
df = df[(df["subject"].str.strip() != "") | (df["body"].str.strip() != "")]

# =========================
# Light text cleaning
# =========================
def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove problematic control characters (but KEEP useful symbols)
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)

    return text

# Apply cleaning
for col in ["subject", "body", "urls", "sender"]:
    df[col] = df[col].apply(clean_text)

# Ensure everything is string (important!)
df = df.astype(str)
df['label'] = df['label'].astype(int)




df["subject"] = df["subject"].apply(clean_text)
df["body"] = df["body"].apply(clean_text)
df["urls"] = df["urls"].apply(clean_text)
df["sender"] = df["sender"].apply(clean_text)


# =========================
# Remove duplicates
# =========================
df = df.drop_duplicates()

# =========================
# Optional: reset index
# =========================
df = df.reset_index(drop=True)

# =========================
# Save cleaned dataset
# =========================

# =========================
# Safe JSONL export
# =========================
# with open("cleaned_phishing_dataset.json", "w", encoding="utf-8") as f:
#     for record in df.to_dict(orient="records"):
#         f.write(json.dumps(record, ensure_ascii=False) + "\n")


df.to_csv("cleaned_phishing_dataset.csv", index=False)
# df.to_json('cleaned_phishing_dataset.json', orient='records', lines=True)
print("✅ Cleaned dataset saved as cleaned_phishing_dataset.csv")