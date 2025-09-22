import pdfplumber
import re
import spacy
from transformers import pipeline

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

# --- Resume Parser Functions ---

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ''
        for page in pdf.pages:
            full_text += page.extract_text() + '\n'
    return full_text

def extract_basic_info(text):
    info = {
        "Name": None,
        "Email": None,
        "Phone": None
    }

    # Use spaCy for name extraction
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not info["Name"]:
            info["Name"] = ent.text

    # Regex for email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    if email_match:
        info["Email"] = email_match.group(0)

    # Regex for phone number
    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    if phone_match:
        info["Phone"] = phone_match.group(0)

    return info

# --- Sentiment Analyzer Function ---

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

# --- Main Execution ---

def main():
    print("=== Resume Parser ===")
    resume_path = input("Enter path to resume PDF: ").strip()

    try:
        resume_text = extract_text_from_pdf(resume_path)
        basic_info = extract_basic_info(resume_text)

        print("\nExtracted Resume Info:")
        for key, value in basic_info.items():
            print(f"{key}: {value if value else 'Not Found'}")
    except Exception as e:
        print("Error reading resume:", e)
        return

    print("\n=== Sentiment Analyzer ===")
    user_text = input("Enter a review or comment: ").strip()
    if user_text:
        label, score = analyze_sentiment(user_text)
        print(f"Sentiment: {label} (Confidence: {score:.2f})")
    else:
        print("No input provided for sentiment analysis.")

if __name__ == "__main__":
    main()
