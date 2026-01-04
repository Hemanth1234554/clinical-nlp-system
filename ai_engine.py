from transformers import pipeline
import spacy
import pytesseract
from PIL import Image
import pdfplumber
import io

# --- CONFIGURATION ---
# UPDATE THIS PATH TO WHERE YOU INSTALLED TESSERACT
# Example: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Your specific path from earlier:
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\tesseract.exe'

# Load Spacy for basic processing
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Fallback if model isn't found
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_text_from_file(uploaded_file):
    """
    Extracts text from Image or PDF.
    """
    file_type = uploaded_file.type
    text = ""

    try:
        # 1. Handle Images
        if "image" in file_type:
            image = Image.open(uploaded_file)
            print("Extracting text from image using OCR...")
            text = pytesseract.image_to_string(image)
        
        # 2. Handle PDFs
        elif "pdf" in file_type:
            print("Extracting text from PDF...")
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        
        return text

    except Exception as e:
        return f"Error reading file: {e}"

def summarize_medical_text(text):
    """
    Summarizes the extracted text using BART.
    """
    if len(text) < 50:
        return "Text is too short to summarize."

    print("Loading Summarization Model...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Truncate to avoid memory crash on laptops
    input_text = text[:3000] 
    
    summary_result = summarizer(input_text, max_length=150, min_length=30, do_sample=False)
    return summary_result[0]['summary_text']

def get_entities(text):
    """
    Detects medical entities using a biomedical BERT model.
    """
    print("Loading NER Model...")
    ner_pipeline = pipeline("token-classification", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
    
    # Process first 2000 chars to save time
    input_text = text[:2000]
    entities = ner_pipeline(input_text)
    return entities

def calculate_risk_score(text):
    """
    Analyzes text for critical keywords to assign a Triage Risk Score.
    """
    text_lower = text.lower()
    score = 0
    triggers = []

    # 1. CRITICAL INDICATORS (+3 points)
    critical_keywords = ['cardiac arrest', 'severe chest pain', 'stroke', 'unconscious', 'rupture', '103 f', '104 f', 'seizure']
    for k in critical_keywords:
        if k in text_lower:
            score += 3
            triggers.append(f"Critical Hit: {k}")

    # 2. URGENT INDICATORS (+2 points)
    urgent_keywords = ['fracture', 'bleeding', 'fever', '101 f', '102 f', 'shortness of breath', 'vomiting', 'severe pain', 'appendicitis']
    for k in urgent_keywords:
        if k in text_lower:
            score += 2
            triggers.append(f"Urgent Hit: {k}")
            
    # 3. STANDARD INDICATORS (+1 point)
    standard_keywords = ['nausea', 'dizziness', 'cough', 'rash', 'mild', 'headache', 'pain']
    for k in standard_keywords:
        if k in text_lower:
            score += 1
            
    # Determine Level
    if score >= 5:
        level = "CRITICAL (Red)"
        action = "🚨 IMMEDIATE ICU ADMISSION / SURGERY REQUIRED"
        color = "#ff0000" # Red
    elif score >= 2:
        level = "URGENT (Orange)"
        action = "⚠️ ADMIT FOR OBSERVATION & LABS"
        color = "#ff9900" # Orange
    else:
        level = "ROUTINE (Green)"
        action = "✅ DISCHARGE WITH MEDS / HOME CARE"
        color = "#00cc00" # Green
        
    return {
        "score": score,
        "level": level,
        "action": action,
        "triggers": triggers,
        "color": color
    }

def answer_question(context, question):
    """
    Scans the text for answers.
    """
    print("Loading Q&A Model...")
    
    # SAFETY LIMIT: Keep first 4000 chars to prevent crashing
    safe_context = context[:4000] 
    
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    result = qa_pipeline(question=question, context=safe_context)
    return result['answer']