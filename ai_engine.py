import streamlit as st
from transformers import pipeline
import spacy
import pytesseract
from PIL import Image
import pdfplumber
import io
import os  # Fixed: Added missing import

# --- CONFIGURATION ---
# Check if running on local Windows machine or Cloud Linux using os.name
if os.name == 'nt':  # 'nt' is the internal code for Windows
    # Update this path if your local Tesseract is installed elsewhere
    pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

# Load Spacy for basic processing
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- CACHED AI MODELS ---
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

@st.cache_resource
def load_ner():
    return pipeline("token-classification", model="d4data/biomedical-ner-all", aggregation_strategy="simple", device=-1)

@st.cache_resource
def load_qa():
    return pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)

# --- CORE FUNCTIONS ---
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.type
    text = ""
    try:
        if "image" in file_type:
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image)
        elif "pdf" in file_type:
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        return text
    except Exception as e:
        return f"Error reading file: {e}"

def summarize_medical_text(text):
    if len(text) < 50:
        return "Text is too short to summarize."
    summarizer = load_summarizer()
    input_text = text[:3000] 
    summary_result = summarizer(input_text, max_length=150, min_length=30, do_sample=False)
    return summary_result[0]['summary_text']

def get_entities(text):
    ner_pipeline = load_ner()
    input_text = text[:2000]
    entities = ner_pipeline(input_text)
    return entities

def calculate_risk_score(text):
    text_lower = text.lower()
    score = 0
    triggers = []

    critical_keywords = ['cardiac arrest', 'severe chest pain', 'stroke', 'unconscious', 'rupture', '103 f', '104 f', 'seizure']
    for k in critical_keywords:
        if k in text_lower:
            score += 3
            triggers.append(f"Critical: {k}")

    urgent_keywords = ['fracture', 'bleeding', 'fever', '101 f', '102 f', 'shortness of breath', 'vomiting', 'severe pain', 'appendicitis']
    for k in urgent_keywords:
        if k in text_lower:
            score += 2
            triggers.append(f"Urgent: {k}")
            
    standard_keywords = ['nausea', 'dizziness', 'cough', 'rash', 'mild', 'headache', 'pain']
    for k in standard_keywords:
        if k in text_lower:
            score += 1
            
    if score >= 5:
        level = "CRITICAL (Red)"
        action = "🚨 IMMEDIATE ICU ADMISSION / SURGERY REQUIRED"
        color = "#ff0000" 
    elif score >= 2:
        level = "URGENT (Orange)"
        action = "⚠️ ADMIT FOR OBSERVATION & LABS"
        color = "#ff9900" 
    else:
        level = "ROUTINE (Green)"
        action = "✅ DISCHARGE WITH MEDS / HOME CARE"
        color = "#00cc00" 
        
    return {
        "score": score,
        "level": level,
        "action": action,
        "triggers": triggers,
        "color": color
    }

def answer_question(context, question):
    qa_pipeline = load_qa()
    safe_context = context[:4000] 
    result = qa_pipeline(question=question, context=safe_context)
    return result['answer']