import streamlit as st
from transformers import pipeline
import spacy
import pytesseract
from PIL import Image
import pdfplumber
import io
import os

# --- CONFIGURATION ---
# Check if running on local Windows machine or Cloud Linux using os.name
if os.name == 'nt':  # 'nt' is the internal code for Windows
    # Update this path if your local Tesseract is installed elsewhere
    pytesseract.pytesseract.tesseract_cmd = r'D:\DevData\Tessarat\tesseract.exe'

# --- SELF-HEALING SPACY LOADER ---
# This block forces the cloud to download the model if it's missing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Downloading now...")
    from spacy.cli import download # type: ignore
    download("en_core_web_sm")
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
    """
    Build a comprehensive, clean structured clinical summary from a patient medical record.
    Uses section-aware parsing and deduplication for accurate, non-repetitive output.
    """
    import re

    if len(text) < 50:
        return "Text is too short to summarize."

    text_lower = text.lower()
    lines = text.split('\n')

    # ─────────────────────────────────────────────────────────────
    # STEP 1: SECTION SEGMENTATION
    # Split the document into named clinical sections
    # ─────────────────────────────────────────────────────────────
    section_headers = {
        'hpi':        r'history of present illness|hpi',
        'pmh':        r'past medical history|past surgical|medical history|surgical\s*[-–]',
        'fhx':        r'family history',
        'social':     r'social history',
        'ros':        r'review of systems',
        'vitals':     r'vital signs',
        'physical':   r'physical examination|general:',
        'assessment': r'assessment|differential diagnosis|impression',
        'plan':       r'^plan\s*:',
        'allergies':  r'allerg',
        'medications':r'medication',
    }

    sections = {k: [] for k in section_headers}
    current_section = 'hpi'

    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue
        matched = False
        for sec, pattern in section_headers.items():
            if re.search(pattern, line_strip, re.IGNORECASE):
                current_section = sec
                matched = True
                break
        if not matched:
            sections[current_section].append(line_strip)

    def sec_text(key):
        return ' '.join(sections.get(key, []))

    # ─────────────────────────────────────────────────────────────
    # STEP 2: DEMOGRAPHICS
    # ─────────────────────────────────────────────────────────────
    patient_name = None
    patient_age  = None
    patient_sex  = None

    # Name from "Patient Name:" header line
    name_hdr = re.search(r'patient\s*name\s*:\s*([A-Za-z,\s]+)', text, re.IGNORECASE)
    if name_hdr:
        raw = name_hdr.group(1).strip().split('\n')[0].strip(' ,')
        # Reformat "Last, First" → "First Last"
        if ',' in raw:
            parts = [p.strip() for p in raw.split(',')]
            patient_name = f"{parts[1]} {parts[0]}" if len(parts) >= 2 else raw
        else:
            patient_name = raw

    # Age + gender from free text
    age_m = re.search(r'(\d{1,3})\s*[-–]?\s*y(?:ear)?(?:s)?[\s/-]*o(?:ld)?\s+([A-Za-z]+)', text, re.IGNORECASE)
    if age_m:
        patient_age = age_m.group(1)
        gender_raw  = age_m.group(2).lower()
        if gender_raw in ('male', 'm', 'man'):
            patient_sex = 'male'
        elif gender_raw in ('female', 'f', 'woman', 'wf', 'bf'):
            patient_sex = 'female'
        else:
            patient_sex = gender_raw

    # ─────────────────────────────────────────────────────────────
    # STEP 3: CHIEF COMPLAINT & HPI
    # ─────────────────────────────────────────────────────────────
    hpi_body = sec_text('hpi') or text[:1500]  # fallback to top of document

    # Duration — pick the FIRST clear duration mentioned
    duration = None
    for m in re.finditer(
        r'(\d+)\s*[-–]?\s*(day|week|month|year|hour)s?',
        hpi_body, re.IGNORECASE
    ):
        val, unit = m.group(1), m.group(2).lower()
        duration = f"{val} {unit}{'s' if int(val) != 1 else ''}"
        break

    # Chief complaint — prefer "Chief Complaint" section line
    chief = None
    cc_m = re.search(r'chief complaint\s*[&:]\s*(?:id\s*:)?\s*(.+)', text, re.IGNORECASE)
    if cc_m:
        chief = cc_m.group(1).strip()[:120]
    if not chief:
        # Fallback: first mention of a symptom
        sym_scan = re.search(
            r'(chest pain|shortness of breath|dyspnea|abdominal pain|headache|'
            r'fever|cough|nausea|vomiting|dizziness|palpitations)',
            hpi_body, re.IGNORECASE
        )
        chief = sym_scan.group(1) if sym_scan else "clinical evaluation"

    # Symptom details from HPI
    symptom_details = []
    sym_map = [
        (r'exertional|on exertion',                    'exertional'),
        (r'radiates?(?:d)? to|radiation to',           'radiating'),
        (r'shortness of breath|dyspnea',               'shortness of breath'),
        (r'diaphor|sweating',                          'diaphoresis'),
        (r'nausea',                                    'nausea'),
        (r'vomiting',                                  'vomiting'),
        (r'palpitations',                              'palpitations'),
        (r'dizziness|dizzy',                           'dizziness'),
        (r'syncope|faint',                             'syncope'),
        (r'sob|shortness',                             'SOB'),
        (r'orthopnea',                                 'orthopnea'),
        (r'paroxysmal nocturnal',                      'PND'),
    ]
    seen_syms = set()
    for pattern, label in sym_map:
        if re.search(pattern, hpi_body, re.IGNORECASE) and label not in seen_syms:
            symptom_details.append(label)
            seen_syms.add(label)

    # Radiation destination
    rad_m = re.search(r'radiates?\s+(?:to|up to|down to|into)\s+(?:the\s+)?([a-z\s]+)', hpi_body, re.IGNORECASE)
    radiation_str = f"radiating to {rad_m.group(1).strip()}" if rad_m else None

    # Pain character
    char_m = re.search(
        r'(dull|sharp|aching|burning|stabbing|pressure|squeezing|tightness|throbbing)',
        hpi_body, re.IGNORECASE
    )
    pain_char = char_m.group(1).lower() if char_m else None

    # ─────────────────────────────────────────────────────────────
    # STEP 4: PAST MEDICAL & SURGICAL HISTORY (deduplicated)
    # ─────────────────────────────────────────────────────────────
    pmh_body = sec_text('pmh')

    condition_map = [
        (r'\bhypertension\b|\bhtn\b',                          'Hypertension'),
        (r'\bdiabetes\b|\bdm\s*(?:type\s*)?\d?\b|\bt2dm\b',   'Diabetes mellitus'),
        (r'\basthma\b',                                        'Asthma'),
        (r'\bcopd\b|chronic obstructive',                      'COPD'),
        (r'\batrial fibrillation\b|\bafib\b|\baf\b',           'Atrial fibrillation'),
        (r'\bcad\b|coronary artery disease',                   'Coronary artery disease (CAD)'),
        (r'\bheart failure\b|\bchf\b',                         'Congestive heart failure'),
        (r'\bstroke\b|\bcva\b',                                'Prior stroke/CVA'),
        (r'\bhyperlipidemia\b|\bdyslipidemia\b|\bhigh cholesterol\b', 'Hyperlipidemia'),
        (r'\bpeptic ulcer\b|\bpud\b',                          'Peptic ulcer disease'),
        (r'\brenal\b.*\bfailure\b|\bckd\b',                   'Chronic kidney disease'),
        (r'\bhypothyroidism\b|\bthyroid\b',                    'Thyroid disease'),
        (r'\bosteoporosis\b',                                  'Osteoporosis'),
        (r'\banemia\b',                                        'Anemia'),
        (r'\bgerd\b|gastro.esophageal reflux',                 'GERD'),
        (r'\bhysterectomy\b',                                  'Hysterectomy (surgical)'),
        (r'\boophorectomy\b|\bbso\b',                          'Oophorectomy (surgical)'),
        (r'\bappendectomy\b',                                   'Appendectomy (surgical)'),
        (r'\bcholecystectomy\b',                               'Cholecystectomy (surgical)'),
        (r'\bbunionectomy\b',                                  'Bunionectomy (surgical)'),
        (r'\bmenopause\b|surgical menopause',                  'Surgical menopause'),
        (r'\bpenicillin\b.*\ballerg|\ballerg.*\bpenicillin',   'Penicillin allergy'),
    ]

    pmh_entries = {}   # label → year_str or None (ordered dict via insertion)
    search_scope = (pmh_body + ' ' + text_lower)  # search full text for conditions

    for pattern, label in condition_map:
        # Only record each label ONCE (deduplication)
        if label in pmh_entries:
            continue
        m = re.search(pattern, search_scope, re.IGNORECASE)
        if not m:
            continue
        # Find associated year — look in a window around the match
        start = max(0, m.start() - 80)
        end   = min(len(search_scope), m.end() + 80)
        context = search_scope[start:end]

        year_ago = re.search(r'(\d{1,2})\s*(?:year|yr)s?\s*ago', context, re.IGNORECASE)
        abs_year  = re.search(r'\b(19|20)\d{2}\b', context)

        if year_ago:
            pmh_entries[label] = f"{year_ago.group(1)} year{'s' if int(year_ago.group(1))>1 else ''} ago"
        elif abs_year:
            pmh_entries[label] = abs_year.group(0)
        else:
            pmh_entries[label] = None

    # ─────────────────────────────────────────────────────────────
    # STEP 5: FAMILY HISTORY
    # ─────────────────────────────────────────────────────────────
    fhx_body = sec_text('fhx') + ' ' + text_lower
    fhx_items = []
    fhx_map = [
        (r'premature\s+cad|early\s+(?:ascvd|cad|heart)',     'premature CAD'),
        (r'(?:father|mother|brother|sister|parent).*heart attack|mi', 'family MI'),
        (r'family.*coronary|coronary.*family',                'coronary artery disease'),
        (r'family.*diabetes',                                 'diabetes'),
        (r'family.*hypertension',                             'hypertension'),
        (r'family.*cancer',                                   'cancer'),
        (r'family.*stroke',                                   'stroke'),
    ]
    seen_fhx = set()
    for pattern, label in fhx_map:
        if re.search(pattern, fhx_body, re.IGNORECASE) and label not in seen_fhx:
            fhx_items.append(label)
            seen_fhx.add(label)

    # ─────────────────────────────────────────────────────────────
    # STEP 6: SOCIAL HISTORY
    # ─────────────────────────────────────────────────────────────
    soc_body = sec_text('social') + ' ' + text_lower
    social_items = []

    smoking = None
    if re.search(r'non[- ]?smok|no\s+tobacco|tobacco\s*use\s*:\s*none|does not smoke', soc_body, re.IGNORECASE):
        smoking = 'Non-smoker'
    elif re.search(r'smok|tobacco', soc_body, re.IGNORECASE):
        pk_m = re.search(r'(\d+)\s*pack', soc_body, re.IGNORECASE)
        smoking = f"Smoker ({pk_m.group(1)}-pack history)" if pk_m else 'Smoker'
    if smoking:
        social_items.append(smoking)

    alcohol = None
    if re.search(r'denies\s+alcohol|no\s+alcohol|alcohol.*none', soc_body, re.IGNORECASE):
        alcohol = 'Non-drinker'
    elif re.search(r'alcohol|beer|wine|drink', soc_body, re.IGNORECASE):
        alcohol = 'Social drinker'
    if alcohol:
        social_items.append(alcohol)

    # Hormone/medication note
    if re.search(r'not on hormone|no hormone replacement|no hrt', soc_body, re.IGNORECASE):
        social_items.append('Not on HRT')

    # ─────────────────────────────────────────────────────────────
    # STEP 7: VITALS
    # ─────────────────────────────────────────────────────────────
    vitals_body = sec_text('vitals')
    vitals = {}
    bp_m   = re.search(r'blood pressure\s*:?\s*(\d+/\d+)|bp\s*:?\s*(\d+/\d+)', vitals_body, re.IGNORECASE)
    pr_m   = re.search(r'pulse\s*:?\s*(\d+)|heart rate\s*:?\s*(\d+)', vitals_body, re.IGNORECASE)
    temp_m = re.search(r'temp(?:erature)?\s*:?\s*([\d.]+)', vitals_body, re.IGNORECASE)
    rr_m   = re.search(r'respiration\s*:?\s*(\d+)|rr\s*:?\s*(\d+)', vitals_body, re.IGNORECASE)
    spo2_m = re.search(r'spo2\s*:?\s*(\d+)%?|o2\s*sat\s*:?\s*(\d+)', vitals_body, re.IGNORECASE)

    if bp_m:   vitals['BP']   = (bp_m.group(1) or bp_m.group(2))
    if pr_m:   vitals['HR']   = (pr_m.group(1) or pr_m.group(2))
    if temp_m: vitals['Temp'] = f"{temp_m.group(1)}°C"
    if rr_m:   vitals['RR']   = (rr_m.group(1) or rr_m.group(2))
    if spo2_m: vitals['SpO2'] = f"{(spo2_m.group(1) or spo2_m.group(2))}%"

    # ─────────────────────────────────────────────────────────────
    # STEP 8: KEY EXAM FINDINGS
    # ─────────────────────────────────────────────────────────────
    phys_body = sec_text('physical')
    exam_findings = []
    exam_map = [
        (r'murmur',                              'Cardiac murmur noted'),
        (r'third heart sound|s3\b',              'S3 heart sound present'),
        (r'fourth heart sound|s4\b',             'S4 heart sound present'),
        (r'crackle|rale',                        'Pulmonary crackles'),
        (r'wheez',                               'Wheezing'),
        (r'bruit',                               'Abdominal bruit'),
        (r'edema',                               'Peripheral edema'),
        (r'jugular venous|jvp|jvd',              'Elevated JVP'),
        (r'hepatomegaly|liver.*enlarg',          'Hepatomegaly'),
    ]
    for pattern, label in exam_map:
        if re.search(pattern, phys_body + ' ' + text_lower, re.IGNORECASE):
            exam_findings.append(label)

    # ─────────────────────────────────────────────────────────────
    # STEP 9: ASSESSMENT (working/likely diagnoses)
    # ─────────────────────────────────────────────────────────────
    assess_body = sec_text('assessment')
    assessments = []
    if assess_body:
        # Extract numbered assessment items
        for m in re.finditer(r'^\s*\d+\.\s+(.+)', assess_body, re.MULTILINE):
            item = m.group(1).strip()
            if len(item) > 5:
                assessments.append(item[:120])

    # ─────────────────────────────────────────────────────────────
    # STEP 10: BUILD SMOOTH CLINICAL PARAGRAPH NARRATIVE
    # ─────────────────────────────────────────────────────────────
    paragraphs = []

    # ── PARAGRAPH 1: Patient ID + Chief Complaint + HPI ──────────
    # Build opening clause
    name_str = patient_name if patient_name else "The patient"
    if patient_age and patient_sex:
        opening = f"{name_str} is a {patient_age}-year-old {patient_sex}"
    elif patient_age:
        opening = f"{name_str} is a {patient_age}-year-old patient"
    else:
        opening = name_str

    chief_clean = chief.rstrip('.').lower() if chief else "symptoms requiring evaluation"

    # Build one integrated HPI sentence:
    # "[Patient] presents with a [duration] history of [chief complaint]
    #  that is [character] in character, radiating to [location],
    #  aggravated by [factors], and relieved by [factors]."
    hpi_modifiers = []
    if pain_char:
        hpi_modifiers.append(f"that is {pain_char} in character")
    if radiation_str:
        # "radiating to her neck" — lowercase, fits naturally in the chain
        rad_clean = radiation_str.lower().strip().rstrip('.,;')
        hpi_modifiers.append(rad_clean)

    aggravating = []
    if re.search(r'exertional|on exertion|walking|working|physical activity', hpi_body, re.IGNORECASE):
        aggravating.append("physical exertion")
    if re.search(r'lying|recumbent|at rest during sleep|nocturnal', hpi_body, re.IGNORECASE):
        aggravating.append("rest")
    if aggravating:
        hpi_modifiers.append(f"aggravated by {' and '.join(aggravating)}")

    relieving = []
    if re.search(r'relieved.*rest|resolved.*rest|rest.*reliev|resting', hpi_body, re.IGNORECASE):
        relieving.append("rest")
    if re.search(r'nitrat|nitroglycerin', hpi_body, re.IGNORECASE):
        relieving.append("nitrates")
    if relieving:
        hpi_modifiers.append(f"relieved by {' and '.join(relieving)}")

    # Join modifiers smoothly: "that is dull, radiating to the neck, aggravated by exertion, and relieved by rest"
    if hpi_modifiers:
        # The last modifier gets "and" before it if there are multiple
        if len(hpi_modifiers) == 1:
            mod_str = hpi_modifiers[0]
        elif len(hpi_modifiers) == 2:
            mod_str = f"{hpi_modifiers[0]} and {hpi_modifiers[1]}"
        else:
            mod_str = ", ".join(hpi_modifiers[:-1]) + f", and {hpi_modifiers[-1]}"
        if duration:
            p1_sentence = f"{opening} presenting with a {duration} history of {chief_clean}, {mod_str}."
        else:
            p1_sentence = f"{opening} presenting with {chief_clean}, {mod_str}."
    else:
        if duration:
            p1_sentence = f"{opening} presenting with a {duration} history of {chief_clean}."
        else:
            p1_sentence = f"{opening} presenting with {chief_clean}."

    # Capitalise the very first letter properly
    p1_sentence = p1_sentence[0].upper() + p1_sentence[1:]

    # Associated symptoms as a separate complete sentence
    assoc = []
    assoc_map = [
        (r'shortness of breath|dyspnea',          'shortness of breath'),
        (r'diaphor|sweating',                     'diaphoresis'),
        (r'nausea',                               'nausea'),
        (r'vomiting',                             'vomiting'),
        (r'palpitations',                         'palpitations'),
        (r'dizziness|dizzy',                      'dizziness'),
        (r'syncope|faint',                        'syncope'),
        (r'orthopnea',                            'orthopnea'),
        (r'paroxysmal nocturnal dyspnea|pnd\b',  'paroxysmal nocturnal dyspnea'),
    ]
    seen_a = set()
    for pat, lbl in assoc_map:
        if re.search(pat, hpi_body, re.IGNORECASE) and lbl not in seen_a:
            assoc.append(lbl)
            seen_a.add(lbl)

    if assoc:
        if len(assoc) == 1:
            assoc_sentence = f"The episode is associated with {assoc[0]}."
        else:
            assoc_sentence = f"The episode is associated with {', '.join(assoc[:-1])}, and {assoc[-1]}."
        p1_sentence += " " + assoc_sentence

    # Relevant negatives
    negatives = []
    if 'nausea'    not in seen_a and not re.search(r'\bnausea\b',   hpi_body, re.IGNORECASE):
        negatives.append("nausea")
    if 'vomiting'  not in seen_a and not re.search(r'\bvomiting\b', hpi_body, re.IGNORECASE):
        negatives.append("vomiting")
    if 'syncope'   not in seen_a and not re.search(r'\bsyncope\b',  hpi_body, re.IGNORECASE):
        negatives.append("syncope")
    if negatives:
        p1_sentence += f" The patient denies {', '.join(negatives)}."

    paragraphs.append(p1_sentence)

    # ── PARAGRAPH 2: PMH / Surgical / Allergies / Medications ────
    p2_sentences = []

    med_hx  = [(l, y) for l, y in pmh_entries.items() if '(surgical)' not in l and 'allergy' not in l.lower()]
    surg_hx = [(l, y) for l, y in pmh_entries.items() if '(surgical)' in l]
    allergy = [(l, y) for l, y in pmh_entries.items() if 'allergy' in l.lower()]

    if med_hx:
        def fmt_med(label, yr):
            yr_str = f", diagnosed {yr}," if yr else ""
            return f"{label}{yr_str}"
        items = [fmt_med(l, y) for l, y in med_hx]
        if len(items) == 1:
            p2_sentences.append(f"Past medical history is significant for {items[0]}.")
        else:
            p2_sentences.append(f"Past medical history is significant for {', '.join(items[:-1])}, and {items[-1]}.")

    if surg_hx:
        def fmt_surg(label, yr):
            clean = label.replace(' (surgical)', '')
            return f"{clean} ({yr})" if yr else clean
        items = [fmt_surg(l, y) for l, y in surg_hx]
        if len(items) == 1:
            p2_sentences.append(f"Surgical history includes {items[0]}.")
        else:
            p2_sentences.append(f"Surgical history includes {', '.join(items[:-1])}, and {items[-1]}.")

    if allergy:
        allergy_names = [l.replace(' allergy', '').replace(' Allergy', '') for l, _ in allergy]
        p2_sentences.append(
            f"The patient has a documented allergy to {', '.join(allergy_names)}."
        )

    med_body = sec_text('medications')
    meds_found = []
    for drug in ['aspirin', 'ibuprofen', 'metoprolol', 'lisinopril', 'atorvastatin', 'metformin', 'amlodipine']:
        if re.search(drug, med_body + ' ' + text_lower, re.IGNORECASE):
            meds_found.append(drug.capitalize())
    if meds_found:
        p2_sentences.append(f"Current medications include {', '.join(meds_found)}.")

    if p2_sentences:
        paragraphs.append(" ".join(p2_sentences))

    # ── PARAGRAPH 3: Family History + Social History ──────────────
    p3_sentences = []

    if fhx_items:
        fhx_clean = [f.lower() for f in fhx_items]
        if 'premature cad' in fhx_clean and 'coronary artery disease' in fhx_clean:
            fhx_clean = [f for f in fhx_clean if f != 'coronary artery disease']
        if len(fhx_clean) == 1:
            fhx_str = fhx_clean[0]
        else:
            fhx_str = ', '.join(fhx_clean[:-1]) + f', and {fhx_clean[-1]}'
        p3_sentences.append(f"Family history is significant for {fhx_str}.")

    if social_items:
        soc_clauses = []
        for item in social_items:
            if item == 'Non-smoker':
                soc_clauses.append("is a non-smoker")
            elif item == 'Smoker':
                soc_clauses.append("has a history of tobacco use")
            elif item == 'Social drinker':
                soc_clauses.append("consumes alcohol socially")
            elif item == 'Non-drinker':
                soc_clauses.append("denies alcohol use")
            elif item == 'Not on HRT':
                soc_clauses.append("is not currently on hormone replacement therapy")
            else:
                soc_clauses.append(item.lower())
        if soc_clauses:
            soc_str = "The patient " + ", ".join(soc_clauses[:-1])
            if len(soc_clauses) > 1:
                soc_str += f", and {soc_clauses[-1]}."
            else:
                soc_str += soc_clauses[0] + "."
            p3_sentences.append(soc_str)

    if p3_sentences:
        paragraphs.append(" ".join(p3_sentences))

    # ── PARAGRAPH 4: Physical Examination ────────────────────────
    p4_sentences = []

    if vitals:
        v_parts = []
        if 'BP'   in vitals: v_parts.append(f"a blood pressure of {vitals['BP']} mmHg")
        if 'HR'   in vitals: v_parts.append(f"a heart rate of {vitals['HR']} beats per minute")
        if 'RR'   in vitals: v_parts.append(f"a respiratory rate of {vitals['RR']} breaths per minute")
        if 'Temp' in vitals: v_parts.append(f"a temperature of {vitals['Temp']}")
        if 'SpO2' in vitals: v_parts.append(f"an oxygen saturation of {vitals['SpO2']}")
        if v_parts:
            p4_sentences.append(
                f"On examination, vital signs revealed {', '.join(v_parts[:-1])}"
                + (f", and {v_parts[-1]}." if len(v_parts) > 1 else f"{v_parts[0]}.")
            )

    # Rewrite exam findings without repeating "noted" — use descriptive phrases
    exam_desc_map = {
        'Cardiac murmur noted':     'a cardiac murmur',
        'S3 heart sound present':   'a third heart sound',
        'S4 heart sound present':   'a fourth heart sound',
        'Pulmonary crackles':        'bibasilar pulmonary crackles',
        'Wheezing':                  'expiratory wheezing',
        'Abdominal bruit':           'an abdominal bruit',
        'Peripheral edema':          'peripheral edema',
        'Elevated JVP':              'elevated jugular venous pressure',
        'Hepatomegaly':              'hepatomegaly',
    }
    exam_phrases = [exam_desc_map.get(f, f.lower()) for f in exam_findings]

    if exam_phrases:
        if len(exam_phrases) == 1:
            p4_sentences.append(f"Physical examination revealed {exam_phrases[0]}.")
        else:
            p4_sentences.append(
                f"Physical examination revealed {', '.join(exam_phrases[:-1])}, and {exam_phrases[-1]}."
            )

    if p4_sentences:
        paragraphs.append(" ".join(p4_sentences))

    # ── PARAGRAPH 5: Assessment ───────────────────────────────────
    if assessments:
        # Use full text — no truncation. Build a flowing concluding sentence.
        primary = assessments[0].rstrip('.')
        if len(assessments) == 1:
            p5 = (
                f"The most likely diagnosis is {primary.lower()}, supported by the patient's "
                f"clinical presentation, risk factor profile, and examination findings."
            )
        else:
            others = [a.rstrip('.').lower() for a in assessments[1:4]]
            if len(others) == 1:
                others_str = others[0]
            else:
                others_str = ', '.join(others[:-1]) + f', and {others[-1]}'
            p5 = (
                f"The most likely diagnosis is {primary.lower()}, supported by the patient's "
                f"clinical presentation, risk factor profile, and examination findings. "
                f"Additional diagnostic considerations include {others_str}."
            )
        paragraphs.append(p5)

    return "\n\n".join(paragraphs)

def get_entities(text):
    """
    Extract clinical entities using HuggingFace NER model.
    Falls back to a robust rule-based extractor if the model returns no results.
    Returns a list of dicts with keys: word, entity_group, score
    """
    import re
    input_text = text[:3000]
    model_entities = []

    # ── Try HuggingFace NER model ──
    try:
        ner_pipeline = load_ner()
        raw = ner_pipeline(input_text)
        # Normalize entity_group labels — model may use BIO tags or different names
        label_map = {
            # d4data/biomedical-ner-all labels
            'DISEASE_DISORDER':        'DISEASE_DISORDER',
            'SIGN_SYMPTOM':            'SIGN_SYMPTOM',
            'MEDICATION':              'MEDICATION',
            'DIAGNOSTIC_PROCEDURE':    'DIAGNOSTIC_PROCEDURE',
            'ANATOMICAL_LOCATION':     'ANATOMICAL_LOCATION',
            'BIOLOGICAL_STRUCTURE':    'SIGN_SYMPTOM',
            'CLINICAL_EVENT':          'DIAGNOSTIC_PROCEDURE',
            # Common alternative naming schemes
            'Disease_disorder':        'DISEASE_DISORDER',
            'Sign_symptom':            'SIGN_SYMPTOM',
            'Medication':              'MEDICATION',
            'Diagnostic_procedure':    'DIAGNOSTIC_PROCEDURE',
        }
        for ent in raw:
            grp = ent.get('entity_group', ent.get('entity', ''))
            # Strip BIO prefix if present (B-DISEASE_DISORDER → DISEASE_DISORDER)
            grp_clean = re.sub(r'^[BIS]-', '', grp)
            normalized = label_map.get(grp_clean, grp_clean)
            model_entities.append({
                'word':         ent.get('word', ''),
                'entity_group': normalized,
                'score':        round(float(ent.get('score', 0)), 3),
            })
    except Exception:
        pass

    # ── If model returned entities, use them ──
    if model_entities:
        return model_entities

    # ── Fallback: comprehensive rule-based clinical NER ──
    rule_entities = []
    text_lower = text.lower()

    # Problems / Diseases / Symptoms
    problems_lexicon = [
        'chest pain', 'hypertension', 'diabetes', 'coronary artery disease',
        'angina', 'heart failure', 'atrial fibrillation', 'dyspnea',
        'shortness of breath', 'peptic ulcer', 'asthma', 'copd', 'stroke',
        'hysterectomy', 'oophorectomy', 'tachycardia', 'bradycardia',
        'myocardial infarction', 'palpitations', 'syncope', 'edema',
        'nausea', 'vomiting', 'dizziness', 'headache', 'fever', 'fatigue',
        'orthopnea', 'diaphoresis', 'cough', 'hemoptysis', 'anemia',
        'hyperlipidemia', 'obesity', 'cancer', 'pneumonia', 'pleural effusion',
        'back pain', 'abdominal pain', 'anxiety', 'depression',
    ]
    for p in problems_lexicon:
        if p in text_lower:
            rule_entities.append({'word': p.title(), 'entity_group': 'DISEASE_DISORDER', 'score': 0.90})

    # Medications
    meds_lexicon = [
        'aspirin', 'ibuprofen', 'metoprolol', 'lisinopril', 'atorvastatin',
        'metformin', 'amlodipine', 'warfarin', 'heparin', 'morphine',
        'nitroglycerin', 'furosemide', 'cimetidine', 'penicillin', 'amoxicillin',
        'paracetamol', 'acetaminophen', 'tylenol', 'advil', 'diuretic',
        'beta blocker', 'ace inhibitor', 'statin', 'nitrate', 'calcium channel',
    ]
    for m in meds_lexicon:
        if m in text_lower:
            rule_entities.append({'word': m.title(), 'entity_group': 'MEDICATION', 'score': 0.90})

    # Diagnostic procedures / tests
    tests_lexicon = [
        'ecg', 'electrocardiogram', 'echocardiogram', 'cardiac catheterization',
        'x-ray', 'mri', 'ct scan', 'blood test', 'cbc', 'bmp', 'bun',
        'creatinine', 'cholesterol', 'lipid panel', 'troponin', 'stress test',
        'angiography', 'ultrasound', 'lab work', 'electrolytes', 'urinalysis',
        'ventriculogram', 'fundoscopic', 'auscultation', 'percussion',
    ]
    for t in tests_lexicon:
        if t in text_lower:
            rule_entities.append({'word': t.upper() if len(t) <= 5 else t.title(), 'entity_group': 'DIAGNOSTIC_PROCEDURE', 'score': 0.88})

    return rule_entities

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
    """
    Answer a clinical question about the patient record using extractive QA.
    Returns a string answer with confidence context.
    """
    try:
        qa_pipeline = load_qa()
        safe_context = context[:4000]
        result = qa_pipeline(question=question, context=safe_context)  # type: ignore
        answer = result.get('answer', '').strip()
        score  = float(result.get('score', 0))

        if not answer or len(answer) < 2:
            return "The model could not find a clear answer in the provided record. Please rephrase your question or check the patient text."

        # Low confidence — add a caveat
        if score < 0.15:
            return f"{answer} *(Note: Low confidence — please verify against the original record.)*"

        return answer
    except Exception as e:
        return f"Q&A error: {str(e)}. Please ensure the patient record is loaded and try again."