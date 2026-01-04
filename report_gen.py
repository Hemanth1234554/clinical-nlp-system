from fpdf import FPDF

def clean_text(text):
    """
    1. Replaces emojis with text labels (PDF doesn't support emojis).
    2. Fixes encoding errors.
    """
    if not text:
        return ""
    
    # Manual Emoji Replacements
    replacements = {
        "🚨": "[CRITICAL] ",
        "⚠️": "[URGENT] ",
        "✅": "[OK] ",
        "💊": "",
        "🔬": "",
        "📈": ""
    }
    
    for emoji, replacement in replacements.items():
        text = text.replace(emoji, replacement)
        
    # Final encoding safety net
    return text.encode('latin-1', 'replace').decode('latin-1')

class PDFReport(FPDF):
    def header(self):
        # Hospital Logo / Title
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'CLINICAL INTELLIGENCE SYSTEM', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, 'Confidential AI-Generated Medical Report', 0, 1, 'C')
        self.ln(5)
        self.line(10, 30, 200, 30) # Line break

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf(summary, risk_data, entities):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 1. RISK ASSESSMENT
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1. RISK ASSESSMENT', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    # Clean the data before writing
    r_level = clean_text(f"Risk Level: {risk_data['level']}")
    r_action = clean_text(f"Recommended Action: {risk_data['action']}")
    
    pdf.cell(0, 10, r_level, 0, 1)
    pdf.cell(0, 10, r_action, 0, 1)
    pdf.ln(5)

    # 2. SUMMARY
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '2. CLINICAL SUMMARY', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 8, clean_text(summary)) 
    pdf.ln(5)

    # 3. DETECTED ENTITIES
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '3. KEY MEDICAL ENTITIES', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    pdf.multi_cell(0, 8, "The following entities were extracted and verified:")
    
    # Grouping logic
    problems = [e['word'] for e in entities if e['entity_group'].upper() in ["DISEASE_DISORDER", "SIGN_SYMPTOM"]]
    meds = [e['word'] for e in entities if e['entity_group'].upper() in ["MEDICATION", "GENE_PROTEIN"]]
    
    # Clean lists
    prob_str = clean_text(", ".join(set(problems)))
    meds_str = clean_text(", ".join(set(meds)))

    pdf.set_font('Arial', 'B', 10)
    pdf.multi_cell(0, 8, f"Identified Symptoms/Diseases: {prob_str}")
    pdf.multi_cell(0, 8, f"Identified Medications: {meds_str}")
    
    # Return as binary string for the download button
    return pdf.output(dest='S').encode('latin-1', 'replace')