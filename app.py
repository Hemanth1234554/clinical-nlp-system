import streamlit as st
import ai_engine
import db_manager
import pandas as pd
import report_gen

# 1. Page Config
st.set_page_config(page_title="Clinical NLP Summarizer", page_icon="🏥", layout="wide")
db_manager.init_db()

# --- CSS ---
st.markdown("""
    <style>
    .entity-box { display: inline-block; padding: 5px 10px; margin: 2px; border-radius: 5px; font-weight: bold; color: white; }
    .risk-card { padding: 20px; border-radius: 10px; margin-bottom: 20px; color: white; font-weight: bold; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# 2. Sidebar
st.sidebar.title("🏥 Hospital Dashboard")
st.sidebar.subheader("📈 Patient Traffic")
stats = db_manager.get_entity_stats()
if stats:
    chart_data = pd.DataFrame.from_dict(stats, orient='index', columns=['Count'])
    st.sidebar.bar_chart(chart_data)

st.sidebar.markdown("---")
st.sidebar.subheader("📜 Recent History")
history_data = db_manager.get_all_summaries()
if history_data:
    for row in history_data[:5]:
        with st.sidebar.expander(f"{row[3]}"):
            st.write(f"**Summary:** {row[2]}")

# 3. Main Area
st.title("🏥 Clinical Intelligence System")
st.markdown("### Advanced NLP: OCR, Risk Triage, NER & Q/A")

col1, col2 = st.columns([1, 1])

# --- STATE MANAGEMENT ---
if 'final_text' not in st.session_state:
    st.session_state['final_text'] = ""
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False
if 'risk' not in st.session_state:
    st.session_state['risk'] = None
if 'summary' not in st.session_state:
    st.session_state['summary'] = ""
if 'entities' not in st.session_state:
    st.session_state['entities'] = []

# --- INPUT COLUMN ---
with col1:
    st.subheader("📥 Input Data")
    input_type = st.radio("Select Input Type:", ("Paste Text", "Upload File (Image/PDF)"))
    
    if input_type == "Paste Text":
        text_input = st.text_area("Paste patient record here:", height=300)
        if text_input: st.session_state['final_text'] = text_input
    else:
        uploaded_file = st.file_uploader("Upload Medical Record", type=["png", "jpg", "jpeg", "pdf"])
        if uploaded_file is not None:
            with st.spinner("Extracting text..."):
                extracted = ai_engine.extract_text_from_file(uploaded_file)
                if extracted:
                    st.session_state['final_text'] = extracted
                    st.success("Text Extracted!")
                    with st.expander("View Text"):
                        st.text(extracted)
    
    analyze_btn = st.button("Analyze Record", type="primary")

# --- ANALYSIS COLUMN ---
with col2:
    st.subheader("📊 AI Analysis")
    
    if analyze_btn and st.session_state['final_text']:
        with st.spinner("Running Clinical Decision Support Models..."):
            try:
                text = st.session_state['final_text']
                summary = ai_engine.summarize_medical_text(text)
                raw_entities = ai_engine.get_entities(text)
                risk_data = ai_engine.calculate_risk_score(text)
                db_manager.save_summary(text, summary)
                
                # Update Session State
                st.session_state['summary'] = summary
                st.session_state['entities'] = raw_entities
                st.session_state['risk'] = risk_data
                st.session_state['analyzed'] = True
                
            except Exception as e:
                st.error(f"Error: {e}")

    # Display Results
    if st.session_state['analyzed']:
        # 1. RISK CARD
        risk = st.session_state['risk']
        if risk:
            st.markdown(f"""
                <div class="risk-card" style="background-color: {risk['color']};">
                    <h2>RISK LEVEL: {risk['level']}</h2>
                    <p>AI SUGGESTION: {risk['action']}</p>
                    <small>Detected Triggers: {", ".join(risk['triggers'])}</small>
                </div>
            """, unsafe_allow_html=True)
        
        # 2. TABS
        tab1, tab2, tab3 = st.tabs(["📝 Summary", "🔍 Entity Detection", "🤖 Dr. AI Assistant"])
        
        with tab1:
            st.info(st.session_state['summary'])
            
            # --- NEW PDF EXPORT FEATURE ---
            if st.session_state['analyzed']:
                pdf_data = report_gen.create_pdf(
                    st.session_state['summary'],
                    st.session_state['risk'],
                    st.session_state['entities']
                )
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    # The old text download
                    st.download_button("📄 Download Text", data=st.session_state['summary'], file_name="summary.txt")
                with col_d2:
                    # The NEW PDF download
                    st.download_button(
                        label="📕 Download Official PDF", 
                        data=pdf_data, 
                        file_name="Patient_Report.pdf", 
                        mime="application/pdf",
                        type="primary" 
                    )
        
        with tab2:
            st.markdown("### 🧬 Medical Entity Extractor (Strict)")
            
            # MERGE TOKENS
            clean_entities = []
            current_word = ""; current_type = ""
            for entity in st.session_state['entities']:
                word = entity['word']; entity_type = entity['entity_group']
                if word.startswith("##"):
                    current_word += word.replace("##", "")
                else:
                    if current_word: clean_entities.append({"word": current_word, "type": current_type})
                    current_word = word; current_type = entity_type
            if current_word: clean_entities.append({"word": current_word, "type": current_type})
            
            # --- STRICT FILTERING LOGIC ---
            ignore_types = ["SEVERITY", "DETAILED_DESCRIPTION", "LAB_VALUE", "DOSAGE", "ADMINISTRATION", 
                            "DATE", "CLINICAL_EVENT", "NONBIOLOGICAL_LOCATION", "DURATION", "ACTIVITY"]
            # Added "complaints", "recovering", "history" to the ignore list
            noise_words = ["daily", "twice", "oral", "tablet", "capsule", "mg", "ml", 
                           "history", "date", "hospital", "patient", "complaints", "recovering"]
            
            problems = []; meds = []; tests = []
            
            for ent in clean_entities:
                t = ent['type'].upper()
                w = ent['word']
                
                # Filters
                if t in ignore_types: continue
                if len(w) < 3 and w.upper() not in ["CT", "MRI", "ER", "OR", "IV"]: continue
                if w.lower() in noise_words: continue
                if any(char.isdigit() for char in w): continue

                w_clean = w.capitalize()

                if t in ["DISEASE_DISORDER", "SIGN_SYMPTOM"]:
                    problems.append(w_clean)
                elif t in ["MEDICATION", "GENE_PROTEIN"]:
                    meds.append(w_clean)
                elif t in ["DIAGNOSTIC_PROCEDURE", "BIOLOGICAL_STRUCTURE", "THERAPEUTIC_PROCEDURE"]:
                    tests.append(w_clean)

            # Columns Display
            c1, c2, c3 = st.columns(3)
            with c1:
                st.error(f"**🚨 Problems ({len(set(problems))})**")
                for p in set(problems): st.markdown(f"- {p}")
            with c2:
                st.success(f"**💊 Medications ({len(set(meds))})**")
                for m in set(meds): st.markdown(f"- {m}")
            with c3:
                st.info(f"**🔬 Procedures ({len(set(tests))})**")
                for t in set(tests): st.markdown(f"- {t}")
            
        with tab3:
            st.markdown("### Ask questions about this patient")
            question = st.text_input("Example: What medication is prescribed?")
            if st.button("Ask AI"):
                if question:
                    with st.spinner("Thinking..."):
                        safe_context = st.session_state['final_text'][:4000]
                        answer = ai_engine.answer_question(safe_context, question)
                        st.success(f"**Answer:** {answer}")