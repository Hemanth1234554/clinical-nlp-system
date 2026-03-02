import streamlit as st
import ai_engine
import db_manager
import pandas as pd
import report_gen

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical NLP Summarizer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
db_manager.init_db()

# ─── GLOBAL CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --navy:       #0f1724;
    --navy-light: #1a2640;
    --navy-card:  #162032;
    --accent:     #4f8ef7;
    --accent2:    #7c5cbf;
    --teal:       #2dd4bf;
    --surface:    #f8faff;
    --text-dark:  #1e2d45;
    --text-muted: #6b7fa3;
    --white:      #ffffff;
    --red:        #ef4444;
    --orange:     #f97316;
    --green:      #22c55e;
    --radius:     14px;
    --shadow:     0 8px 32px rgba(15,23,36,0.18);
}

/* ── Reset & Font ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Main background: light ── */
.stApp {
    background: linear-gradient(135deg, #1a1f36 0%, #16213e 100%);
    min-height: 100vh;
    color: #e0e8f0;
}

/* ── Pre/code and markdown visibility for dark mode ── */
.stApp pre, .stApp code, .stApp .stText, .stApp .stMarkdown, [data-testid="stExpander"] * {
    color: #e0e8f0 !important;
}
.stApp pre, [data-testid="stExpander"] pre, [data-testid="stExpander"] code {
    background: #1f2738 !important;
    border: 1px solid rgba(255,255,255,0.03) !important;
    padding: 0.75rem !important;
    border-radius: 8px !important;
    white-space: pre-wrap !important;
    overflow-wrap: break-word !important;
}

.stTextArea textarea, .stTextInput input, .stTextInput > div > div > input {
    color: #e0e8f0 !important;
    background: #1f2738 !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
    opacity: 1 !important;
    border: 1px solid rgba(79,142,247,0.12) !important;
}
.stTextArea textarea::placeholder {
    color: #6b7fa3 !important;
    opacity: 0.7 !important;
}
.stTextArea textarea::-webkit-outer-spin-button,
.stTextArea textarea::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
}
/* disabled/read-only overrides */
.stTextArea textarea[disabled], .stTextInput input[disabled] {
    color: #e0e8f0 !important;
    background: rgba(79,142,247,0.1) !important;
    opacity: 1 !important;
}

/* Force text visibility in textareas - combat Streamlit's default styles */
textarea {
    color: #e0e8f0 !important;
    caret-color: #93c5fd !important;
}
textarea[value] {
    color: #e0e8f0 !important;
}

/* ── Sidebar: dark navy ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1724 0%, #141d2e 100%) !important;
    border-right: 2px solid rgba(79,142,247,0.2) !important;
}
[data-testid="stSidebar"] * {
    color: #e0e8f0 !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #4f8ef7 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stSidebarNav"] { display: none; }

/* ── Hero Header ── */
.hero-banner {
    background: linear-gradient(135deg, var(--navy) 0%, #1e3a6e 50%, var(--navy-light) 100%);
    border-radius: var(--radius);
    padding: 2.4rem 2.8rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow);
    animation: fadeSlideDown 0.6s ease both;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(79,142,247,0.25) 0%, transparent 70%);
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(124,92,191,0.2) 0%, transparent 70%);
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.01em;
}
.hero-sub {
    font-size: 0.95rem;
    color: #93c5fd;
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(79,142,247,0.2);
    border: 1px solid rgba(79,142,247,0.4);
    color: #93c5fd;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
}

.section-card {
    background: linear-gradient(135deg, #23283a 0%, #262f45 100%);
    border-radius: var(--radius);
    padding: 1.6rem 1.8rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    border: 1px solid rgba(79,142,247,0.08);
    animation: fadeSlideUp 0.5s ease both;
    margin-bottom: 1.2rem;
    color: #e0e8f0;
}
.section-title {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4f8ef7;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Risk Cards ── */
.risk-card {
    border-radius: var(--radius);
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    animation: pulseIn 0.5s ease both;
    box-shadow: var(--shadow);
}
.risk-card.critical { background: linear-gradient(135deg, #7f1d1d, #b91c1c); border-left: 5px solid #ef4444; }
.risk-card.urgent   { background: linear-gradient(135deg, #78350f, #c2410c); border-left: 5px solid #f97316; }
.risk-card.routine  { background: linear-gradient(135deg, #14532d, #15803d); border-left: 5px solid #22c55e; }
.risk-icon { font-size: 2.4rem; flex-shrink: 0; }
.risk-level {
    font-size: 1.2rem; font-weight: 700;
    color: var(--white); margin: 0 0 4px 0;
    letter-spacing: 0.03em;
}
.risk-action { color: rgba(255,255,255,0.85); font-size: 0.88rem; margin: 0 0 6px 0; }
.risk-triggers { color: rgba(255,255,255,0.6); font-size: 0.75rem; }

/* ── Entity Columns ── */
.entity-col {
    background: #252d44;
    border-radius: var(--radius);
    padding: 1.2rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    border-top: 4px solid transparent;
    animation: fadeSlideUp 0.5s ease both;
    color: #e0e8f0;
}
.entity-col.problems { border-top-color: var(--red); }
.entity-col.meds     { border-top-color: var(--green); }
.entity-col.tests    { border-top-color: var(--accent); }
.entity-col-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.8rem;
}
.entity-col.problems .entity-col-label { color: var(--red); }
.entity-col.meds     .entity-col-label { color: var(--green); }
.entity-col.tests    .entity-col-label { color: var(--accent); }
.entity-tag {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px 2px;
}
.entity-tag.problem { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
.entity-tag.med     { background: rgba(34,197,94,0.15); color: #86efac; border: 1px solid rgba(34,197,94,0.3); }
.entity-tag.test    { background: rgba(79,142,247,0.15); color: #93c5fd; border: 1px solid rgba(79,142,247,0.3); }

/* ── Summary Box ── */
.summary-box {
    background: #252d44;
    border-left: 4px solid var(--accent);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 2rem 2.2rem;
    font-size: 1.05rem;
    color: #e0e8f0;
    line-height: 1.9;
    animation: fadeSlideUp 0.5s ease both;
    margin-bottom: 1.5rem;
    min-height: 280px;
    max-height: 480px;
    overflow-y: auto;
}

/* ── Sidebar Stats ── */
.stat-pill {
    background: rgba(79,142,247,0.1);
    border: 1px solid rgba(79,142,247,0.25);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #93c5fd;
    display: inline-block;
    margin: 3px 2px;
}
.sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(79,142,247,0.3), transparent);
    margin: 1rem 0;
}
.history-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
    font-size: 0.78rem;
    color: rgba(224,232,240,0.86);
    cursor: pointer;
    transition: background 0.2s;
}
.history-item:hover { background: rgba(255,255,255,0.06); }

/* ── Button overrides ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 2rem !important;
    color: white !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 16px rgba(79,142,247,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(79,142,247,0.45) !important;
}
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: transform 0.15s !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }

/* ── Tab overrides ── */
.stTabs [data-baseweb="tab-list"] {
    background: #1a1f36 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    font-weight: 500 !important;
    color: #6b7fa3 !important;
    background: transparent !important;
    padding: 0.45rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: #252d44 !important;
    color: #4f8ef7 !important;
    box-shadow: 0 2px 8px rgba(79,142,247,0.2) !important;
}

/* ── Input overrides ── */
.stTextArea textarea, .stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid rgba(79,142,247,0.3) !important;
    background: #1f2738 !important;
    font-size: 0.92rem !important;
    transition: border 0.2s !important;
    color: #e0e8f0 !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color: #6b7fa3 !important;
    opacity: 0.7 !important;
}
.stTextArea textarea:focus, .stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.2) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(79,142,247,0.4) !important;
    border-radius: 12px !important;
    background: #1f2738 !important;
    padding: 1rem !important;
    transition: border-color 0.2s !important;
    color: #e0e8f0 !important;
}
[data-testid="stFileUploader"] * {
    color: #e0e8f0 !important;
    background: transparent !important;
}
[data-testid="stFileUploader"] button {
    background: #4f8ef7 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploader"] span, [data-testid="stFileUploader"] p, [data-testid="stFileUploader"] div {
    color: #e0e8f0 !important;
}
[data-testid="stFileUploader"] .stFileUploaderFile {
    background: rgba(79,142,247,0.1) !important;
    border-radius: 6px !important;
    padding: 0.25rem 0.5rem !important;
    margin-top: 0.25rem !important;
    color: #93c5fd !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(79,142,247,0.8) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #1d4ed8, #4338ca) !important;
    color: var(--white) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 16px rgba(29,78,216,0.3) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
[data-testid="stDownloadButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(29,78,216,0.4) !important;
}

/* ── Animations ── */
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseIn {
    0%   { opacity: 0; transform: scale(0.96); }
    60%  { transform: scale(1.01); }
    100% { opacity: 1; transform: scale(1); }
}
@keyframes spin {
    to { transform: rotate(360deg); }
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(79,142,247,0.3); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(79,142,247,0.55); }

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(79,142,247,0.2) !important;
    border-radius: 10px !important;
    background: #252d44 !important;
    color: #e0e8f0 !important;
}
[data-testid="stExpander"] * {
    color: #e0e8f0 !important;
}

/* Ensure extracted/preformatted text is always readable (dark mode) */
.stApp pre, .stApp code, .stApp .stText, .stApp .stMarkdown, [data-testid="stExpander"] * {
    color: #e0e8f0 !important;
}
.stApp pre, [data-testid="stExpander"] pre, [data-testid="stExpander"] code {
    background: #1f2738 !important;
    border: 1px solid rgba(255,255,255,0.03) !important;
    padding: 0.75rem !important;
    border-radius: 8px !important;
    white-space: pre-wrap !important;
    overflow-wrap: break-word !important;
}

/* Tame bright white boxes to be softer and more readable */
.stApp .block-container, .stApp .css-1d391kg, .stApp .css-1v0mbdj,
.stApp .css-18e3th9, .stApp .css-ffhzg2 {
    color: #e0e8f0 !important;
    background: transparent !important;
}

/* ── Radio ── */
.stRadio label { font-weight: 500 !important; color: var(--text-dark) !important; }

/* Main content text */
.main .block-container { padding-top: 0.8rem !important; }
h1, h2, h3 { color: #e0e8f0 !important; }
p { color: #e0e8f0 !important; }

/* GLOBAL: Force all text to be visible - last resort */
* {
    color: #e0e8f0 !important;
}
textarea, input {
    color: #e0e8f0 !important;
}
textarea::selection, input::selection {
    background: rgba(79, 142, 247, 0.4) !important;
    color: #e0e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0;">
        <div style="font-size:1.4rem; font-weight:700; color:#fff; margin-bottom:4px;">🏥 ClinicalAI</div>
        <div style="font-size:0.75rem; color:rgba(255,255,255,0.45); letter-spacing:0.08em; text-transform:uppercase;">Intelligence Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Patient Traffic Stats
    st.markdown('<div style="font-size:0.75rem; font-weight:600; color:#4f8ef7; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.6rem;">📈 Patient Traffic</div>', unsafe_allow_html=True)
    stats = db_manager.get_entity_stats()
    if stats:
        chart_data = pd.DataFrame.from_dict(stats, orient='index', columns=['Count'])
        st.bar_chart(chart_data, color="#4f8ef7")
    else:
        st.markdown('<div style="font-size:0.8rem; color:rgba(255,255,255,0.4); text-align:center; padding:1rem 0;">No data yet</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Recent History
    st.markdown('<div style="font-size:0.75rem; font-weight:600; color:#4f8ef7; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.6rem;">📜 Recent Records</div>', unsafe_allow_html=True)
    history_data = db_manager.get_all_summaries()
    if history_data:
        for row in history_data[:5]:
            summary_preview = row[2][:80] + "..." if len(row[2]) > 80 else row[2]
            st.markdown(f"""
            <div class="history-item">
                <span style="color:#4f8ef7; font-weight:600;">#{row[0]}</span>&nbsp;&nbsp;{summary_preview}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:0.8rem; color:rgba(255,255,255,0.4); text-align:center; padding:1rem 0;">No history yet</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.7rem; color:rgba(255,255,255,0.3); text-align:center; margin-top:0.5rem;">
        Powered by HuggingFace Transformers<br>& spaCy NLP
    </div>
    """, unsafe_allow_html=True)


# ─── HERO BANNER ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🔬 Clinical AI · NLP · v2.0</div>
    <div class="hero-title">Clinical Intelligence System</div>
    <p class="hero-sub">Advanced OCR · Risk Triage · Medical NER · AI Question Answering · PDF Reports</p>
</div>
""", unsafe_allow_html=True)


# ─── SESSION STATE ───────────────────────────────────────────────────────────
for key, default in [
    ('final_text', ''), ('analyzed', False),
    ('risk', None), ('summary', ''), ('entities', []),
    ('qa_answer', ''),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─── MAIN COLUMNS ────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

# ── INPUT COLUMN ────────────────────────────────────────────────────────────
with col1:
    st.markdown('<div style="background:transparent; padding:0; box-shadow:none;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📥 Patient Data Input</div>', unsafe_allow_html=True)

    input_type = st.radio(
        "Select input method:",
        ("✏️ Paste Text", "📎 Upload File (Image / PDF)"),
        horizontal=True,
        label_visibility="collapsed"
    )

    if input_type == "✏️ Paste Text":
        text_input = st.text_area(
            "Paste patient record here:",
            height=280,
            placeholder="e.g. Patient is a 54-year-old male with severe chest pain, fever of 103°F, shortness of breath..."
        )
        if text_input:
            st.session_state['final_text'] = text_input
    else:
        uploaded_file = st.file_uploader(
            "Drop your medical record here",
            type=["png", "jpg", "jpeg", "pdf"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            with st.spinner("🔍 Extracting text via OCR..."):
                extracted = ai_engine.extract_text_from_file(uploaded_file)
                if extracted:
                    st.session_state['final_text'] = extracted
                    st.success("✅ Text extracted successfully!")
                    # show extracted text immediately so user doesn't miss it
                    st.markdown("<div style='margin-top:0.5rem; font-weight:600;'>🔍 Extracted Text:</div>", unsafe_allow_html=True)
                    st.text_area(
                        "Extracted Text",
                        extracted,
                        height=200,
                        disabled=True,
                        label_visibility="hidden",
                        help="This is the text extracted by OCR. You can copy it if needed."
                    )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🧠 Analyze Record", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── ANALYSIS COLUMN ──────────────────────────────────────────────────────────
with col2:
    st.markdown('<div class="section-title">📊 AI Analysis Results</div>', unsafe_allow_html=True)

    if analyze_btn and st.session_state['final_text']:
        with st.spinner("🤖 Running Clinical Decision Support Models..."):
            try:
                text = st.session_state['final_text']
                summary  = ai_engine.summarize_medical_text(text)
                entities = ai_engine.get_entities(text)
                risk     = ai_engine.calculate_risk_score(text)
                db_manager.save_summary(text, summary)

                st.session_state['summary']  = summary
                st.session_state['entities'] = entities
                st.session_state['risk']     = risk
                st.session_state['analyzed'] = True
                st.session_state['qa_answer'] = ''  # clear old Q&A when re-analyzing

            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")

    elif analyze_btn and not st.session_state['final_text']:
        st.warning("⚠️ Please enter or upload patient data first.")

    # ── Display Results ───────────────────────────────────────────────────
    if st.session_state['analyzed']:
        risk = st.session_state['risk']
        if risk:
            level_lower = risk['level'].split()[0].lower()  # critical / urgent / routine
            cls = "critical" if "critical" in level_lower else ("urgent" if "urgent" in level_lower else "routine")
            icon = "🚨" if cls == "critical" else ("⚠️" if cls == "urgent" else "✅")
            triggers_str = ", ".join(risk['triggers']) if risk['triggers'] else "None detected"

            st.markdown(f"""
            <div class="risk-card {cls}">
                <div class="risk-icon">{icon}</div>
                <div>
                    <p class="risk-level">RISK: {risk['level']}</p>
                    <p class="risk-action">{risk['action']}</p>
                    <p class="risk-triggers">Triggers: {triggers_str}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📝 Summary", "🔍 Entity Detection", "🤖 Dr. AI Q&A"])

        # ── Tab 1: Summary ───────────────────────────────────────────────
        with tab1:
            st.markdown(st.session_state["summary"])
            pdf_bytes = report_gen.create_pdf(
                st.session_state['summary'],
                st.session_state['risk'],
                st.session_state['entities']
            )
            st.download_button(
                label="📕 Download Official PDF Report",
                data=pdf_bytes,
                file_name="Patient_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )

        # ── Tab 2: NER ────────────────────────────────────────────────────
        with tab2:
            problems, meds, tests = [], [], []
            for ent in st.session_state['entities']:
                w = ent['word'].replace("##", "")
                t = ent['entity_group']
                if len(w) > 2:
                    if t in ("DISEASE_DISORDER", "SIGN_SYMPTOM"):
                        problems.append(w.capitalize())
                    elif t == "MEDICATION":
                        meds.append(w.capitalize())
                    elif t == "DIAGNOSTIC_PROCEDURE":
                        tests.append(w.capitalize())

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="entity-col problems">
                    <div class="entity-col-label">🚨 Problems ({len(set(problems))})</div>
                    {"".join(f'<span class="entity-tag problem">{p}</span>' for p in set(problems)) or '<span style="color:#9ca3af;font-size:0.78rem;">None found</span>'}
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="entity-col meds">
                    <div class="entity-col-label">💊 Medications ({len(set(meds))})</div>
                    {"".join(f'<span class="entity-tag med">{m}</span>' for m in set(meds)) or '<span style="color:#9ca3af;font-size:0.78rem;">None found</span>'}
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="entity-col tests">
                    <div class="entity-col-label">🔬 Procedures ({len(set(tests))})</div>
                    {"".join(f'<span class="entity-tag test">{t}</span>' for t in set(tests)) or '<span style="color:#9ca3af;font-size:0.78rem;">None found</span>'}
                </div>""", unsafe_allow_html=True)

        # ── Tab 3: Q&A ────────────────────────────────────────────────────
        with tab3:
            st.markdown('<p style="color:#6b7fa3; font-size:0.88rem;">Ask any clinical question about this patient record.</p>', unsafe_allow_html=True)
            question = st.text_input("Your question:", placeholder="e.g. What medication is prescribed?  |  What is the diagnosis?")

            btn_col1, btn_col2 = st.columns([4, 1])
            with btn_col1:
                ask_clicked = st.button("🤖 Ask AI", use_container_width=True)
            with btn_col2:
                if st.button("❌", use_container_width=True, help="Clear answer"):
                    st.session_state['qa_answer'] = ''

            if ask_clicked:
                if question:
                    with st.spinner("💭 Thinking..."):
                        context = st.session_state['final_text'][:4000]
                        st.session_state['qa_answer'] = ai_engine.answer_question(context, question)
                else:
                    st.warning("Please type a question first.")

            if st.session_state.get('qa_answer'):
                st.markdown(f"""
                <div style="background: linear-gradient(135deg,#1f2738,#232a3b);
                    border-left:4px solid #22c55e;
                    border-radius: 0 12px 12px 0;
                    padding:1rem 1.2rem; margin-top:0.6rem;">
                <span style="font-size:0.75rem;font-weight:600;color:#86efac;
                         text-transform:uppercase;letter-spacing:0.08em;">AI Answer</span>
                <p style="font-size:0.95rem; color:#e0e8f0; margin:6px 0 0 0;
                      font-weight:500;">{st.session_state['qa_answer']}</p>
                </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#9ca3af;">
            <div style="font-size:3rem; margin-bottom:1rem;">🏥</div>
            <div style="font-size:0.95rem; font-weight:500;">Enter patient data on the left and click <strong>Analyze Record</strong> to begin.</div>
            <div style="font-size:0.8rem; margin-top:0.5rem; color:#c4cfdf;">AI models will run summarization, NER, and risk triage automatically.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)