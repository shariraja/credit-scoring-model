import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────
# Train Model on Startup
# ─────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv('german_credit_data_updated.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df['Saving accounts'].fillna('unknown', inplace=True)
    df['Checking account'].fillna('unknown', inplace=True)

    cat_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    mappings = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col])
        mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        df[col] = le.transform(df[col])

    df['Credit Risk'] = df['Credit Risk'].map({1: 0, 2: 1})

    X = df.drop(columns=['Credit Risk'])
    y = df['Credit Risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    return model, scaler, mappings

model, scaler, mappings = train_model()

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="S.S_AI Credit Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# Global CSS — Full Industry UI
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #070C1A !important;
    font-family: 'Inter', sans-serif !important;
    color: #E2E8F0 !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 60% at 50% -10%, #1a2744 0%, #070C1A 60%) !important;
    min-height: 100vh;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0D1425; }
::-webkit-scrollbar-thumb { background: #2563EB; border-radius: 2px; }

/* ── Main Content Block ── */
.block-container {
    max-width: 1100px !important;
    padding: 0 2rem 4rem !important;
    margin: 0 auto !important;
}

/* ── HERO HEADER ── */
.hero-section {
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
    position: relative;
}

.brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(37, 99, 235, 0.12);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 100px;
    padding: 0.35rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    color: #60A5FA;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.4rem;
    animation: fadeSlideDown 0.6s ease both;
}

.brand-dot {
    width: 6px; height: 6px;
    background: #3B82F6;
    border-radius: 50%;
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.7); }
}

.hero-title {
    font-size: clamp(2.2rem, 5vw, 3.6rem);
    font-weight: 900;
    line-height: 1.08;
    letter-spacing: -0.03em;
    color: #F8FAFC;
    margin-bottom: 1rem;
    animation: fadeSlideDown 0.7s 0.1s ease both;
}

.hero-title span {
    background: linear-gradient(135deg, #3B82F6 0%, #818CF8 50%, #06B6D4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 1.05rem;
    color: #64748B;
    font-weight: 400;
    max-width: 500px;
    margin: 0 auto 2.5rem;
    line-height: 1.65;
    animation: fadeSlideDown 0.8s 0.2s ease both;
}

/* Stats row */
.stats-row {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    margin-bottom: 2.5rem;
    animation: fadeSlideDown 0.9s 0.3s ease both;
}
.stat-item { text-align: center; }
.stat-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #3B82F6;
    display: block;
}
.stat-label { font-size: 0.72rem; color: #475569; letter-spacing: 0.05em; text-transform: uppercase; }

/* Divider glow */
.glow-divider {
    width: 120px; height: 1px;
    background: linear-gradient(90deg, transparent, #3B82F6, transparent);
    margin: 0 auto 2.5rem;
    animation: fadeSlideDown 0.8s 0.35s ease both;
}

/* ── SECTION LABEL ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    color: #3B82F6;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* ── CARD ── */
.form-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(51, 65, 85, 0.5);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    animation: fadeSlideUp 0.6s ease both;
}
.form-card:hover {
    border-color: rgba(59, 130, 246, 0.3);
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.06);
}

.card-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(51, 65, 85, 0.4);
}

.card-icon {
    width: 36px; height: 36px;
    background: rgba(37, 99, 235, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.25);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}

.card-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #CBD5E1;
}

/* ── Streamlit Widgets Overrides ── */
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.3rem !important;
}

[data-testid="stNumberInput"] input {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(51, 65, 85, 0.6) !important;
    border-radius: 10px !important;
    color: #F1F5F9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 0.9rem !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}

[data-testid="stNumberInput"] input:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12) !important;
    outline: none !important;
}

[data-testid="stSelectbox"] > div > div {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(51, 65, 85, 0.6) !important;
    border-radius: 10px !important;
    color: #F1F5F9 !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.25s !important;
}

[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(59, 130, 246, 0.5) !important;
}

/* Dropdown options */
[data-baseweb="popover"] {
    background: #0F172A !important;
    border: 1px solid rgba(51, 65, 85, 0.7) !important;
    border-radius: 12px !important;
}
[data-baseweb="menu"] li {
    color: #CBD5E1 !important;
    font-size: 0.88rem !important;
}
[data-baseweb="menu"] li:hover {
    background: rgba(37, 99, 235, 0.15) !important;
}

/* Stepper buttons on number input */
[data-testid="stNumberInput"] button {
    background: rgba(37, 99, 235, 0.1) !important;
    border-color: rgba(51, 65, 85, 0.5) !important;
    color: #60A5FA !important;
}

/* ── PREDICT BUTTON ── */
.predict-btn-wrap { padding: 0.5rem 0 1.5rem; }

[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 50%, #1E40AF 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 1rem 2rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.35) !important;
    position: relative !important;
    overflow: hidden !important;
}

[data-testid="stButton"] > button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    transition: left 0.5s ease;
}

[data-testid="stButton"] > button:hover::before { left: 100%; }

[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(37, 99, 235, 0.5) !important;
    background: linear-gradient(135deg, #2563EB 0%, #3B82F6 50%, #1D4ED8 100%) !important;
}

[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── RESULT PANELS ── */
.result-panel {
    border-radius: 20px;
    padding: 2.2rem;
    position: relative;
    overflow: hidden;
    animation: resultReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
}

@keyframes resultReveal {
    from { opacity: 0; transform: translateY(20px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0)  scale(1); }
}

.result-good {
    background: rgba(6, 78, 59, 0.2);
    border: 1px solid rgba(16, 185, 129, 0.35);
    box-shadow: 0 0 50px rgba(16, 185, 129, 0.08);
}

.result-bad {
    background: rgba(127, 29, 29, 0.2);
    border: 1px solid rgba(239, 68, 68, 0.35);
    box-shadow: 0 0 50px rgba(239, 68, 68, 0.08);
}

.result-glow-good {
    position: absolute; top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(16,185,129,0.15), transparent 70%);
    pointer-events: none;
}
.result-glow-bad {
    position: absolute; top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(239,68,68,0.15), transparent 70%);
    pointer-events: none;
}

.result-verdict {
    font-size: clamp(1.5rem, 3vw, 2.2rem);
    font-weight: 900;
    letter-spacing: -0.02em;
    margin: 0.6rem 0 0.3rem;
}
.verdict-good { color: #10B981; }
.verdict-bad  { color: #EF4444; }

.result-sub {
    font-size: 0.88rem;
    color: #64748B;
    margin-bottom: 1.5rem;
}

/* Gauge */
.gauge-wrap { margin: 1.2rem 0; }
.gauge-track {
    height: 8px;
    background: rgba(51, 65, 85, 0.5);
    border-radius: 100px;
    overflow: hidden;
}
.gauge-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 1.2s cubic-bezier(0.16, 1, 0.3, 1);
}
.gauge-good { background: linear-gradient(90deg, #059669, #10B981, #34D399); }
.gauge-bad  { background: linear-gradient(90deg, #B91C1C, #EF4444, #F87171); }

.gauge-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
}

/* Probability chips */
.prob-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.2rem;
}
.prob-chip {
    flex: 1;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(51, 65, 85, 0.5);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.prob-chip-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    display: block;
    margin-bottom: 0.15rem;
}
.prob-chip-val.good { color: #10B981; }
.prob-chip-val.bad  { color: #EF4444; }
.prob-chip-label {
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── FOOTER ── */
.footer-section {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
    border-top: 1px solid rgba(51, 65, 85, 0.25);
    margin-top: 2rem;
}
.footer-logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: #3B82F6;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.footer-copy {
    font-size: 0.72rem;
    color: #334155;
}

/* ── Animations ── */
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-12px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Column spacing */
[data-testid="stColumns"] { gap: 1.2rem !important; }
[data-testid="column"] { padding: 0 !important; }

/* Info/divider */
[data-testid="stMarkdownContainer"] hr {
    border-color: rgba(51, 65, 85, 0.3) !important;
    margin: 1rem 0 !important;
}

/* Vertical rhythm for widgets inside cards */
.form-card [data-testid="stNumberInput"],
.form-card [data-testid="stSelectbox"] {
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="brand-badge">
        <span class="brand-dot"></span>
        S.S_AI &nbsp;·&nbsp; Credit Intelligence Platform
    </div>
    <h1 class="hero-title">
        Predict Credit Risk<br>
        <span>with AI Precision</span>
    </h1>
    <p class="hero-sub">
        Advanced ML-powered scoring engine using logistic regression on the German Credit Dataset.
        Real-time risk assessment for financial professionals.
    </p>
    <div class="stats-row">
        <div class="stat-item">
            <span class="stat-num">1000+</span>
            <span class="stat-label">Training Records</span>
        </div>
        <div class="stat-item">
            <span class="stat-num">9</span>
            <span class="stat-label">Input Features</span>
        </div>
        <div class="stat-item">
            <span class="stat-num">~75%</span>
            <span class="stat-label">Accuracy</span>
        </div>
    </div>
    <div class="glow-divider"></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# FORM — Two Cards Side by Side
# ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="form-card">
        <div class="card-header">
            <div class="card-icon">👤</div>
            <span class="card-title">Applicant Profile</span>
        </div>
    """, unsafe_allow_html=True)

    age             = st.number_input("Age", min_value=18, max_value=100, value=35)
    sex             = st.selectbox("Sex", list(mappings['Sex'].keys()))
    job             = st.selectbox("Job Type", [0, 1, 2, 3],
                                   help="0 = Unskilled  |  1 = Skilled  |  2 = Management  |  3 = Highly Skilled")
    housing         = st.selectbox("Housing", list(mappings['Housing'].keys()))

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="form-card">
        <div class="card-header">
            <div class="card-icon">💰</div>
            <span class="card-title">Financial Details</span>
        </div>
    """, unsafe_allow_html=True)

    credit_amount   = st.number_input("Credit Amount (DM)", min_value=250, max_value=20000, value=3000)
    duration        = st.number_input("Duration (months)", min_value=4, max_value=72, value=12)
    saving_accounts = st.selectbox("Saving Accounts", list(mappings['Saving accounts'].keys()))
    checking_account = st.selectbox("Checking Account", list(mappings['Checking account'].keys()))
    purpose         = st.selectbox("Purpose", list(mappings['Purpose'].keys()))

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────
st.markdown('<div class="predict-btn-wrap">', unsafe_allow_html=True)
predict_clicked = st.button("🔍  Analyze Credit Risk", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────
# RESULT
# ─────────────────────────────────────────
if predict_clicked:
    encoded = [[
        age,
        mappings['Sex'][sex],
        job,
        mappings['Housing'][housing],
        mappings['Saving accounts'][saving_accounts],
        mappings['Checking account'][checking_account],
        credit_amount,
        duration,
        mappings['Purpose'][purpose]
    ]]

    input_scaled = scaler.transform(encoded)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    good_pct = probability[0] * 100
    bad_pct  = probability[1] * 100

    if prediction == 0:
        # GOOD result
        st.markdown(f"""
        <div class="result-panel result-good">
            <div class="result-glow-good"></div>
            <div style="font-size:0.75rem;font-weight:600;color:#059669;letter-spacing:0.1em;text-transform:uppercase;">
                ✦ Assessment Complete
            </div>
            <div class="result-verdict verdict-good">✅ Low Credit Risk</div>
            <div class="result-sub">This applicant demonstrates a strong financial profile.</div>

            <div class="gauge-wrap">
                <div class="gauge-track">
                    <div class="gauge-fill gauge-good" style="width:{good_pct:.1f}%"></div>
                </div>
                <div class="gauge-labels">
                    <span>0%</span>
                    <span style="color:#10B981;font-weight:600">{good_pct:.1f}% Creditworthy</span>
                    <span>100%</span>
                </div>
            </div>

            <div class="prob-row">
                <div class="prob-chip">
                    <span class="prob-chip-val good">{good_pct:.1f}%</span>
                    <span class="prob-chip-label">Good Probability</span>
                </div>
                <div class="prob-chip">
                    <span class="prob-chip-val bad">{bad_pct:.1f}%</span>
                    <span class="prob-chip-label">Risk Probability</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # BAD result
        st.markdown(f"""
        <div class="result-panel result-bad">
            <div class="result-glow-bad"></div>
            <div style="font-size:0.75rem;font-weight:600;color:#DC2626;letter-spacing:0.1em;text-transform:uppercase;">
                ✦ Assessment Complete
            </div>
            <div class="result-verdict verdict-bad">❌ High Credit Risk</div>
            <div class="result-sub">This applicant presents elevated financial risk indicators.</div>

            <div class="gauge-wrap">
                <div class="gauge-track">
                    <div class="gauge-fill gauge-bad" style="width:{bad_pct:.1f}%"></div>
                </div>
                <div class="gauge-labels">
                    <span>0%</span>
                    <span style="color:#EF4444;font-weight:600">{bad_pct:.1f}% Risk Level</span>
                    <span>100%</span>
                </div>
            </div>

            <div class="prob-row">
                <div class="prob-chip">
                    <span class="prob-chip-val good">{good_pct:.1f}%</span>
                    <span class="prob-chip-label">Good Probability</span>
                </div>
                <div class="prob-chip">
                    <span class="prob-chip-val bad">{bad_pct:.1f}%</span>
                    <span class="prob-chip-label">Risk Probability</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="footer-section">
    <div class="footer-logo">S.S_AI</div>
    <div class="footer-copy">Credit Intelligence Platform &nbsp;·&nbsp; Powered by Machine Learning &nbsp;·&nbsp; German Credit Dataset</div>
</div>
""", unsafe_allow_html=True)
