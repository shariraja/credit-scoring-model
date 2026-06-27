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
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #06080F !important;
    font-family: 'Inter', sans-serif !important;
    color: #E2E8F0 !important;
}

[data-testid="stAppViewContainer"] {
    background: #06080F !important;
    min-height: 100vh;
}

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #06080F; }
::-webkit-scrollbar-thumb { background: #7C3AED; border-radius: 2px; }

.block-container {
    max-width: 1080px !important;
    padding: 0 2rem 4rem !important;
    margin: 0 auto !important;
}

/* ── HERO ── */
.hero-section {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
    position: relative;
}

/* Animated floating orbs behind hero */
.hero-section::before {
    content: '';
    position: absolute;
    top: 20px; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 200px;
    background: radial-gradient(ellipse, rgba(124,58,237,0.18) 0%, rgba(6,182,212,0.08) 50%, transparent 70%);
    pointer-events: none;
    animation: heroGlow 4s ease-in-out infinite alternate;
}

@keyframes heroGlow {
    from { opacity: 0.6; transform: translateX(-50%) scaleX(0.9); }
    to   { opacity: 1;   transform: translateX(-50%) scaleX(1.1); }
}

.brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(6,182,212,0.12));
    border: 1px solid rgba(124,58,237,0.45);
    border-radius: 100px;
    padding: 0.4rem 1.1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    color: #A78BFA;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.4rem;
    animation: fadeDown 0.6s ease both;
    box-shadow: 0 0 20px rgba(124,58,237,0.2);
}

.brand-dot {
    width: 7px; height: 7px;
    background: #7C3AED;
    border-radius: 50%;
    box-shadow: 0 0 8px #7C3AED;
    animation: pulseDot 1.8s infinite;
}
@keyframes pulseDot {
    0%,100% { transform: scale(1); box-shadow: 0 0 8px #7C3AED; }
    50%      { transform: scale(1.4); box-shadow: 0 0 16px #A78BFA; }
}

.hero-title {
    font-size: clamp(2.2rem, 5vw, 3.5rem);
    font-weight: 900;
    line-height: 1.08;
    letter-spacing: -0.03em;
    color: #F8FAFC;
    margin-bottom: 0.9rem;
    animation: fadeDown 0.7s 0.1s ease both;
}
.hero-title .grad {
    background: linear-gradient(135deg, #A78BFA 0%, #06B6D4 40%, #F472B6 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 1rem;
    color: #475569;
    max-width: 480px;
    margin: 0 auto 2rem;
    line-height: 1.7;
    animation: fadeDown 0.8s 0.2s ease both;
}

.stats-row {
    display: flex;
    justify-content: center;
    gap: 3rem;
    margin-bottom: 2.5rem;
    animation: fadeDown 0.9s 0.3s ease both;
}
.stat-item { text-align: center; }
.stat-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    display: block;
    background: linear-gradient(135deg, #A78BFA, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-label { font-size: 0.68rem; color: #334155; letter-spacing: 0.07em; text-transform: uppercase; margin-top: 0.1rem; }

.glow-line {
    width: 140px; height: 1px;
    background: linear-gradient(90deg, transparent, #7C3AED, #06B6D4, transparent);
    margin: 0 auto 2.5rem;
    animation: fadeDown 0.8s 0.35s ease both;
}

/* ── CARDS — VIBRANT GLOW BORDERS ── */

/* Card wrapper for animated border */
.card-outer {
    border-radius: 22px;
    padding: 2px;                      /* border thickness */
    margin-bottom: 1.4rem;
    position: relative;
    animation: fadeUp 0.6s ease both;
}

/* Card-1: Purple → Cyan */
.card-outer.card-purple {
    background: linear-gradient(135deg, #7C3AED, #06B6D4, #7C3AED);
    background-size: 300% 300%;
    animation: fadeUp 0.6s ease both, gradShift 4s ease infinite;
    box-shadow: 0 0 30px rgba(124,58,237,0.25), 0 0 60px rgba(6,182,212,0.1);
}

/* Card-2: Pink → Orange */
.card-outer.card-pink {
    background: linear-gradient(135deg, #EC4899, #F97316, #EC4899);
    background-size: 300% 300%;
    animation: fadeUp 0.7s 0.05s ease both, gradShiftPink 4s ease infinite;
    box-shadow: 0 0 30px rgba(236,72,153,0.25), 0 0 60px rgba(249,115,22,0.1);
}

@keyframes gradShift {
    0%,100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}
@keyframes gradShiftPink {
    0%,100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}

.card-inner {
    background: #0C1020;
    border-radius: 20px;
    padding: 1.8rem;
    height: 100%;
}

/* Card hover lift */
.card-outer:hover {
    transform: translateY(-3px);
    transition: transform 0.3s ease;
}
.card-outer {
    transition: transform 0.3s ease;
}

.card-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.4rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.card-icon-purple {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, rgba(124,58,237,0.3), rgba(6,182,212,0.2));
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem;
    box-shadow: 0 0 12px rgba(124,58,237,0.3);
}

.card-icon-pink {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, rgba(236,72,153,0.3), rgba(249,115,22,0.2));
    border: 1px solid rgba(236,72,153,0.4);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem;
    box-shadow: 0 0 12px rgba(236,72,153,0.3);
}

.card-title-purple {
    font-size: 0.88rem;
    font-weight: 700;
    background: linear-gradient(135deg, #A78BFA, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.02em;
}
.card-title-pink {
    font-size: 0.88rem;
    font-weight: 700;
    background: linear-gradient(135deg, #F472B6, #FB923C);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.02em;
}

/* ── WIDGET OVERRIDES ── */
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
    font-size: 0.73rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

[data-testid="stNumberInput"] input {
    background: rgba(6, 8, 15, 0.8) !important;
    border: 1px solid rgba(124,58,237,0.25) !important;
    border-radius: 10px !important;
    color: #C4B5FD !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.18) !important;
    outline: none !important;
    color: #E9D5FF !important;
}

[data-testid="stSelectbox"] > div > div {
    background: rgba(6, 8, 15, 0.8) !important;
    border: 1px solid rgba(236,72,153,0.25) !important;
    border-radius: 10px !important;
    color: #FBCFE8 !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.25s !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(236,72,153,0.55) !important;
    box-shadow: 0 0 0 3px rgba(236,72,153,0.1) !important;
}

[data-baseweb="popover"] {
    background: #0C1020 !important;
    border: 1px solid rgba(124,58,237,0.35) !important;
    border-radius: 12px !important;
}
[data-baseweb="menu"] li { color: #CBD5E1 !important; font-size: 0.88rem !important; }
[data-baseweb="menu"] li:hover { background: rgba(124,58,237,0.18) !important; }

[data-testid="stNumberInput"] button {
    background: rgba(124,58,237,0.12) !important;
    border-color: rgba(124,58,237,0.3) !important;
    color: #A78BFA !important;
}

/* ── PREDICT BUTTON ── */
.btn-wrap { padding: 0.4rem 0 1.2rem; }

[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #7C3AED 0%, #06B6D4 50%, #EC4899 100%) !important;
    background-size: 200% 200% !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 1.05rem 2rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.03em !important;
    cursor: pointer !important;
    box-shadow: 0 4px 24px rgba(124,58,237,0.4), 0 0 0 1px rgba(124,58,237,0.2) !important;
    animation: btnGrad 3s ease infinite !important;
    transition: transform 0.25s, box-shadow 0.25s !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="stButton"] > button::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.1) 50%, transparent 100%);
    transform: translateX(-100%);
    transition: transform 0.5s ease;
}
[data-testid="stButton"] > button:hover::after { transform: translateX(100%); }
[data-testid="stButton"] > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 40px rgba(124,58,237,0.55), 0 0 0 1px rgba(6,182,212,0.3) !important;
}
[data-testid="stButton"] > button:active { transform: translateY(-1px) !important; }

@keyframes btnGrad {
    0%,100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}

/* ── RESULT PANELS ── */
.result-outer-good {
    border-radius: 22px;
    padding: 2px;
    background: linear-gradient(135deg, #10B981, #06B6D4, #A78BFA);
    background-size: 300% 300%;
    box-shadow: 0 0 40px rgba(16,185,129,0.25), 0 0 80px rgba(6,182,212,0.1);
    animation: fadeUp 0.6s cubic-bezier(0.16,1,0.3,1) both, gradShift 4s ease infinite;
    margin-top: 0.5rem;
}

.result-outer-bad {
    border-radius: 22px;
    padding: 2px;
    background: linear-gradient(135deg, #EF4444, #F97316, #EC4899);
    background-size: 300% 300%;
    box-shadow: 0 0 40px rgba(239,68,68,0.25), 0 0 80px rgba(249,115,22,0.1);
    animation: fadeUp 0.6s cubic-bezier(0.16,1,0.3,1) both, gradShiftPink 4s ease infinite;
    margin-top: 0.5rem;
}

.result-inner {
    background: #080D18;
    border-radius: 20px;
    padding: 2rem 2.2rem;
    position: relative;
    overflow: hidden;
}

/* Ambient glow blob inside result */
.blob-good {
    position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(16,185,129,0.18), transparent 65%);
    pointer-events: none;
    animation: blobFloat 5s ease-in-out infinite alternate;
}
.blob-bad {
    position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(239,68,68,0.18), transparent 65%);
    pointer-events: none;
    animation: blobFloat 5s ease-in-out infinite alternate;
}
@keyframes blobFloat {
    from { transform: translate(0, 0) scale(1); }
    to   { transform: translate(-15px, 15px) scale(1.1); }
}

.result-tag-good {
    font-size: 0.68rem; font-weight: 700;
    color: #10B981;
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-tag-bad {
    font-size: 0.68rem; font-weight: 700;
    color: #F87171;
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.result-verdict {
    font-size: clamp(1.6rem, 3.5vw, 2.4rem);
    font-weight: 900;
    letter-spacing: -0.025em;
    margin-bottom: 0.3rem;
    line-height: 1.1;
}
.verdict-good {
    background: linear-gradient(135deg, #34D399, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.verdict-bad {
    background: linear-gradient(135deg, #FCA5A5, #FB923C);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.result-desc { font-size: 0.85rem; color: #475569; margin-bottom: 1.6rem; }

/* Gauge */
.gauge-wrap { margin-bottom: 1.4rem; }
.gauge-label-row {
    display: flex; justify-content: space-between;
    margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #475569;
}
.gauge-track {
    height: 10px;
    background: rgba(255,255,255,0.05);
    border-radius: 100px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
}
.gauge-fill-good {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, #059669, #10B981, #34D399, #06B6D4);
    background-size: 200%;
    animation: gaugeFill 1.2s cubic-bezier(0.16,1,0.3,1) both, shimmer 2s linear infinite;
}
.gauge-fill-bad {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, #B91C1C, #EF4444, #F87171, #F97316);
    background-size: 200%;
    animation: gaugeFill 1.2s cubic-bezier(0.16,1,0.3,1) both, shimmerBad 2s linear infinite;
}
@keyframes gaugeFill {
    from { width: 0% !important; }
}
@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}
@keyframes shimmerBad {
    0%   { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

/* Probability chips */
.prob-row { display: flex; gap: 1rem; }
.prob-chip {
    flex: 1;
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.prob-chip-good {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(6,182,212,0.08));
    border: 1px solid rgba(16,185,129,0.3);
}
.prob-chip-bad {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(249,115,22,0.08));
    border: 1px solid rgba(239,68,68,0.3);
}
.prob-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 800;
    display: block;
    margin-bottom: 0.2rem;
}
.prob-val.good { color: #34D399; }
.prob-val.bad  { color: #F87171; }
.prob-chip-label { font-size: 0.68rem; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── FOOTER ── */
.footer {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
    border-top: 1px solid rgba(124,58,237,0.12);
    margin-top: 2rem;
}
.footer-logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem; font-weight: 700;
    background: linear-gradient(135deg, #A78BFA, #06B6D4, #F472B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.1em;
}
.footer-copy { font-size: 0.7rem; color: #1E293B; margin-top: 0.3rem; }

/* ── Keyframes ── */
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Column gap */
[data-testid="stColumns"] { gap: 1.2rem !important; }
[data-testid="column"] { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="brand-badge">
        <span class="brand-dot"></span>
        S.S_AI &nbsp;·&nbsp; Credit Intelligence Platform
    </div>
    <h1 class="hero-title">
        Predict Credit Risk<br>
        <span class="grad">with AI Precision</span>
    </h1>
    <p class="hero-sub">
        ML-powered scoring engine trained on the German Credit Dataset.
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
    <div class="glow-line"></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# FORM CARDS
# ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card-outer card-purple">
      <div class="card-inner">
        <div class="card-header">
          <div class="card-icon-purple">👤</div>
          <span class="card-title-purple">Applicant Profile</span>
        </div>
    """, unsafe_allow_html=True)

    age     = st.number_input("Age", min_value=18, max_value=100, value=35)
    sex     = st.selectbox("Sex", list(mappings['Sex'].keys()))
    job     = st.selectbox("Job Type", [0, 1, 2, 3],
                           help="0=Unskilled  |  1=Skilled  |  2=Management  |  3=Highly Skilled")
    housing = st.selectbox("Housing", list(mappings['Housing'].keys()))

    st.markdown("</div></div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card-outer card-pink">
      <div class="card-inner">
        <div class="card-header">
          <div class="card-icon-pink">💰</div>
          <span class="card-title-pink">Financial Details</span>
        </div>
    """, unsafe_allow_html=True)

    credit_amount    = st.number_input("Credit Amount (DM)", min_value=250, max_value=20000, value=3000)
    duration         = st.number_input("Duration (months)", min_value=4, max_value=72, value=12)
    saving_accounts  = st.selectbox("Saving Accounts", list(mappings['Saving accounts'].keys()))
    checking_account = st.selectbox("Checking Account", list(mappings['Checking account'].keys()))
    purpose          = st.selectbox("Purpose", list(mappings['Purpose'].keys()))

    st.markdown("</div></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────
st.markdown('<div class="btn-wrap">', unsafe_allow_html=True)
predict_clicked = st.button("✦  Analyze Credit Risk", use_container_width=True)
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
        st.markdown(f"""
        <div class="result-outer-good">
          <div class="result-inner">
            <div class="blob-good"></div>
            <div class="result-tag-good">✦ Assessment Complete</div>
            <div class="result-verdict verdict-good">✅ Low Credit Risk</div>
            <div class="result-desc">This applicant demonstrates a strong financial profile and is likely creditworthy.</div>
            <div class="gauge-wrap">
              <div class="gauge-label-row">
                <span>Creditworthiness Score</span>
                <span style="color:#34D399;font-weight:700">{good_pct:.1f}%</span>
              </div>
              <div class="gauge-track">
                <div class="gauge-fill-good" style="width:{good_pct:.1f}%"></div>
              </div>
            </div>
            <div class="prob-row">
              <div class="prob-chip prob-chip-good">
                <span class="prob-val good">{good_pct:.1f}%</span>
                <span class="prob-chip-label">Good Probability</span>
              </div>
              <div class="prob-chip prob-chip-bad">
                <span class="prob-val bad">{bad_pct:.1f}%</span>
                <span class="prob-chip-label">Risk Probability</span>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-outer-bad">
          <div class="result-inner">
            <div class="blob-bad"></div>
            <div class="result-tag-bad">✦ Assessment Complete</div>
            <div class="result-verdict verdict-bad">❌ High Credit Risk</div>
            <div class="result-desc">This applicant presents elevated risk indicators. Loan approval requires further review.</div>
            <div class="gauge-wrap">
              <div class="gauge-label-row">
                <span>Risk Level</span>
                <span style="color:#F87171;font-weight:700">{bad_pct:.1f}%</span>
              </div>
              <div class="gauge-track">
                <div class="gauge-fill-bad" style="width:{bad_pct:.1f}%"></div>
              </div>
            </div>
            <div class="prob-row">
              <div class="prob-chip prob-chip-good">
                <span class="prob-val good">{good_pct:.1f}%</span>
                <span class="prob-chip-label">Good Probability</span>
              </div>
              <div class="prob-chip prob-chip-bad">
                <span class="prob-val bad">{bad_pct:.1f}%</span>
                <span class="prob-chip-label">Risk Probability</span>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-logo">S.S_AI</div>
    <div class="footer-copy">Credit Intelligence Platform · Powered by Machine Learning · German Credit Dataset</div>
</div>
""", unsafe_allow_html=True)
