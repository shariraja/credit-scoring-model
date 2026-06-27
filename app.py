import streamlit as st
import pickle
import numpy as np

# ─────────────────────────────────────────
# Load Model, Scaler, Mappings
# ─────────────────────────────────────────
with open('lr_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(page_title="Credit Scoring Model", page_icon="💳", layout="centered")

st.title("💳 Credit Scoring Model")
st.markdown("Customer details bharein aur credit risk predict karein.")
st.divider()

# ─────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    age            = st.number_input("Age", min_value=18, max_value=100, value=35)
    job            = st.selectbox("Job Type", [0, 1, 2, 3], help="0=Unskilled, 1=Skilled, 2=Management, 3=Highly Skilled")
    housing        = st.selectbox("Housing", list(mappings['Housing'].keys()))
    saving_accounts = st.selectbox("Saving Accounts", list(mappings['Saving accounts'].keys()))
    credit_amount  = st.number_input("Credit Amount (DM)", min_value=250, max_value=20000, value=3000)

with col2:
    sex              = st.selectbox("Sex", list(mappings['Sex'].keys()))
    duration         = st.number_input("Duration (months)", min_value=4, max_value=72, value=12)
    checking_account = st.selectbox("Checking Account", list(mappings['Checking account'].keys()))
    purpose          = st.selectbox("Purpose", list(mappings['Purpose'].keys()))

st.divider()

# ─────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────
if st.button("🔍 Predict Credit Risk", use_container_width=True):

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

    st.divider()
    if prediction == 0:
        st.success(f"✅ GOOD Credit Risk — Good Probability: {probability[0]:.2%}")
    else:
        st.error(f"❌ BAD Credit Risk — Bad Probability: {probability[1]:.2%}")