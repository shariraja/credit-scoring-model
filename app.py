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

    # Mappings
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
st.set_page_config(page_title="Credit Scoring Model", page_icon="💳", layout="centered")
st.title("💳 Credit Scoring Model")
st.markdown("Customer details bharein aur credit risk predict karein.")
st.divider()

# ─────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    age             = st.number_input("Age", min_value=18, max_value=100, value=35)
    job             = st.selectbox("Job Type", [0, 1, 2, 3], help="0=Unskilled, 1=Skilled, 2=Management, 3=Highly Skilled")
    housing         = st.selectbox("Housing", list(mappings['Housing'].keys()))
    saving_accounts = st.selectbox("Saving Accounts", list(mappings['Saving accounts'].keys()))
    credit_amount   = st.number_input("Credit Amount (DM)", min_value=250, max_value=20000, value=3000)

with col2:
    sex              = st.selectbox("Sex", list(mappings['Sex'].keys()))
    duration         = st.number_input("Duration (months)", min_value=4, max_value=72, value=12)
    checking_account = st.selectbox("Checking Account", list(mappings['Checking account'].keys()))
    purpose          = st.selectbox("Purpose", list(mappings['Purpose'].keys()))

st.divider()

# ─────────────────────────────────────────
# Predict
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
