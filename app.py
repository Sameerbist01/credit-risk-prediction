import streamlit as st
import requests

st.title("🏦 Credit Risk & Loan Approval Predictor")
st.write("Enter applicant details to evaluate loan default probability.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 25)
    income = st.number_input("Annual Income ($)", 1000, 500000, 50000)
    emp_len = st.number_input("Employment Duration (Years)", 0, 50, 3)
    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])

with col2:
    loan_amt = st.number_input("Loan Amount ($)", 500, 50000, 10000)
    int_rate = st.number_input("Interest Rate (%)", 5.0, 30.0, 11.0)
    grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    hist_default = st.selectbox("Historical Default on File", ["Y", "N"])
    cred_hist_len = st.number_input("Credit History Length (Years)", 0, 40, 4)

pct_income = loan_amt / max(income, 1)

if st.button("Evaluate Credit Risk"):
    payload = {
        "person_age": age,
        "person_income": income,
        "person_emp_length": emp_len,
        "loan_intent": intent,
        "loan_grade": grade,
        "loan_amnt": loan_amt,
        "loan_int_rate": int_rate,
        "loan_percent_income": pct_income,
        "cb_person_default_on_file": hist_default,
        "cb_person_cred_hist_length": cred_hist_len,
        "person_home_ownership": home_ownership
    }
    
    # Send request to local or deployed API
    res = requests.post("http://127.0.0.1:8000/predict", json=payload)
    result = res.json()
    
    if result["default_probability"] > 0.5:
        st.error(f"⚠️ Result: {result['default_risk']} | Default Probability: {result['default_probability']*100:.1f}%")
    else:
        st.success(f"✅ Result: {result['default_risk']} | Default Probability: {result['default_probability']*100:.1f}%")