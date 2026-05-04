import streamlit as st
import joblib
import pandas as pd
model = joblib.load("models/xgboost_44features.pkl")
X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = joblib.load("data/processed/44features.pkl")
st.title("Fintech Credit Scoring System")
limit_bal = st.number_input("Credit Limit", min_value=0)
age = st.number_input("Age", min_value=18, max_value=100)
pay_0 = st.selectbox("Latest Repayment Status", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])

input_data = pd.DataFrame([{
"X1": limit_bal,
"X5": age,
"X6": pay_0,
}])

if st.button("Predict Default Risk"):
    processed = preprocessor.transform(input_data)
    prob = model.predict_proba(processed)[0][1]
    st.write(f"Default Probability: {prob:.2%}")
    if prob < 0.2:
        st.success("Low Risk")
    elif prob < 0.5:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")
