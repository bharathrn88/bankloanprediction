import streamlit as st
import numpy as np
import pickle

# Set page configuration
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

# Load the trained model and scaler
@st.cache_resource
def load_model():
    with open("loan_approval_model.pkl", "rb") as file:
        return pickle.load(file)

@st.cache_resource
def load_scaler():
    with open("loan_scaler.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()
scaler = load_scaler()

# App title
st.title("Bank Loan Prediction")
st.write("Fill in the details below to check your loan eligibility.")

# User input fields
st.sidebar.header("User Input Features")
no_of_dependents = st.sidebar.selectbox("Number of Dependents", [0, 1, 2, 3, 4, 5, 6, 7])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.sidebar.number_input("Annual Income", min_value=0.0, value=500.0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0, value=1000.0)
loan_term = st.sidebar.number_input("Loan Term (in months)", min_value=0.0, value=360.0)
cibil_score = st.sidebar.number_input("CIBIL Score", min_value=0, max_value=900, value=750)
Assets = st.sidebar.number_input("Total Assets Value", min_value=0.0, value=10000.0)

# Convert categorical inputs to numerical format
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# Create input array
input_data = np.array([[no_of_dependents, education, self_employed, income_annum,
                        loan_amount, loan_term, cibil_score, Assets]])

# Ensure input matches expected features by the scaler
try:
    input_data_scaled = scaler.transform(input_data)

    # Predict loan approval
    if st.button("Predict"):
        prediction = model.predict(input_data_scaled)[0]
        probability = model.predict_proba(input_data_scaled)[:, 1][0] * 100
        
        if prediction == 1:
            result_message = "You can apply for the loan!"
        else:
            result_message = "Unfortunately, You Can't Apply for Loan."
        
        st.success(f"{result_message}\nApproval Probability: {probability:.2f}%")

except Exception as e:
    st.error(f"An error occurred: {e}. Please check your inputs.")
