import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
model = joblib.load('trained_model.pkl')
scaler = joblib.load('scaler.pkl')

# Set background image (optional)
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.pinimg.com/736x/3d/c5/e0/3dc5e0f10680796810c68917f44c9c9a.jpg");
    background-size: cover;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
</style>
"""
st.markdown("""
<style>
.sticky-title {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    background-color: black;
    padding: 1rem;
    z-index: 999;
    border-bottom: 1px solid #ccc;
}

.sticky-spacer {
    height: 100px;
}
</style>
""", unsafe_allow_html=True)

app = st.selectbox("Select App", ["House Price Predictor", "Loan Approval Predictor"])
st.markdown(page_bg_img, unsafe_allow_html=True)

if app == "House Price Predictor":
    st.subheader("House Price Prediction")


    st.markdown('<div class="sticky-title"><h1>House Price Predictor</h1><p>Enter house features below to predict the <strong>median home price</strong> in $1000s.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="sticky-spacer"></div>', unsafe_allow_html=True)

    CRIM = st.number_input("CRIM", value=0.1, help="Per capita crime rate by town")
    ZN = st.number_input("ZN", value=18.0, help="Proportion of residential land zoned for large lots")
    INDUS = st.number_input("INDUS", value=2.31, help="Proportion of non-retail business acres")
    CHAS = st.selectbox("CHAS", options=[0, 1], help="1 if tract bounds river; 0 otherwise")
    NOX = st.number_input("NOX", value=0.538, help="Nitric oxide concentration (parts per 10 million)")
    RM = st.number_input("RM", value=6.0, help="Average number of rooms per house")
    AGE = st.number_input("AGE", value=65.2, help="% of owner-occupied units built before 1940")
    DIS = st.number_input("DIS", value=4.1, help="Weighted distances to employment centers")
    RAD = st.number_input("RAD", value=1, help="Index of accessibility to highways")
    TAX = st.number_input("TAX", value=296, help="Full-value property-tax rate per $10,000")
    PTRATIO = st.number_input("PTRATIO", value=15.3, help="Pupil-teacher ratio by town")
    B = st.number_input("B", value=396.9, help="1000(Bk - 0.63)^2 where Bk is % of Black population")
    LSTAT = st.number_input("LSTAT", value=5.0, help="% of lower status population")


    if st.button("Predict Price"):
        
        input_data = [[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS,RAD, TAX, PTRATIO, B, LSTAT]]

        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_data)

        prediction = model.predict(input_scaled)
        price = prediction[0] * 1000 
    
        st.success(f"Estimated House Price: ${price:,.2f}")
        st.markdown("---")
        st.markdown('Model predicts the median house price based on input features.', unsafe_allow_html=True)
        
        
elif app == "Loan Approval Predictor":
    st.subheader("Loan Approval Prediction")

    loan_model = joblib.load("loan_model.pkl")
    label_encoders = joblib.load("loan_label_encoders.pkl")


    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", value=5000)
    CoapplicantIncome = st.number_input("Coapplicant Income", value=2000)
    LoanAmount = st.number_input("Loan Amount", value=150)
    Loan_Amount_Term = st.number_input("Loan Term (in days)", value=360)
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    if st.button("Check Loan Approval"):
      
        input_df = pd.DataFrame({
            "Gender": [Gender],
            "Married": [Married],
            "Dependents": [Dependents],
            "Education": [Education],
            "Self_Employed": [Self_Employed],
            "ApplicantIncome": [ApplicantIncome],
            "CoapplicantIncome": [CoapplicantIncome],
            "LoanAmount": [LoanAmount],
            "Loan_Amount_Term": [Loan_Amount_Term],
            "Credit_History": [Credit_History],
            "Property_Area": [Property_Area]
        })
        for col in input_df.columns:
            if col in label_encoders:
                le = label_encoders[col]
                input_df[col] = le.transform(input_df[col])

        prediction = loan_model.predict(input_df)[0]
        status = "Approved" if prediction == 1 else "Not Approved"
        st.success(f"Loan Status: {status}")        

