import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("salary_model.pkl")
encoders = joblib.load("salary_model_compressed.pkl")

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction App")

# Input fields
age = st.slider("Age", 18, 80, 30)
workclass = st.selectbox("Workclass", encoders['workclass'].classes_)
fnlwgt = st.number_input("Final Weight (fnlwgt)", value=200000)
education = st.selectbox("Education", encoders['education'].classes_)
educational_num = st.slider("Educational Number", 1, 16, 10)
marital_status = st.selectbox("Marital Status", encoders['marital-status'].classes_)
occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
relationship = st.selectbox("Relationship", encoders['relationship'].classes_)
race = st.selectbox("Race", encoders['race'].classes_)
gender = st.selectbox("Gender", encoders['gender'].classes_)
capital_gain = st.number_input("Capital Gain", 0, 99999, step=100)
capital_loss = st.number_input("Capital Loss", 0, 4356, step=50)
hours_per_week = st.slider("Hours per Week", 1, 100, 40)
native_country = st.selectbox("Native Country", encoders['native-country'].classes_)

# Prepare input
input_dict = {
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt,
    'education': education,
    'educational-num': educational_num,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'gender': gender,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}
input_df = pd.DataFrame([input_dict])

# Encode input
for col in input_df.columns:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

# Predict
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)[0]
    income_label = encoders['income'].inverse_transform([prediction])[0]
    st.success(f"🔮 Predicted Salary: **{income_label}**")
