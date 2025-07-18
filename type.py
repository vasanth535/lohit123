import streamlit as st
import pandas as pd
import joblib

# Load model safely
try:
    model = joblib.load("salary_model_compressed.pkl")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Title
st.title("Employee Salary Prediction")

# Input form
age = st.number_input("Age", min_value=18, max_value=100, value=30)
education = st.selectbox("Education", ["Bachelors", "HS-grad", "Masters", "Doctorate", "Some-college"])
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)
fnlwgt = st.number_input("FNLWGT", min_value=1, value=100000)
educational_num = st.number_input("Educational-Num", min_value=1, value=13)
workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Federal-gov"])

# Create input DataFrame
input_df = pd.DataFrame([{
    "age": age,
    "education": education,
    "capital-gain": capital_gain,
    "capital-loss": capital_loss,
    "hours-per-week": hours_per_week,
    "fnlwgt": fnlwgt,
    "educational-num": educational_num,
    "workclass": workclass
}])

# Predict button
if st.button("Predict"):
    if hasattr(model, "predict"):
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"🔮 Predicted Salary Class: {prediction}")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.error("⚠️ Loaded object is not a valid model.")
