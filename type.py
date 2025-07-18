import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
try:
    model = joblib.load("salary_model_compressed.pkl")
    encoder_dict = joblib.load("label_encoders.pkl")  # this is a dictionary of mappings
except Exception as e:
    st.error(f"❌ Failed to load model or encoder: {e}")
    st.stop()

# Title
st.title("💼 Employee Salary Prediction")

# Inputs
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

# Manual encoding if encoder is a dictionary
try:
    for col in ["education", "workclass"]:
        if col in encoder_dict:
            input_df[col] = input_df[col].map(encoder_dict[col])
except Exception as e:
    st.error(f"⚠️ Encoding failed: {e}")
    st.stop()

# Display for debug
st.subheader("📋 Encoded Input Data")
st.write(input_df)

# Predict
if st.button("🔮 Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Predicted Salary Class: {prediction}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
