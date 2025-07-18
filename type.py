import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
try:
    model = joblib.load("salary_model_compressed.pkl")
    education_encoder = joblib.load("education_encoder.pkl")
    workclass_encoder = joblib.load("workclass_encoder.pkl")
except Exception as e:
    st.error(f"❌ Failed to load model or encoders: {e}")
    st.stop()

# Streamlit Title
st.title("💼 Employee Salary Prediction")

# Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
education = st.selectbox("Education", education_encoder.classes_.tolist())
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)
fnlwgt = st.number_input("FNLWGT", min_value=1, value=100000)
educational_num = st.number_input("Educational-Num", min_value=1, value=13)
workclass = st.selectbox("Workclass", workclass_encoder.classes_.tolist())

# Encode categorical fields
try:
    education_encoded = education_encoder.transform([education])[0]
    workclass_encoded = workclass_encoder.transform([workclass])[0]
except Exception as e:
    st.error(f"⚠️ Encoding failed: {e}")
    st.stop()

# Create input DataFrame
input_df = pd.DataFrame([{
    "age": age,
    "education": education_encoded,
    "capital-gain": capital_gain,
    "capital-loss": capital_loss,
    "hours-per-week": hours_per_week,
    "fnlwgt": fnlwgt,
    "educational-num": educational_num,
    "workclass": workclass_encoded
}])

# Show inputs
st.subheader("📋 Model Input")
st.write(input_df)

# Predict
if st.button("🔮 Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Predicted Salary Class: {prediction}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
