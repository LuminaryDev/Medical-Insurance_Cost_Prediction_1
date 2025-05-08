import streamlit as st
import pandas as pd
import joblib

# Load the trained model
try:
    final_model = joblib.load("optimized_random_forest_model.pkl")
    encoders = joblib.load("encoders.pkl")  # Load encoders for categorical columns
except FileNotFoundError:
    st.error("Model or encoder file not found.")
    st.stop()

st.title("🏥 Medical Insurance Cost Prediction App")
st.write("Enter the following information to predict the insurance cost:")

# Collect user input
age = st.number_input("Age", min_value=18, max_value=100, step=1)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=1.0, max_value=50.0)
children = st.selectbox("Number of Children", list(range(0, 11)))  # better UX than slider
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

if st.button("Predict"):
    # Create DataFrame
    input_data = {
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    }
    input_df = pd.DataFrame(input_data)

    # Encode categorical features
    try:
        for col in ["sex", "smoker", "region"]:
            input_df[col] = encoders[col].transform(input_df[col])
    except Exception as e:
        st.error(f"Encoding error: {e}")
        st.stop()

    # Predict
    try:
        prediction = final_model.predict(input_df)[0]
        st.success(f"💰 Estimated Insurance Cost: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
