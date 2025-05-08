import streamlit as st
import pandas as pd
import joblib

# Load the trained model
try:
    final_model = joblib.load("optimized_random_forest_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file not found. Ensure 'optimized_random_forest_model.pkl' exists.")
    st.stop()

st.title("🏥 Medical Insurance Cost Prediction App")

st.write("Enter the following information to predict the insurance cost:")

# Collect user input
age = st.number_input("Age", min_value=18, max_value=100, step=1)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0)
children = st.slider("Number of Children", 0, 5)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# Convert to DataFrame for prediction
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    # Preprocess categorical variables if necessary (depends on how you trained the model)
    # If you used one-hot encoding or LabelEncoder during training, apply the same here.

    prediction = final_model.predict(input_data)[0]
    st.success(f"💰 Estimated Insurance Cost: ${prediction:,.2f}")
