import streamlit as st
import pandas as pd
import joblib

# Load the trained model
try:
    final_model = joblib.load("optimized_random_forest_model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Ensure 'optimized_random_forest_model.pkl' exists.")
    st.stop()

# Load the encoder for 'region' (OneHotEncoder)
try:
    region_encoder = joblib.load("region_encoder.pkl")
except FileNotFoundError:
    st.error("Encoder file not found. Ensure 'region_encoder.pkl' exists.")
    st.stop()

st.title("🏥 Medical Insurance Cost Prediction App")

st.write("Enter the following information to predict the insurance cost:")

# Collect user input
age = st.number_input("Age", min_value=18, max_value=100, step=1)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=1.0, max_value=50.0)
children = st.number_input("Number of Children", 0, 15)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# Convert user input to DataFrame for prediction
if st.button("Predict"):
    input_data = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }

    input_df = pd.DataFrame([input_data])

    # Binary Encoding for 'sex' and 'smoker'
    input_df['sex'] = input_df['sex'].map({'female': 1, 'male': 0})
    input_df['smoker'] = input_df['smoker'].map({'yes': 1, 'no': 0})

    # One-Hot Encoding for 'region' using the saved encoder
    region_encoded = region_encoder.transform(input_df[['region']])
    region_columns = [f"region_{category}" for category in region_encoder.categories_[0][1:]]
    region_df = pd.DataFrame(region_encoded, columns=region_columns, index=input_df.index)

    # Combine the encoded columns back into the input DataFrame
    input_df = pd.concat([input_df.drop('region', axis=1), region_df], axis=1)

    # Make prediction
    prediction = final_model.predict(input_df)[0]
    st.success(f"💰 Estimated Insurance Cost: ${prediction:,.2f}")
