import streamlit as st
import numpy as np
import joblib

st.title("💳 Credit Card Fraud Detection App")

# Select model
model_choice = st.selectbox("Select Model", ["Baseline Model", "Fine-tuned Model"])
model_path = "baseline_model.pkl" if model_choice == "Baseline Model" else "tuned_model.pkl"

# Load model
model = joblib.load(model_path)

# Inputs
st.header("Enter Transaction Features")

time = st.number_input("Time", min_value=0.0, step=1.0)
amount = st.number_input("Amount", min_value=0.0, step=0.1)

v_features = []
for i in range(1, 29):
    v = st.number_input(f"V{i}", value=0.0, step=0.1)
    v_features.append(v)

# Extra engineered features
hour = st.number_input("Hour", min_value=0, max_value=23, step=1)
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
hour_sin = st.number_input("Hour_Sin", value=0.0, step=0.01)
hour_cos = st.number_input("Hour_Cos", value=0.0, step=0.01)
time_diff = st.number_input("Time_Diff", value=0.0, step=0.1)

# Encode time_of_day to numeric (manual example)
time_of_day_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
time_of_day_encoded = time_of_day_map[time_of_day]

# Combine all features
features = [time] + v_features + [amount, hour, time_of_day_encoded, hour_sin, hour_cos, time_diff]
features_array = np.array(features).reshape(1, -1)

# Predict
if st.button("Predict Fraud"):
    prediction = model.predict(features_array)[0]
    result = "🚨 Fraud Detected!" if prediction == 1 else "✅ Transaction is Legitimate"
    st.subheader(result)
