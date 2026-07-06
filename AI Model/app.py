import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("best_mental_health_model.pkl")

st.set_page_config(
    page_title="Mental Health Risk Predictor",
    layout="wide"
)

st.title("🧠 Mental Health Risk Predictor")
st.write(
    "Predict mental health risk based on digital lifestyle and social media behavior."
)

# =========================
# Numerical Inputs
# =========================
age = st.number_input("Age", 10, 80, 22)

device_hours_per_day = st.slider(
    "Device Hours Per Day",
    0.0, 24.0, 8.0
)

phone_unlocks = st.number_input(
    "Phone Unlocks Per Day",
    0, 500, 100
)

notifications_per_day = st.number_input(
    "Notifications Per Day",
    0, 1000, 150
)

social_media_mins = st.number_input(
    "Social Media Minutes",
    0, 1000, 180
)

study_mins = st.number_input(
    "Study Minutes",
    0, 1000, 120
)

physical_activity_days = st.slider(
    "Physical Activity Days Per Week",
    0, 7, 3
)

sleep_hours = st.slider(
    "Sleep Hours",
    0.0, 12.0, 7.0
)

focus_score = st.slider(
    "Focus Score",
    0.0, 10.0, 5.0
)

productivity_score = st.slider(
    "Productivity Score",
    0.0, 10.0, 5.0
)

digital_dependence_score = st.slider(
    "Digital Dependence Score",
    0.0, 10.0, 5.0
)

digital_wellness_score = st.slider(
    "Digital Wellness Score",
    0.0, 10.0, 5.0
)

# =========================
# Categorical Inputs
# =========================
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age_group = st.selectbox(
    "Age Group",
    ["18-24", "25-34", "35-44", "45-54", "55+"]
)

income_level = st.selectbox(
    "Income Level",
    ["Low", "Medium", "High"]
)

education_level = st.selectbox(
    "Education Level",
    ["High School", "Bachelor", "Master", "PhD"]
)

daily_role = st.selectbox(
    "Daily Role",
    ["Student", "Employee", "Freelancer", "Unemployed"]
)

device_type = st.selectbox(
    "Device Type",
    ["Smartphone", "Laptop", "Tablet"]
)

screen_time_category = st.selectbox(
    "Screen Time Category",
    ["Low", "Moderate", "High"]
)

sleep_category = st.selectbox(
    "Sleep Category",
    ["Poor Sleep", "Normal Sleep", "Oversleep"]
)

productivity_category = st.selectbox(
    "Productivity Category",
    ["Low", "Moderate", "Excellent"]
)

region = st.selectbox(
    "Region",
    ["Africa", "Asia", "Europe", "North America", "South America"]
)

# =========================
# Prediction Button
# =========================
if st.button("Predict Risk"):

    input_data = pd.DataFrame({
        "age": [age],
        "device_hours_per_day": [device_hours_per_day],
        "phone_unlocks": [phone_unlocks],
        "notifications_per_day": [notifications_per_day],
        "social_media_mins": [social_media_mins],
        "study_mins": [study_mins],
        "physical_activity_days": [physical_activity_days],
        "sleep_hours": [sleep_hours],
        "focus_score": [focus_score],
        "productivity_score": [productivity_score],
        "digital_dependence_score": [digital_dependence_score],
        "digital_wellness_score": [digital_wellness_score],
        "gender": [gender],
        "age_group": [age_group],
        "income_level": [income_level],
        "education_level": [education_level],
        "daily_role": [daily_role],
        "device_type": [device_type],
        "screen_time_category": [screen_time_category],
        "sleep_category": [sleep_category],
        "productivity_category": [productivity_category],
        "region": [region]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("⚠️ High Mental Health Risk Detected")
    else:
        st.success("✅ Low Mental Health Risk")

    st.metric(
        label="Risk Probability",
        value=f"{probability * 100:.2f}%"
    )