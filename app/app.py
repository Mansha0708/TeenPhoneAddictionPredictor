import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt # New import for Matplotlib
import seaborn as sns # New import for Seaborn

# Set Streamlit page configuration
st.set_page_config(page_title="Teen Phone Addiction Predictor", layout="centered", initial_sidebar_state="expanded")

# --- Load the Trained Model ---
# Ensure 'model/addiction_model.pkl' is in the correct path relative to your app.py
try:
    model = joblib.load("model/addiction_model.pkl")
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model file 'addiction_model.pkl' not found. Please ensure it's in the 'model/' directory.")
    st.stop() # Stop the app if model is not found

# --- Streamlit UI ---
st.title("📱 Teen Phone Addiction Predictor")
st.markdown("---")
st.write("Enter the details below to estimate a teen's phone addiction level (0–10).")

# Create columns for better layout
col1, col2 = st.columns(2)

# --- Collect User Inputs ---
# These inputs will be used to construct the 35 features for the model.

with col1:
    st.header("Demographics & Core Usage")
    age = st.slider("Age", 10, 19, 14)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"]) # Order matters for consistent one-hot encoding
    school_grade = st.selectbox("School Grade", ["6th", "7th", "8th", "9th", "10th", "11th", "12th", "Other"]) # Comprehensive list
    daily_usage_hours = st.slider("Daily Phone Usage (hours)", 0.0, 24.0, 5.0, 0.1)
    sleep_hours = st.slider("Sleep per Day (hours)", 0.0, 12.0, 7.0, 0.1)
    screen_time_before_bed = st.slider("Screen Time Before Bed (hours)", 0.0, 5.0, 1.0, 0.1)
    phone_checks_per_day = st.slider("Phone Checks per Day", 0, 300, 50)
    apps_used_daily = st.slider("Apps Used Daily", 0, 50, 10)
    self_esteem = st.slider("Self-Esteem (1-10)", 1, 10, 5)
    parental_control = st.slider("Parental Control (1-10)", 1, 10, 5)


with col2:
    st.header("Behavioral & Social Factors")
    time_on_social_media = st.slider("Time on Social Media (hours)", 0.0, 10.0, 2.0, 0.1)
    time_on_gaming = st.slider("Time on Gaming (hours)", 0.0, 10.0, 1.0, 0.1)
    time_on_education = st.slider("Time on Education (hours)", 0.0, 10.0, 2.0, 0.1)
    academic_performance = st.slider("Academic Performance (0-100)", 0, 100, 75)
    social_interactions = st.slider("Social Interactions (1-10)", 1, 10, 5)
    exercise_hours = st.slider("Exercise Hours (daily average)", 0.0, 3.0, 0.5, 0.1)
    anxiety_level = st.slider("Anxiety Level (0-10)", 0, 10, 3)
    depression_level = st.slider("Depression Level (0-10)", 0, 10, 2)
    family_communication = st.slider("Family Communication (1-10)", 1, 10, 7)
    weekend_usage_hours = st.slider("Weekend Usage (hours/day)", 0.0, 24.0, 6.0, 0.1)
    phone_usage_purpose = st.selectbox("Primary Phone Usage Purpose", ["Communication", "Education", "Entertainment", "Gaming", "Other", "Social Media"]) # Comprehensive list


st.markdown("---")

# --- Prediction Button ---
if st.button("Predict Addiction Level", type="primary"):
    # 1. Create a temporary DataFrame for one-hot encoding
    # This ensures consistency with pd.get_dummies used during training
    input_df_temp = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'School_Grade': school_grade,
        'Daily_Usage_Hours': daily_usage_hours,
        'Sleep_Hours': sleep_hours,
        'Academic_Performance': academic_performance,
        'Social_Interactions': social_interactions,
        'Exercise_Hours': exercise_hours,
        'Anxiety_Level': anxiety_level,
        'Depression_Level': depression_level,
        'Self_Esteem': self_esteem,
        'Parental_Control': parental_control,
        'Screen_Time_Before_Bed': screen_time_before_bed,
        'Phone_Checks_Per_Day': phone_checks_per_day,
        'Apps_Used_Daily': apps_used_daily,
        'Time_on_Social_Media': time_on_social_media,
        'Time_on_Gaming': time_on_gaming,
        'Time_on_Education': time_on_education,
        'Family_Communication': family_communication,
        'Weekend_Usage_Hours': weekend_usage_hours,
        'Phone_Usage_Purpose': phone_usage_purpose
    }])

    # Define categorical columns for one-hot encoding (must match training script)
    categorical_cols_for_ohe = ['Gender', 'School_Grade', 'Phone_Usage_Purpose']

    # Perform One-Hot Encoding (matching training script's drop_first=True)
    # Ensure all possible categories are known to pd.get_dummies to create consistent columns
    # This is a critical step to ensure the same columns are generated as during training.
    # We need to explicitly define all possible categories for each OHE column
    # to prevent missing columns if a specific category isn't selected by the user.
    # The categories here must match the ones used during training for pd.get_dummies
    # If your training data had other categories, you might need to add them here.
    # For simplicity, we'll assume the categories listed in the selectboxes cover all.

    # Create dummy variables with known categories to ensure consistent columns
    input_df_encoded = pd.get_dummies(
        input_df_temp,
        columns=categorical_cols_for_ohe,
        drop_first=True,
        dtype=int # Ensure boolean values are converted to 0/1 integers
    )

    # Manually add any missing one-hot encoded columns that might not have been created
    # if the user didn't select a category that would generate them.
    # This list must exactly match the columns generated by your training script's pd.get_dummies
    # based on the 35 features you provided.
    final_expected_ohe_cols = [
        'Gender_Male', 'Gender_Other',
        'School_Grade_11th', 'School_Grade_12th', 'School_Grade_7th',
        'School_Grade_8th', 'School_Grade_9th', # Assuming '6th' or '10th' was dropped
        'Phone_Usage_Purpose_Education', 'Phone_Usage_Purpose_Gaming',
        'Phone_Usage_Purpose_Other', 'Phone_Usage_Purpose_Social Media' # Assuming 'Communication' or 'Entertainment' was dropped
    ]

    for col in final_expected_ohe_cols:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0 # Add missing OHE columns with 0

    # 2. Calculate Engineered Features
    # Ensure +1 to avoid division by zero for ratios, as done in training
    night_usage = daily_usage_hours - screen_time_before_bed
    social_to_edu_ratio = (time_on_social_media + 1) / (time_on_education + 1)
    gaming_to_social_ratio = (time_on_gaming + 1) / (time_on_social_media + 1)
    weekend_overuse = weekend_usage_hours - daily_usage_hours
    phone_obsessiveness = phone_checks_per_day / (apps_used_daily + 1)
    sleep_deficit = 8 - sleep_hours # Assuming 8 hours is ideal sleep

    # Add engineered features to the DataFrame
    input_df_encoded['Night_Usage'] = night_usage
    input_df_encoded['Social_to_Edu_Ratio'] = social_to_edu_ratio
    input_df_encoded['Gaming_to_Social_Ratio'] = gaming_to_social_ratio
    input_df_encoded['Weekend_Overuse'] = weekend_overuse
    input_df_encoded['Phone_Obsessiveness'] = phone_obsessiveness
    input_df_encoded['Sleep_Deficit'] = sleep_deficit

    # 3. Reorder columns to match the exact order expected by the model (35 features)
    # This list MUST EXACTLY match the X.columns.tolist() output from your Jupyter Notebook training script.
    # Based on the user's provided list of 35 columns:
    expected_model_features_order = [
        'Age', 'Daily_Usage_Hours', 'Sleep_Hours', 'Academic_Performance',
        'Social_Interactions', 'Exercise_Hours', 'Anxiety_Level',
        'Depression_Level', 'Self_Esteem', 'Parental_Control',
        'Screen_Time_Before_Bed', 'Phone_Checks_Per_Day', 'Apps_Used_Daily',
        'Time_on_Social_Media', 'Time_on_Gaming', 'Time_on_Education',
        'Family_Communication', 'Weekend_Usage_Hours',
        'Gender_Male', 'Gender_Other',
        'School_Grade_11th', 'School_Grade_12th', 'School_Grade_7th',
        'School_Grade_8th', 'School_Grade_9th',
        'Phone_Usage_Purpose_Education', 'Phone_Usage_Purpose_Gaming',
        'Phone_Usage_Purpose_Other', 'Phone_Usage_Purpose_Social Media',
        'Night_Usage', 'Social_to_Edu_Ratio', 'Gaming_to_Social_Ratio',
        'Weekend_Overuse', 'Phone_Obsessiveness', 'Sleep_Deficit'
    ]

    # Ensure all expected columns are present, fill with 0 if not (e.g., if a specific OHE column wasn't created)
    # This is a final safety check for column consistency before prediction.
    for col in expected_model_features_order:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0

    # Reorder the DataFrame columns
    input_final = input_df_encoded[expected_model_features_order]

    # --- Removed Debugging Info ---
    # st.write("---")
    # st.subheader("Input Data for Prediction (Debugging Info):")
    # st.write(f"Shape: {input_final.shape}")
    # st.write("Columns in order:", input_final.columns.tolist())
    # st.write(input_final)
    # st.write("---")


    # Make prediction
    try:
        prediction = model.predict(input_final)[0]
        st.subheader("📊 Predicted Addiction Level:")
        st.success(f"**{prediction:.2f} / 10**")

        # Provide feedback based on prediction
        st.markdown("---")
        if prediction > 7.5:
            st.error("🚨 **High Risk!** This teen shows signs of high phone addiction. Consider immediate intervention and professional help.")
            st.info("💡 **Suggestions:** Implement strict screen time limits, encourage alternative activities, seek professional counseling, and monitor digital well-being tools.")
        elif prediction > 5.0:
            st.warning("⚠️ **Moderate Risk.** There's a noticeable risk of phone addiction. Encourage healthier habits.")
            st.info("💡 **Suggestions:** Promote balanced screen time, encourage outdoor activities and face-to-face interactions, set clear boundaries, and discuss digital etiquette.")
        else:
            st.success("✅ **Low Risk!** This teen shows healthy phone usage habits. Keep up the good work!")
            st.info("💡 **Suggestions:** Continue reinforcing positive habits, maintain open communication about online safety, and encourage diverse interests.")

        st.markdown("---")
        st.subheader("Visualizing Your Inputs")

        # --- Visual 1: Daily Usage Hours Distribution (Example) ---
        # For a real distribution, you'd need a dataset. Here, we'll just plot the input value.
        # A more meaningful plot would compare the input to a known distribution from your training data.
        st.write("#### Daily Phone Usage")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot([daily_usage_hours], bins=[0, 5, 10, 15, 20, 25], kde=False, color='skyblue', ax=ax)
        ax.set_title("Your Daily Phone Usage (Hours)")
        ax.set_xlabel("Hours")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig) # Close the figure to prevent memory issues

        # --- Visual 2: Predicted Addiction Level Bar (More Dynamic) ---
        st.write("#### Predicted Addiction Level Meter")
        # Create a simple bar to represent the addiction level
        addiction_score = int(prediction * 10) # Scale to 0-100 for percentage bar
        st.progress(addiction_score)
        st.write(f"Your predicted score is **{prediction:.2f} / 10**")


    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure your 'addiction_model.pkl' was trained with the exact same 35 features and order as expected by this app.")

st.markdown("---")
st.info("This prediction is an estimate. For a comprehensive assessment, consult with a professional.")

