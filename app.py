import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved preprocessor and model
preprocessor = joblib.load('preprocessor_gdm.pkl')
model = joblib.load('best_logistic_regression_model.pkl')

st.title("Gestational Diabetes Prediction Tool")

# Capture unique identifier (optional, used for tracking purposes)
study_id = st.text_input('Study Participant ID', '')

# Ethnic origin selection (stored as object in model)
ethnic_origin = st.selectbox('Ethnic Origin of Patient', 
                             ['CAUCASIAN', 'SOUTH EAST ASIAN', 'OTHER', 'BLACK', 'ASIAN', 'MIDDLE EASTERN'])

# Age at booking (stored as int64)
age_at_booking = st.number_input('Age at Booking', min_value=18, max_value=50, step=1)

# Skill level (stored as int64)
skill_level_display = st.selectbox('Skill Level', 
                                   ['0 - Unemployed', 
                                    '1 - Elementary occupations', 
                                    '2 - Clerical support workers/Skilled workers/Assemblers',
                                    '3 - Technicians', 
                                    '4 - Managers and Professionals'])
skill_level = int(skill_level_display.split(" ")[0])

# Hx_GDM (stored as int64: 1 for YES, 0 for NO)
hx_gdm = st.selectbox('History of Gestational Diabetes (Hx_GDM)', ['YES', 'NO'])
hx_gdm_numeric = 1 if hx_gdm == 'YES' else 0

# BMI (stored as float64)
bmi = st.number_input('BMI', min_value=15.0, max_value=50.0, step=0.1)

# FH Diabetes (stored as object)
fh_diabetes = st.selectbox('Family History of Diabetes (FH Diabetes)', ['YES', 'NO'])

# Other Endocrine problems (stored as int64: 1 for YES, 0 for NO)
other_endocrine_probs = st.selectbox('Other Endocrine Problems', ['YES', 'NO'])
other_endocrine_probs_numeric = 1 if other_endocrine_probs == 'YES' else 0

# Systolic BP at booking (stored as int64)
systolic_bp = st.number_input('Systolic Blood Pressure at Booking', min_value=20, max_value=200, step=1)

# Diastolic BP at booking (stored as float64)
diastolic_bp = st.number_input('Diastolic Blood Pressure at Booking', min_value=20, max_value=200, step=1)

# Parity (stored as int64)
parity = st.number_input('Parity (excluding multiple)', min_value=0, max_value=20, step=1)

# Button to make prediction
if st.button('Predict Gestational Diabetes'):
    # Create a DataFrame for the input data with the correct format and column names
    input_data = pd.DataFrame({
        'Ethnic Origin of Patient': [ethnic_origin],        # object
        'Age at booking': [age_at_booking],                # int64
        'Skill Level': [skill_level],                      # int64
        'Hx_GDM': [hx_gdm_numeric],                        # int64
        'BMI': [bmi],                                     # float64
        'FH Diabetes': [fh_diabetes],                      # object
        'Other Endocrine probs': [other_endocrine_probs_numeric],  # int64
        'Systolic BP at booking': [systolic_bp],           # int64
        'Diastolic BP at booking': [diastolic_bp],         # float64
        'Parity (not inc.multiple)': [parity]              # int64
    })
    
    # Apply the saved preprocessor to the input data
    try:
        input_data_processed = preprocessor.transform(input_data)

        # Make prediction using the saved model
        prediction = model.predict(input_data_processed)

        # Display the result
        if prediction == 1:
            st.error(f'Prediction for Study ID {study_id}: HIGH Risk of Gestational Diabetes')
            st.write("This result indicates a high risk of developing gestational diabetes. Flag for OGTT at week 16")
        else:
            st.success(f'Prediction for Study ID {study_id}: LOW Risk of Gestational Diabetes')
            st.write("This result indicates a low risk of developing gestational diabetes. Please follow regular prenatal care.")
    
    except Exception as e:
        st.write(f"Error during preprocessing or prediction: {str(e)}")
