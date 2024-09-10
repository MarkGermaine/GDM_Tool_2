import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved preprocessor and model
preprocessor = joblib.load('preprocessor_gdm.pkl')
model = joblib.load('best_logistic_regression_model.pkl')

st.title("Gestational Diabetes Prediction Tool")

# Capture unique identifier (optional, used for tracking purposes)
study_id = st.text_input('Study Participant ID', '')

# Ethnic origin selection
ethnic_origin = st.selectbox('Ethnic Origin of Patient', 
                             ['CAUCASIAN', 'SOUTH EAST ASIAN', 'OTHER', 'BLACK', 'ASIAN', 'MIDDLE EASTERN'])

# Age at booking (integer)
age_at_booking = st.number_input('Age at Booking', min_value=18, max_value=50, step=1)

# Skill level (0-4 with description)
skill_level = st.selectbox('Skill Level', 
                           ['0 - Unemployed', 
                            '1 - Elementary occupations', 
                            '2 - Clerical support workers/Skilled workers/Assemblers',
                            '3 - Technicians', 
                            '4 - Managers and Professionals'])

# Hx_GDM (YES/NO)
hx_gdm = st.selectbox('History of Gestational Diabetes (Hx_GDM)', ['YES', 'NO'])

# BMI (float)
bmi = st.number_input('BMI', min_value=15.0, max_value=50.0, step=0.1)

# FH Diabetes (YES/NO)
fh_diabetes = st.selectbox('Family History of Diabetes (FH Diabetes)', ['YES', 'NO'])

# Other Endocrine problems (YES/NO)
other_endocrine_probs = st.selectbox('Other Endocrine Problems', ['YES', 'NO'])

# Systolic BP at booking (integer)
systolic_bp = st.number_input('Systolic Blood Pressure at booking', min_value=20, max_value=200, step=1)

# Diastolic BP at booking (integer)
diastolic_bp = st.number_input('Diastolic Blood Pressure at booking', min_value=20, max_value=200, step=1)

# Parity (integer)
parity = st.number_input('Parity (excluding multiple)', min_value=0, max_value=20, step=1)

# Button to make prediction
if st.button('Predict Gestational Diabetes'):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'Ethnic Origin of Patient': [ethnic_origin],
        'FH Diabetes': [fh_diabetes],
        'Age at booking': [age_at_booking],
        'Skill Level': [skill_level],
        'Hx_GDM': [hx_gdm],
        'BMI': [bmi],
        'Other Endocrine probs': [other_endocrine_probs],
        'Systolic BP at booking': [systolic_bp],  # Matching column name
        'Diastolic BP at booking': [diastolic_bp],  # Matching column name
        'Parity (not inc.multiple)': [parity]
    })
    
    # Apply the saved preprocessor to the input data
    input_data_processed = preprocessor.transform(input_data)

    # Make prediction using the saved model
    prediction = model.predict(input_data_processed)

    # Display the result
    st.write(f'Prediction for Study ID {study_id}: {"Gestational Diabetes" if prediction == 1 else "No Gestational Diabetes"}')
