import streamlit as st
import joblib
import pandas as pd

model = joblib.load('salary_model.pkl')

st.title("💼 Data Science Salary Predictor")
st.write("Estimate a data science salary based on role details.")

st.header("Enter Job Details")

experience_level = st.selectbox("Experience Level", ["EN", "MI", "SE", "EX"])
employment_type = st.selectbox("Employment Type", ["FT", "PT", "CT", "FL"])
job_title = st.selectbox("Job Title", ["Data Scientist", "Data Engineer", "Data Analyst", 
                                          "Machine Learning Engineer", "Research Scientist", 
                                          "Data Science Manager", "Data Architect", "Other"])
company_size = st.selectbox("Company Size", ["S", "M", "L"])
company_location = st.selectbox("Company Location", ["US", "GB", "IN", "CA", "DE", "Other"])
remote_ratio = st.selectbox("Remote Ratio", [0, 50, 100])
work_year = st.number_input("Work Year", min_value=2020, max_value=2026, value=2024)

if st.button("Predict Salary"):
    # Build input row matching training columns
    input_dict = {col: 0 for col in model.feature_names_in_}
    input_dict['work_year'] = work_year
    input_dict['remote_ratio'] = remote_ratio
    
    exp_col = f'experience_level_{experience_level}'
    if exp_col in input_dict:
        input_dict[exp_col] = 1
    
    emp_col = f'employment_type_{employment_type}'
    if emp_col in input_dict:
        input_dict[emp_col] = 1
    
    title_col = f'job_title_{job_title}'
    if title_col in input_dict:
        input_dict[title_col] = 1
    
    size_col = f'company_size_{company_size}'
    if size_col in input_dict:
        input_dict[size_col] = 1
    
    loc_col = f'company_location_{company_location}'
    if loc_col in input_dict:
        input_dict[loc_col] = 1
    
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted salary: ${prediction:,.0f}")
