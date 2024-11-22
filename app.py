import pickle
import streamlit as st
import numpy as np
import os

# Load the logistic regression model
model_path = os.path.join(os.getcwd(), 'model_diabetes.sav')
model_logistic = pickle.load(open(model_path, 'rb'))

# App Title and Description
st.set_page_config(page_title="Diabetes Prediction", layout="wide")
st.title("Diabetes Prediction Application")
st.markdown("""
    Predict the likelihood of diabetes based on various health parameters. Enter the required details to get the prediction.
""")

# Sidebar Layout
st.sidebar.header("Enter Patient Information")
st.sidebar.subheader("Health Parameters")
Pregnancies = st.sidebar.number_input('Number of Pregnancies', min_value=0, step=1)
Glucose = st.sidebar.number_input('Glucose Level (mg/dL)', min_value=0.0, step=1.0)
BloodPressure = st.sidebar.number_input('Blood Pressure Level (mm Hg)', min_value=0.0, step=1.0)
SkinThickness = st.sidebar.number_input('Skin Thickness (mm)', min_value=0.0, step=1.0)
Insulin = st.sidebar.number_input('Insulin Level (mu U/mL)', min_value=0.0, step=1.0)
BMI = st.sidebar.number_input('BMI (Body Mass Index)', min_value=0.0, step=0.1)
DiabetesPedigreeFunction = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, step=0.01)
Age = st.sidebar.number_input('Age', min_value=0, step=1)

# Sidebar for Model Selection
st.sidebar.subheader("Model Selection")
model_choice = st.sidebar.radio(
    "Choose a Prediction Model",
    ['Logistic Regression', 'Random Forest', 'Gaussian']
)

# Prediction Button and Output
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("Prediction Results")
if st.button('Predict Diabetes'):
    # Select the model based on user choice
    if model_choice == 'Logistic Regression':
        model = model_logistic
    # Add other models if needed (e.g., Random Forest, Gaussian)
    # elif model_choice == 'Random Forest':
    #    model = model_random_forest
    # else:
    #    model = model_gaussian

    # Prepare input data for prediction
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    try:
        diabetes_prediction = model.predict(input_data)
        if diabetes_prediction[0] == 1:
            st.error("The patient is **likely to have diabetes**.", icon="🚨")
        else:
            st.success("The patient is **unlikely to have diabetes**.", icon="✅")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}", icon="⚠️")

# Additional Information Section
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("About Diabetes Prediction")
st.markdown("""
    This application uses various health parameters to predict whether a person is likely to develop diabetes. 
    The prediction is based on a logistic regression model, but other models can be selected for comparison.
    
    ### Key Health Parameters:
    - **Pregnancies**: Number of pregnancies the patient has had.
    - **Glucose Level**: Blood sugar level after fasting.
    - **Blood Pressure**: Pressure of the blood against the walls of the arteries.
    - **Skin Thickness**: Measurement of skinfold thickness.
    - **Insulin Level**: Amount of insulin in the blood.
    - **BMI**: Body Mass Index, an indicator of body fat based on height and weight.
    - **Diabetes Pedigree Function**: A function indicating the likelihood of diabetes based on family history.
    - **Age**: The patient's age.
    
    ### Model Selection:
    - **Logistic Regression**: A statistical method used to model the relationship between a dependent variable and one or more independent variables.
    - **Random Forest**: An ensemble learning method that constructs multiple decision trees and merges them to improve the accuracy of predictions.
    - **Gaussian**: Likely refers to a Gaussian Naive Bayes model, which assumes that the features follow a Gaussian distribution.
""")

# Add Footer Information
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    **Developed by**: Your Name | **Github**: [your-github-link](https://github.com)
    For educational purposes only. Please consult a healthcare professional for medical advice.
""")
