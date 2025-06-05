import streamlit as st
import numpy as np
import pickle

# Set page config
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model and scaler
with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Header
st.title("🏥 Diabetes Risk Predictor")
st.markdown("""
    This application helps predict the likelihood of diabetes based on various health parameters.
    Please fill in your details below to get a prediction.
""")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1,
                                help="Enter the number of times pregnant (0 if not applicable)")
    age = st.number_input('Age', min_value=10, max_value=100, value=30,
                         help="Enter your age in years")
    bmi = st.number_input('BMI (Body Mass Index)', min_value=0.0, max_value=70.0, value=25.0,
                         help="Enter your BMI (weight in kg / height in m²)")
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5,
                         help="A function which scores likelihood of diabetes based on family history")

with col2:
    st.subheader("Medical Measurements")
    glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=200, value=120,
                            help="Enter your glucose level in mg/dL")
    blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=150, value=70,
                                   help="Enter your diastolic blood pressure in mm Hg")
    skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20,
                                   help="Enter your triceps skin fold thickness in mm")
    insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=900, value=80,
                            help="Enter your 2-Hour serum insulin level")

# Prediction button
st.markdown("---")
if st.button('Predict Diabetes Risk'):
    with st.spinner('Analyzing your data...'):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = classifier.predict(input_scaled)
        
        if prediction[0] == 1:
            st.markdown("""
                <div class="prediction-box" style="background-color: #ffebee;">
                    <h2 style="color: #c62828;">⚠️ High Risk of Diabetes</h2>
                    <p>Based on the provided information, there is a high risk of diabetes. 
                    Please consult with a healthcare professional for proper evaluation and guidance.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="prediction-box" style="background-color: #e8f5e9;">
                    <h2 style="color: #2e7d32;">✅ Low Risk of Diabetes</h2>
                    <p>Based on the provided information, there is a low risk of diabetes. 
                    However, it's always good to maintain a healthy lifestyle and regular check-ups.</p>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Note: This is a prediction tool and should not be used as a substitute for professional medical advice.</p>
    </div>
""", unsafe_allow_html=True)