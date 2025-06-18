
import streamlit as st
import pickle
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    with open('lr.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.set_page_config(page_title="University Admission Predictor", layout="centered")
st.title("🎓 University Admission Predictor")

st.markdown("Enter your academic profile below to estimate your chances of admission:")

# Input fields
gre = st.number_input("GRE Score (out of 340)", min_value=260, max_value=340, value=300)
toefl = st.number_input("TOEFL Score (out of 120)", min_value=0, max_value=120, value=100)
rating = st.selectbox("University Rating (1-5)", [1, 2, 3, 4, 5])
sop = st.slider("SOP Strength", 1.0, 5.0, 3.0, 0.5)
lor = st.slider("LOR Strength", 1.0, 5.0, 3.0, 0.5)
cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, value=8.0)
research = st.radio("Research Experience", ["No", "Yes"])

# Convert inputs to numeric
research_val = 1 if research == "Yes" else 0

# Prepare input array
input_data = np.array([[gre, toefl, rating, sop, lor, cgpa, research_val]])

# Predict
if st.button("Predict Admission"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ You are likely to be admitted!")
    else:
        st.error("❌ You may not be admitted. Consider improving your profile.")

st.markdown("---")
st.caption("Developed using Streamlit · Logistic Regression Model")
