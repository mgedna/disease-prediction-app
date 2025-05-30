import streamlit as st
import numpy as np
import joblib
import base64
import re
from fpdf import FPDF
import matplotlib.pyplot as plt
from PIL import Image
from pdf2image import convert_from_bytes
import easyocr

st.set_page_config(page_title="Disease Prediction and Early Diagnosis", layout="wide")

def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)

def create_pdf(input_data, prediction, proba, feature_names, report_title, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=report_title, ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt="Input Values:", ln=True)
    for name, val in zip(feature_names, input_data[0]):
        pdf.cell(200, 10, txt=f"{name}: {val}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}", ln=True)
    pdf.cell(200, 10, txt=f"Probability: {proba:.2f}", ln=True)
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Download Report as PDF</a>'
    return href

def plot_probability_bar(probability):
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.barh([0], [probability], color='crimson' if probability > 0.5 else 'seagreen')
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_title(f"Risk Probability: {probability:.2f}")
    st.pyplot(fig)

st.markdown("<h1 style='text-align:center;'>🧠 Disease Prediction and Early Diagnosis</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Optional: Upload Lab Report (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])
glucose_value, cholesterol_value = 120, 180

if uploaded_file:
    reader = easyocr.Reader(['ro', 'en'])
    text = ""
    try:
        if "pdf" in uploaded_file.type:
            images = convert_from_bytes(uploaded_file.read())
            for img in images:
                result = reader.readtext(np.array(img), detail=0)
                text += "\n".join(result)
        else:
            image = Image.open(uploaded_file).convert("RGB")
            result = reader.readtext(np.array(image), detail=0)
            text = "\n".join(result)
        st.markdown("### 📝 Extracted Report Text")
        st.text_area("OCR Result", text, height=200)

        glucose_match = re.search(r"(?i)gluco[zăz]?a?.*?:?\s*(\d{2,3})", text)
        cholesterol_match = re.search(r"(?i)colesterol.*?:?\s*(\d{2,3})", text)

        if glucose_match:
            glucose_value = int(glucose_match.group(1))
            st.success(f"Extracted Glucose: {glucose_value}")
        if cholesterol_match:
            cholesterol_value = int(cholesterol_match.group(1))
            st.success(f"Extracted Cholesterol: {cholesterol_value}")
    except Exception as e:
        st.error(f"Failed to process file: {e}")

disease = st.selectbox("Select Disease", ["Cardiovascular Disease", "Diabetes"])

if disease == "Cardiovascular Disease":
    st.markdown("### 🫀 Cardiovascular Risk Inputs")
    with st.form("cv_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
            height = st.slider("Height (cm)", 130, 210, 170)
            weight = st.slider("Weight (kg)", 30, 200, 70)
            ap_hi = st.slider("Systolic BP", 90, 200, 120)
            ap_lo = st.slider("Diastolic BP", 60, 150, 80)
            age = st.slider("Age (years)", 20, 100, 45)
        with col2:
            cholesterol = st.slider("Cholesterol", 100, 300, cholesterol_value)
            gluc = st.slider("Glucose", 50, 200, glucose_value)
            smoke = st.selectbox("Smokes?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            alco = st.selectbox("Drinks Alcohol?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            active = st.selectbox("Physically Active?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        submit_cv = st.form_submit_button("🧪 Predict Cardiovascular Risk")

    if submit_cv:
        bmi = calculate_bmi(weight, height)
        st.info(f"📏 Calculated BMI: {bmi}")
        chol_2 = 1 if cholesterol > 200 and cholesterol <= 240 else 0
        chol_3 = 1 if cholesterol > 240 else 0
        gluc_2 = 1 if glucose_value > 110 and glucose_value <= 140 else 0
        gluc_3 = 1 if glucose_value > 140 else 0
        input_data = np.array([[gender, height, weight, ap_hi, ap_lo, chol_2, chol_3, gluc_2, gluc_3,
                                smoke, alco, active, age, bmi]])
        model = joblib.load("models/model_cardio.pkl")
        scaler = joblib.load("models/scaler_cardio.pkl")
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]
        st.success(f"Prediction: {'High' if prediction else 'Low'} Risk (Prob: {proba:.2f})")
        plot_probability_bar(proba)
        features = ['Gender', 'Height', 'Weight', 'Systolic BP', 'Diastolic BP', 'Chol_2', 'Chol_3',
                    'Gluc_2', 'Gluc_3', 'Smoke', 'Alcohol', 'Active', 'Age', 'BMI']
        st.markdown(create_pdf(input_data, prediction, proba, features,
                               "Cardiovascular Disease Report", "cardio_report.pdf"), unsafe_allow_html=True)

elif disease == "Diabetes":
    st.markdown("### 🩸 Diabetes Risk Inputs")
    with st.form("db_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.slider("Pregnancies", 0, 20, 1)
            glucose = st.slider("Glucose Level", 50, 200, glucose_value)
            bp = st.slider("Blood Pressure", 40, 122, 70)
            skin = st.slider("Skin Thickness", 0, 100, 20)
        with col2:
            insulin = st.slider("Insulin", 0, 900, 85)
            weight = st.slider("Weight (kg)", 30, 200, 80)
            height = st.slider("Height (cm)", 130, 210, 175)
            bmi = calculate_bmi(weight, height)
            st.info(f"📏 Calculated BMI: {bmi}")
            dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
            age = st.slider("Age (years)", 10, 100, 40)
        submit_db = st.form_submit_button("🧪 Predict Diabetes Risk")

    if submit_db:
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        model = joblib.load("models/model_diabetes.pkl")
        scaler = joblib.load("models/scaler_diabetes.pkl")
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]
        st.success(f"Prediction: {'High' if prediction else 'Low'} Risk (Prob: {proba:.2f})")
        plot_probability_bar(proba)
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age']
        st.markdown(create_pdf(input_data, prediction, proba, features,
                               "Diabetes Risk Report", "diabetes_report.pdf"), unsafe_allow_html=True)

st.markdown("<hr><center>© 2025 Ego AI Medical App</center>", unsafe_allow_html=True)
