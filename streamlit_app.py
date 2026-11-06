import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
from pypdf import PdfReader
from PIL import Image
import pytesseract

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="DiaCare AI",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------
# LOAD TRAINED MODEL
# --------------------------------

@st.cache_resource
def load_model():
    return joblib.load("diacare_rf_model.pkl")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load the trained model: {e}")

# --------------------------------
# CUSTOM DESIGN
# --------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #f4faf7;
}
h1, h2, h3, p, label {
    color: #163b32;
}
div.stButton > button {
    background-color: #087f5b;
    color: white;
    border-radius: 10px;
    font-weight: bold;
    border: none;
}
div.stButton > button:hover {
    background-color: #065f46;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# HEADER
# --------------------------------

st.title("🩺 DiaCare AI")
st.subheader("Your Health. Your Awareness. Your Future.")

st.write(
    "AI-powered diabetes screening prototype "
    "designed to encourage early health awareness."
)

st.info(
    "DiaCare AI is an educational screening prototype. "
    "It does not diagnose diabetes or replace professional medical advice."
)

# --------------------------------
# NAVIGATION
# --------------------------------

home, screening, scan, chatbot, monitor = st.tabs([
    "🏠 Home",
    "🔍 Screening",
    "📄 Scan & Analyse",
    "💬 Talk to DiaCare",
    "📊 Connect & Monitor"
])

# --------------------------------
# HOME
# --------------------------------

with home:
    st.header("Welcome to DiaCare AI 💚")

    st.write("""
    DiaCare AI aims to make diabetes risk awareness
    easier and more accessible.

    ### What can you do?

    - 🔍 Complete a diabetes symptom screening.
    - 📄 Upload a medical report for text extraction.
    - 💬 Ask basic diabetes awareness questions.
    - 📊 Record glucose readings manually.
    """)

    st.success(
        "Small steps towards better health awareness!"
    )

# --------------------------------
# SCREENING
# --------------------------------

symptoms = [
    "polyuria",
    "polydipsia",
    "sudden_weight_loss",
    "weakness",
    "polyphagia",
    "genital_thrush",
    "visual_blurring",
    "itching",
    "irritability",
    "delayed_healing",
    "partial_paresis",
    "muscle_stiffness",
    "alopecia",
    "obesity"
]

symptom_labels = {
    "polyuria": "Frequent urination",
    "polydipsia": "Excessive thirst",
    "sudden_weight_loss": "Sudden weight loss",
    "weakness": "General weakness",
    "polyphagia": "Excessive hunger",
    "genital_thrush": "Genital thrush",
    "visual_blurring": "Blurred vision",
    "itching": "Itching",
    "irritability": "Irritability",
    "delayed_healing": "Delayed wound healing",
    "partial_paresis": "Partial muscle weakness",
    "muscle_stiffness": "Muscle stiffness",
    "alopecia": "Hair loss",
    "obesity": "Obesity"
}

with screening:
    st.header("Diabetes Risk Screening")
    st.write("Please answer the following questions.")

    age = st.number_input(
        "Your age",
        min_value=1,
        max_value=120,
        value=25
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    answers = {}

    st.subheader("Symptoms")

    for symptom in symptoms:
        answers[symptom] = st.selectbox(
            symptom_labels[symptom],
            ["No", "Yes"],
            key=f"screen_{symptom}"
        )

    if st.button("🔍 Analyse My Responses"):

        if not model_loaded:
            st.error("The trained model could not be loaded.")

        else:
            input_data = {
                "age": age,
                "gender": gender
            }

            input_data.update(answers)

            input_df = pd.DataFrame([input_data])

            prediction = model.predict(input_df)[0]

            if prediction == "Positive":
                st.warning(
                    "⚠️ Higher Screening Indication"
                )
                st.write(
                    "Your responses indicate a higher screening "
                    "signal. Please consult a qualified healthcare "
                    "professional for proper assessment."
                )

            else:
                st.success(
                    "✅ Lower Screening Indication"
                )
                st.write(
                    "Your responses indicate a lower screening "
                    "signal. This does not rule out diabetes. "
                    "Consult a healthcare professional if you "
                    "have concerns."
                )

# --------------------------------
# SCAN AND ANALYSE
# --------------------------------

with scan:
    st.header("📄 Scan & Analyse Medical Reports")

    st.write(
        "Upload a PDF or image to extract text relating "
        "to glucose and HbA1c."
    )

    uploaded_file = st.file_uploader(
        "Upload your report",
        type=["pdf", "png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        extracted_text = ""

        try:
            if uploaded_file.name.lower().endswith(".pdf"):
                reader = PdfReader(uploaded_file)

                for page in reader.pages:
                    extracted_text += page.extract_text() or ""

            else:
                image_file = Image.open(uploaded_file)
                extracted_text = pytesseract.image_to_string(
                    image_file
                )

            if extracted_text.strip():

                st.subheader("Extracted Report Text")
                st.text_area(
                    "Report contents",
                    extracted_text,
                    height=250
                )

                keywords = [
                    "glucose",
                    "blood sugar",
                    "hba1c",
                    "a1c"
                ]

                relevant_lines = [
                    line for line in extracted_text.splitlines()
                    if any(
                        word in line.lower()
                        for word in keywords
                    )
                ]

                st.subheader("Relevant References")

                if relevant_lines:
                    for line in relevant_lines:
                        st.info(line)
                else:
                    st.write(
                        "No glucose or HbA1c references found."
                    )

            else:
                st.warning(
                    "No readable text was found. "
                    "Try a clearer report or an image."
                )

        except Exception as e:
            st.error(f"Report extraction error: {e}")

        st.caption(
            "Extracted text is not a medical interpretation. "
            "Please consult a healthcare professional."
        )

# --------------------------------
# TALK TO DIACARE
# --------------------------------

def chatbot_response(message):

    msg = message.lower()

    if any(word in msg for word in [
        "emergency", "unconscious", "chest pain",
        "difficulty breathing"
    ]):
        return (
            "These symptoms may require urgent medical attention. "
            "Contact local emergency services immediately."
        )

    elif "symptom" in msg:
        return (
            "Common diabetes symptoms can include frequent "
            "urination, excessive thirst, increased hunger, "
            "unexplained weight loss and tiredness."
        )

    elif "hba1c" in msg or "a1c" in msg:
        return (
            "HbA1c is a blood test that reflects average blood "
            "glucose over approximately the previous 2–3 months. "
            "Ask a healthcare professional to interpret your result."
        )

    elif "glucose" in msg or "blood sugar" in msg:
        return (
            "Blood glucose measures the amount of sugar in your "
            "blood. Results need to be interpreted using the "
            "correct units, test type and clinical context."
        )

    elif "food" in msg or "diet" in msg:
        return (
            "A balanced eating pattern with vegetables, whole "
            "grains, appropriate portions and limited sugary "
            "drinks can support general health. A dietitian can "
            "provide individual advice."
        )

    elif "prevent" in msg:
        return (
            "Regular physical activity, balanced nutrition, "
            "healthy habits and appropriate health check-ups "
            "can support diabetes prevention."
        )

    elif "doctor" in msg:
        return (
            "A qualified healthcare professional can arrange "
            "appropriate tests and interpret your results."
        )

    else:
        return (
            "I can help with general questions about diabetes "
            "symptoms, HbA1c, glucose, healthy habits and "
            "medical check-ups. What would you like to know?"
        )


with chatbot:
    st.header("💬 Talk to DiaCare")

    st.write(
        "Ask questions about diabetes awareness and healthy habits."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    user_message = st.chat_input(
        "Ask DiaCare a question..."
    )

    if user_message:

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_message
        })

        response = chatbot_response(user_message)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })

        st.rerun()

# --------------------------------
# CONNECT AND MONITOR
# --------------------------------

with monitor:
    st.header("📊 Connect & Monitor")

    st.write(
        "Record your glucose readings manually."
    )

    if "readings" not in st.session_state:
        st.session_state.readings = []

    reading = st.number_input(
        "Glucose reading",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

    unit = st.selectbox(
        "Unit",
        ["mg/dL", "mmol/L"]
    )

    note = st.text_input(
        "Additional notes (optional)"
    )

    if st.button("Save Reading"):

        st.session_state.readings.append({
            "Date and Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M"
            ),
            "Reading": reading,
            "Unit": unit,
            "Notes": note
        })

        st.success("Reading saved for this session!")

    if st.session_state.readings:

        st.subheader("Your Recorded Readings")

        df = pd.DataFrame(st.session_state.readings)

        st.dataframe(
            df,
            use_container_width=True
        )

        st.caption(
            "Readings are stored only during the current session. "
            "They are not permanently saved."
        )

    else:
        st.info("No readings recorded yet.")

# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "DiaCare AI | AI-powered diabetes awareness prototype | "
    "Not a substitute for professional medical diagnosis."
)
