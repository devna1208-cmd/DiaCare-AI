import gradio as gr
import pandas as pd
import joblib
import re
from datetime import datetime
from pypdf import PdfReader
from PIL import Image
import pytesseract


# ==========================================
# 1. LOAD TRAINED AI MODEL
# ==========================================

model = joblib.load("diacare_rf_model.pkl")


# ==========================================
# 2. DIABETES SCREENING
# ==========================================

def predict_diabetes(
    age, gender, polyuria, polydipsia,
    sudden_weight_loss, weakness, polyphagia,
    genital_thrush, visual_blurring, itching,
    irritability, delayed_healing, partial_paresis,
    muscle_stiffness, alopecia, obesity
):

    if age is None or gender is None:
        return "Please enter your age and gender."

    answers = [
        polyuria, polydipsia, sudden_weight_loss,
        weakness, polyphagia, genital_thrush,
        visual_blurring, itching, irritability,
        delayed_healing, partial_paresis,
        muscle_stiffness, alopecia, obesity
    ]

    if any(answer is None for answer in answers):
        return "Please answer all the screening questions."

    patient_data = pd.DataFrame([{
        "age": int(age),
        "gender": gender,
        "polyuria": polyuria,
        "polydipsia": polydipsia,
        "sudden_weight_loss": sudden_weight_loss,
        "weakness": weakness,
        "polyphagia": polyphagia,
        "genital_thrush": genital_thrush,
        "visual_blurring": visual_blurring,
        "itching": itching,
        "irritability": irritability,
        "delayed_healing": delayed_healing,
        "partial_paresis": partial_paresis,
        "muscle_stiffness": muscle_stiffness,
        "alopecia": alopecia,
        "obesity": obesity
    }])

    prediction = model.predict(patient_data)[0]

    if prediction == "Positive":
        return """
## Higher Screening Indication

Your responses indicate a higher screening indication.

Please consult a qualified healthcare professional
for appropriate medical assessment.
        
**Remember:** This is not a medical diagnosis.
"""

    return """
## Lower Screening Indication

Your responses indicate a lower screening indication.

This does not rule out diabetes. Consult a healthcare
professional if you have concerns or symptoms.

**Remember:** This is not a medical diagnosis.
"""


# ==========================================
# 3. SCAN AND ANALYSE REPORTS
# ==========================================

def scan_report(file):

    if file is None:
        return "Please upload a report first."

    try:
        filename = file.lower()

        if filename.endswith(".pdf"):

            reader = PdfReader(file)
            extracted_text = ""

            for page in reader.pages:
                extracted_text += page.extract_text() or ""

            if not extracted_text.strip():
                return (
                    "No readable text was found in this PDF. "
                    "Please upload a clear JPG or PNG image instead."
                )

        elif filename.endswith((".png", ".jpg", ".jpeg")):

            image = Image.open(file)
            extracted_text = pytesseract.image_to_string(image)

        else:
            return "Please upload a PDF, JPG or PNG file."

        lines = extracted_text.splitlines()

        keywords = ["glucose", "blood sugar", "hba1c", "a1c"]

        results = [
            line.strip()
            for line in lines
            if any(word in line.lower() for word in keywords)
        ]

        findings = (
            "\n".join(["- " + item for item in results])
            if results
            else "No glucose or HbA1c references were identified."
        )

        return f"""
## Report Extraction Complete!

### Possible glucose and HbA1c references

{findings}

### Extracted Report Text

{extracted_text[:4000]}

---

**Important:** This is automated text extraction,
not medical interpretation. Please verify extracted
values and units against the original report.
"""

    except Exception as error:
        return f"Unable to read the report: {str(error)}"


# ==========================================
# 4. TALK TO DIACARE CHATBOT
# ==========================================

def talk_to_diacare(message, history):

    if not message:
        return history or [], ""

    question = message.lower()

    if any(word in question for word in [
        "chest pain", "difficulty breathing",
        "unconscious", "fainting"
    ]):

        answer = """
⚠️ **Please seek emergency medical help immediately.**

These symptoms may indicate a serious medical problem.
Do not rely on this chatbot for emergency assistance.
"""

    elif "symptom" in question:

        answer = """
### Common Diabetes Symptoms

- Frequent urination
- Excessive thirst
- Increased hunger
- Unexplained weight loss
- Blurred vision
- Unusual tiredness

These symptoms can have other causes too.
Consult a healthcare professional for assessment.
"""

    elif "hba1c" in question or "a1c" in question:

        answer = """
### What is HbA1c?

HbA1c is a blood test that measures average
blood glucose over approximately 2–3 months.

A healthcare professional can explain your result
using the appropriate clinical reference ranges.
"""

    elif "glucose" in question or "blood sugar" in question:

        answer = """
### What is Blood Glucose?

Blood glucose is the amount of sugar present
in your blood.

It is an important measurement when screening
for and monitoring diabetes.
"""

    elif "prevent" in question:

        answer = """
### Healthy Habits

- Eat a balanced diet.
- Stay physically active.
- Maintain a healthy weight.
- Get enough sleep.
- Attend regular health check-ups.
"""

    elif "food" in question or "eat" in question:

        answer = """
### Healthy Eating Tips

- Include vegetables and fruits.
- Choose whole grains.
- Include healthy protein sources.
- Limit sugary drinks.
- Pay attention to portion sizes.
"""

    elif "doctor" in question or "consult" in question:

        answer = """
### When Should You Consult a Doctor?

Consider consulting a healthcare professional
if you experience excessive thirst, frequent
urination, unexplained weight loss or persistent tiredness.
"""

    else:

        answer = """
Hello! 👋 I'm DiaCare, your diabetes health-awareness assistant.

I can help you learn about:

- Diabetes symptoms
- HbA1c
- Blood glucose
- Prevention
- Healthy eating
- When to consult a doctor

Try asking me one of these questions!

I provide general health information,
not medical diagnosis or treatment.
"""

    history = history or []

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return history, ""


# ==========================================
# 5. CONNECT AND MONITOR
# ==========================================

def add_glucose_reading(value, unit, note, history):

    history = history or []

    if value is None or value <= 0:
        return "Please enter a valid glucose reading.", history, pd.DataFrame(history)

    reading = {
        "Date": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "Glucose Reading": value,
        "Unit": unit,
        "Note": note if note else "No note"
    }

    history.append(reading)

    return (
        "Your glucose reading has been added successfully!",
        history,
        pd.DataFrame(history)
    )


# ==========================================
# 6. APPLICATION DESIGN
# ==========================================

custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    background: #f4faf7 !important;
    color: #000000 !important;
}

.gradio-container p,
.gradio-container label,
.gradio-container span,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
    color: #000000 !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    color: #000000 !important;
    background: #ffffff !important;
}

.gradio-container .tab-nav button,
.gradio-container button[role="tab"] {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    background: #e5eee9 !important;
    opacity: 1 !important;
    font-weight: bold !important;
}

.gradio-container .tab-nav button.selected,
.gradio-container button[role="tab"][aria-selected="true"] {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    background: #b7e4d2 !important;
}

.gradio-container button {
    font-weight: bold !important;
}

footer {
    display: none !important;
}
"""


# ==========================================
# 7. CREATE DIACARE AI DASHBOARD
# ==========================================

with gr.Blocks(title="DiaCare AI") as demo:

    gr.Markdown("""
# 💚 DiaCare AI

### AI-Powered Diabetes Risk Screening and Awareness

Helping people understand diabetes risk indicators
and encouraging early health awareness.

**Disclaimer:** DiaCare AI is an educational screening
prototype. It does not diagnose diabetes or replace
professional medical advice.
""")

    with gr.Tabs():

        # HOME
        with gr.Tab("Home"):

            gr.Markdown("""
# Welcome to DiaCare AI! 💚

Your diabetes health-awareness companion.

### What can DiaCare AI do?

🩺 **Screening:** Complete an interactive health questionnaire.

📄 **Scan & Analyse:** Extract possible glucose and HbA1c
references from uploaded reports.

💬 **Talk to DiaCare:** Ask general diabetes awareness questions.

📊 **Connect & Monitor:** Record glucose readings.

### Our Mission

To encourage early diabetes awareness and make
health information easier to access.
""")

        # SCREENING
        with gr.Tab("Screening"):

            gr.Markdown("""
## Diabetes Risk Screening

Please answer all questions below.
""")

            age = gr.Number(
                label="Age",
                minimum=1,
                maximum=120,
                precision=0
            )

            gender = gr.Dropdown(
                ["Female", "Male"],
                label="Gender"
            )

            polyuria = gr.Dropdown(["Yes", "No"], label="Frequent urination")
            polydipsia = gr.Dropdown(["Yes", "No"], label="Excessive thirst")
            sudden_weight_loss = gr.Dropdown(["Yes", "No"], label="Sudden weight loss")
            weakness = gr.Dropdown(["Yes", "No"], label="Weakness")
            polyphagia = gr.Dropdown(["Yes", "No"], label="Excessive hunger")
            genital_thrush = gr.Dropdown(["Yes", "No"], label="Genital thrush")
            visual_blurring = gr.Dropdown(["Yes", "No"], label="Blurred vision")
            itching = gr.Dropdown(["Yes", "No"], label="Itching")
            irritability = gr.Dropdown(["Yes", "No"], label="Irritability")
            delayed_healing = gr.Dropdown(["Yes", "No"], label="Delayed healing")
            partial_paresis = gr.Dropdown(["Yes", "No"], label="Partial paresis")
            muscle_stiffness = gr.Dropdown(["Yes", "No"], label="Muscle stiffness")
            alopecia = gr.Dropdown(["Yes", "No"], label="Hair loss")
            obesity = gr.Dropdown(["Yes", "No"], label="Obesity")

            screen_button = gr.Button("Analyse My Responses")

            screening_result = gr.Markdown()

            screen_button.click(
                predict_diabetes,
                inputs=[
                    age, gender, polyuria, polydipsia,
                    sudden_weight_loss, weakness, polyphagia,
                    genital_thrush, visual_blurring, itching,
                    irritability, delayed_healing, partial_paresis,
                    muscle_stiffness, alopecia, obesity
                ],
                outputs=screening_result
            )

        # SCAN AND ANALYSE
        with gr.Tab("Scan & Analyse"):

            gr.Markdown("""
## Scan & Analyse Your Report

Upload a PDF or image of a report to extract
possible glucose and HbA1c references.
""")

            report_file = gr.File(
                label="Upload Medical Report",
                file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                type="filepath"
            )

            scan_button = gr.Button("Scan Report")

            scan_result = gr.Markdown()

            scan_button.click(
                scan_report,
                inputs=report_file,
                outputs=scan_result
            )

        # TALK TO DIACARE
        with gr.Tab("Talk to DiaCare"):

            gr.Markdown("""
## Talk to DiaCare 💬

Ask questions about diabetes awareness,
symptoms, HbA1c, glucose and prevention.
""")

            chatbot = gr.Chatbot(
                label="DiaCare Assistant",
                height=400
            )

            message = gr.Textbox(
                label="Your Message",
                placeholder="Ask DiaCare a question..."
            )

            send_button = gr.Button("Send")

            send_button.click(
                talk_to_diacare,
                inputs=[message, chatbot],
                outputs=[chatbot, message]
            )

            message.submit(
                talk_to_diacare,
                inputs=[message, chatbot],
                outputs=[chatbot, message]
            )

        # CONNECT AND MONITOR
        with gr.Tab("Connect & Monitor"):

            gr.Markdown("""
## Connect & Monitor 📊

Record your glucose readings and view your
entries during this application session.
""")

            glucose_value = gr.Number(
                label="Glucose Reading",
                minimum=0
            )

            glucose_unit = gr.Dropdown(
                ["mg/dL", "mmol/L"],
                label="Unit",
                value="mg/dL"
            )

            glucose_note = gr.Textbox(
                label="Note",
                placeholder="Example: Before breakfast"
            )

            glucose_state = gr.State([])

            monitor_button = gr.Button("Add Reading")

            monitor_message = gr.Markdown()

            glucose_table = gr.Dataframe(
                headers=[
                    "Date",
                    "Glucose Reading",
                    "Unit",
                    "Note"
                ],
                label="Reading History"
            )

            monitor_button.click(
                add_glucose_reading,
                inputs=[
                    glucose_value,
                    glucose_unit,
                    glucose_note,
                    glucose_state
                ],
                outputs=[
                    monitor_message,
                    glucose_state,
                    glucose_table
                ]
            )


# ==========================================
# 8. LAUNCH APPLICATION
# ==========================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="teal",
            neutral_hue="slate"
        ),
        css=custom_css
    )
