import os
import joblib
import gradio as gr
import pandas as pd
import pytesseract

from PIL import Image
from pypdf import PdfReader


# =========================================================
# 1. LOAD MACHINE LEARNING MODEL
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "diacare_rf_model.pkl")

model = joblib.load(MODEL_PATH)

print("DiaCare AI model loaded successfully!")


# =========================================================
# 2. TRANSLATIONS
# =========================================================

TRANSLATIONS = {
    "English": {
        "title": "🩺 DiaCare AI",
        "subtitle": "Your Health. Your Awareness. Your Future.",
        "description": "AI-powered diabetes screening prototype",
        "language": "🌍 Select Your Language",
        "welcome": "Welcome to DiaCare AI!",
        "home_text": "Your personal health awareness assistant. Explore our screening, report analysis, chatbot and glucose monitoring features.",
        "screening_title": "Diabetes Risk Screening",
        "screening_info": "Answer the questions below to receive an AI-based screening indication. This is not a medical diagnosis.",
        "age": "Age",
        "gender": "Gender",
        "female": "Female",
        "male": "Male",
        "yes": "Yes",
        "no": "No",
        "analyse": "🔍 Analyse My Responses",
        "clear": "Clear Answers",
        "result": "Screening Result",
        "scan_title": "Medical Report Analysis",
        "scan_info": "Upload a PDF or image of your medical report for text extraction and basic analysis.",
        "upload": "Upload Your Report",
        "analyse_report": "Analyse Report",
        "chat_title": "Talk to DiaCare",
        "chat_info": "Ask general questions about diabetes, symptoms and healthy habits.",
        "message": "Type your message here",
        "send": "Send",
        "monitor_title": "Glucose Monitoring",
        "glucose_info": "Record your blood glucose readings and monitor your entries.",
        "glucose": "Blood Glucose (mg/dL)",
        "add": "Add Reading",
        "higher": "⚠️ Higher Screening Indication",
        "lower": "✅ Lower Screening Indication",
        "higher_text": "Your responses indicate a higher screening risk. Please consult a qualified healthcare professional for proper testing and advice.",
        "lower_text": "Your responses indicate a lower screening risk. This does not rule out diabetes. Consult a healthcare professional if you have concerns.",
        "invalid_age": "Please enter a valid age.",
        "report_error": "Unable to read the report. Please upload a valid PDF or image.",
        "empty_report": "No readable text was found in the uploaded report.",
        "report_success": "Report text extracted successfully:",
        "glucose_error": "Please enter a valid glucose reading.",
        "glucose_success": "Glucose reading added successfully!",
        "disclaimer": "⚠️ Disclaimer: DiaCare AI is an educational screening prototype, not a medical diagnostic tool. It cannot replace professional medical advice, diagnosis or treatment."
    },

    "French": {
        "title": "🩺 DiaCare AI",
        "subtitle": "Votre santé. Votre sensibilisation. Votre avenir.",
        "description": "Prototype de dépistage du diabète basé sur l'IA",
        "language": "🌍 Choisissez votre langue",
        "welcome": "Bienvenue sur DiaCare AI !",
        "home_text": "Votre assistant personnel de sensibilisation à la santé. Découvrez nos fonctionnalités de dépistage, d'analyse de rapports, de chatbot et de suivi de la glycémie.",
        "screening_title": "Dépistage du diabète",
        "screening_info": "Répondez aux questions ci-dessous pour obtenir une indication de dépistage basée sur l'IA. Ceci n'est pas un diagnostic médical.",
        "age": "Âge",
        "gender": "Sexe",
        "female": "Femme",
        "male": "Homme",
        "yes": "Oui",
        "no": "Non",
        "analyse": "🔍 Analyser mes réponses",
        "clear": "Effacer les réponses",
        "result": "Résultat du dépistage",
        "scan_title": "Analyse d'un rapport médical",
        "scan_info": "Téléchargez un rapport médical PDF ou une image pour extraire son texte.",
        "upload": "Télécharger votre rapport",
        "analyse_report": "Analyser le rapport",
        "chat_title": "Parler à DiaCare",
        "chat_info": "Posez des questions générales sur le diabète, les symptômes et les habitudes saines.",
        "message": "Écrivez votre message ici",
        "send": "Envoyer",
        "monitor_title": "Suivi de la glycémie",
        "glucose_info": "Enregistrez vos mesures de glycémie et consultez vos relevés.",
        "glucose": "Glycémie (mg/dL)",
        "add": "Ajouter une mesure",
        "higher": "⚠️ Indication de dépistage plus élevée",
        "lower": "✅ Indication de dépistage plus faible",
        "higher_text": "Vos réponses indiquent un risque de dépistage plus élevé. Veuillez consulter un professionnel de santé.",
        "lower_text": "Vos réponses indiquent un risque de dépistage plus faible. Cela n'exclut pas le diabète. Consultez un professionnel de santé si vous avez des inquiétudes.",
        "invalid_age": "Veuillez saisir un âge valide.",
        "report_error": "Impossible de lire le rapport. Veuillez télécharger un PDF ou une image valide.",
        "empty_report": "Aucun texte lisible n'a été trouvé dans le rapport.",
        "report_success": "Texte du rapport extrait avec succès :",
        "glucose_error": "Veuillez saisir une mesure de glycémie valide.",
        "glucose_success": "Mesure de glycémie ajoutée avec succès !",
        "disclaimer": "⚠️ Avertissement : DiaCare AI est un prototype éducatif de dépistage et non un outil de diagnostic médical. Il ne remplace pas les conseils, le diagnostic ou le traitement d'un professionnel de santé."
    },

    "Creole": {
        "title": "🩺 DiaCare AI",
        "subtitle": "To lasante. To konsians. To lavenir.",
        "description": "En prototip depistaz diabet avek lintelizans artifisiel",
        "language": "🌍 Swazir to lang",
        "welcome": "Bienveni lor DiaCare AI!",
        "home_text": "To asistan personel pou to lasante. Dekouver nou bann fonksion depistaz, analiz rapor, chatbot ek swiv glikoz.",
        "screening_title": "Depistaz Diabet",
        "screening_info": "Reponn bann kestion anba pou gagn enn indikasion depistaz avek IA. Sa pa enn diagnoz medikal.",
        "age": "Laz",
        "gender": "Sex",
        "female": "Fam",
        "male": "Misie",
        "yes": "Wi",
        "no": "Non",
        "analyse": "🔍 Analiz Mo Bann Repons",
        "clear": "Efas Repons",
        "result": "Rezilta Depistaz",
        "scan_title": "Analiz Rapor Medikal",
        "scan_info": "Met enn rapor medikal PDF ouswa enn zimaz pou analiz so text.",
        "upload": "Met To Rapor",
        "analyse_report": "Analiz Rapor",
        "chat_title": "Koz avek DiaCare",
        "chat_info": "Poz bann kestion zeneral lor diabet, bann sinptom ek bann bon labitid.",
        "message": "Ekri to mesaz isi",
        "send": "Avoye",
        "monitor_title": "Swivi Glikoz",
        "glucose_info": "Anrezistre to bann rezilta glikoz pou swiv to bann mezir.",
        "glucose": "Glikoz dan Disan (mg/dL)",
        "add": "Azout Rezilta",
        "higher": "⚠️ Indikasion Depistaz pli Eleve",
        "lower": "✅ Indikasion Depistaz pli Ba",
        "higher_text": "To bann repons indik enn risk depistaz pli eleve. Silvouple al get enn profesionnnel lasante pou fer bann test ek gagn bann konsey.",
        "lower_text": "To bann repons indik enn risk depistaz pli ba. Sa pa vedir to pena diabet. Al get enn profesionnnel lasante si to ena dout.",
        "invalid_age": "Silvouple rant enn laz valid.",
        "report_error": "Nou pa kapav lir sa rapor-la. Silvouple met enn PDF ouswa zimaz valid.",
        "empty_report": "Nou pa finn trouv okenn text ki kapav lir dan rapor-la.",
        "report_success": "Text rapor finn extraire avek sikse:",
        "glucose_error": "Silvouple rant enn mezir glikoz valid.",
        "glucose_success": "Rezilta glikoz finn anrezistre avek sikse!",
        "disclaimer": "⚠️ Not: DiaCare AI se enn prototip edikatif pou depistaz, pa enn zouti diagnoz medikal. Li pa ranplas konsey, diagnoz ouswa tretman enn profesionnnel lasante."
    }
}


# =========================================================
# 3. DIABETES SCREENING
# =========================================================

def predict_diabetes(
    age, gender, polyuria, polydipsia,
    sudden_weight_loss, weakness, polyphagia,
    genital_thrush, visual_blurring, itching,
    irritability, delayed_healing, partial_paresis,
    muscle_stiffness, alopecia, obesity,
    language="English"
):

    t = TRANSLATIONS.get(language, TRANSLATIONS["English"])

    try:
        if age is None:
            return t["invalid_age"]

        age = int(age)

        if age < 1 or age > 120:
            return t["invalid_age"]

        answers = [
            polyuria, polydipsia, sudden_weight_loss,
            weakness, polyphagia, genital_thrush,
            visual_blurring, itching, irritability,
            delayed_healing, partial_paresis,
            muscle_stiffness, alopecia, obesity
        ]

        if gender is None or any(answer is None for answer in answers):
            return "Please answer all screening questions before submitting."

        input_data = pd.DataFrame([[
            age,
            gender,
            *answers
        ]], columns=[
            "Age",
            "Gender",
            "Polyuria",
            "Polydipsia",
            "sudden weight loss",
            "weakness",
            "Polyphagia",
            "Genital thrush",
            "visual blurring",
            "Itching",
            "Irritability",
            "delayed healing",
            "partial paresis",
            "muscle stiffness",
            "Alopecia",
            "Obesity"
        ])

        prediction = model.predict(input_data)[0]

        if str(prediction).strip().lower() == "positive":
            return f"""
### {t["higher"]}

{t["higher_text"]}
"""

        return f"""
### {t["lower"]}

{t["lower_text"]}
"""

    except Exception as e:
        return f"An error occurred: {str(e)}"


# =========================================================
# 4. MEDICAL REPORT SCANNING
# =========================================================

def scan_report(file, language="English"):

    t = TRANSLATIONS.get(language, TRANSLATIONS["English"])

    if file is None:
        return t["report_error"]

    try:
        file_path = file.name if hasattr(file, "name") else str(file)
        extracted_text = ""

        if file_path.lower().endswith(".pdf"):

            reader = PdfReader(file_path)

            for page in reader.pages:
                extracted_text += page.extract_text() or ""

        else:

            image_file = Image.open(file_path)
            extracted_text = pytesseract.image_to_string(image_file)

        if not extracted_text.strip():
            return t["empty_report"]

        return f"""
### {t["report_success"]}

{extracted_text}
"""

    except Exception as e:
        print("Report scanning error:", e)
        return t["report_error"]


# =========================================================
# 5. DIAcare CHATBOT
# =========================================================

def talk_to_diacare(message, history, language="English"):

    if not message or not message.strip():
        return "", history

    message_lower = message.lower()

    if any(word in message_lower for word in [
        "symptom", "symptoms", "sign", "symptome", "sinptom"
    ]):
        response = (
            "Common diabetes symptoms may include frequent urination, "
            "increased thirst, unexplained weight loss, fatigue and blurred vision. "
            "Please consult a healthcare professional for assessment."
        )

    elif any(word in message_lower for word in [
        "food", "diet", "eat", "nutrition", "manze", "nourriture"
    ]):
        response = (
            "A balanced diet can support overall health. Include vegetables, "
            "whole grains, lean proteins and suitable portions. "
            "A healthcare professional or dietitian can provide individual advice."
        )

    elif any(word in message_lower for word in [
        "exercise", "sport", "activity", "exercice"
    ]):
        response = (
            "Regular physical activity can support blood sugar management "
            "and general health. Choose activities suitable for your abilities "
            "and seek medical advice before making major changes."
        )

    elif any(word in message_lower for word in [
        "diabetes", "diabet", "diabète"
    ]):
        response = (
            "Diabetes is a condition that affects how the body regulates blood "
            "glucose. There are different types, and a healthcare professional "
            "can explain testing and treatment options."
        )

    elif any(word in message_lower for word in [
        "hello", "hi", "hey", "bonjour", "salut", "bonzur"
    ]):
        response = (
            "Hello! Welcome to DiaCare AI. I can share general information "
            "about diabetes, symptoms and healthy habits. How can I help?"
        )

    else:
        response = (
            "I can help with general information about diabetes, symptoms, "
            "healthy eating and physical activity. For personal medical advice, "
            "please consult a qualified healthcare professional."
        )

    history = history or []

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response}
    ]

    return "", history


# =========================================================
# 6. GLUCOSE MONITORING
# =========================================================

def add_glucose_reading(value, readings, language="English"):

    t = TRANSLATIONS.get(language, TRANSLATIONS["English"])

    try:
        value = float(value)

        if value <= 0 or value > 1000:
            return readings or [], t["glucose_error"]

        readings = readings or []

        readings.append({
            "Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "Glucose Reading (mg/dL)": value
        })

        return readings, t["glucose_success"]

    except Exception:
        return readings or [], t["glucose_error"]


# =========================================================
# 7. LANGUAGE UPDATE FUNCTION
# =========================================================

def update_language(language):

    t = TRANSLATIONS.get(language, TRANSLATIONS["English"])

    return (
        gr.update(label=t["language"]),
        t["title"],
        t["subtitle"],
        t["description"],
        t["welcome"],
        t["home_text"],
        t["screening_title"],
        t["screening_info"],
        gr.update(label=t["age"]),
        gr.update(
            label=t["gender"],
            choices=[
                (t["female"], "Female"),
                (t["male"], "Male")
            ]
        ),
        t["analyse"],
        t["clear"],
        gr.update(),
        t["scan_title"],
        t["scan_info"],
        gr.update(label=t["upload"]),
        t["analyse_report"],
        t["chat_title"],
        t["chat_info"],
        gr.update(placeholder=t["message"]),
        t["send"],
        t["monitor_title"],
        t["glucose_info"],
        gr.update(label=t["glucose"]),
        t["add"],
        t["disclaimer"]
    )


# =========================================================
# 8. CUSTOM DESIGN
# =========================================================

CUSTOM_CSS = """
body {
    background: #f0fdf4 !important;
}

.gradio-container {
    background: #f0fdf4 !important;
    color: #111827 !important;
}

h1, h2, h3, p, label, span {
    color: #111827 !important;
}

button {
    color: #ffffff !important;
    font-weight: bold !important;
}

.tab-nav button {
    color: #111827 !important;
    font-weight: bold !important;
}

textarea, input {
    color: #111827 !important;
}

footer {
    display: none !important;
}
"""


# =========================================================
# 9. GRADIO APPLICATION
# =========================================================

with gr.Blocks(
    title="DiaCare AI",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="teal",
        neutral_hue="slate"
    )
) as demo:

    # LANGUAGE SELECTOR

    language = gr.Dropdown(
        choices=[
            ("🇬🇧 English", "English"),
            ("🇫🇷 Français", "French"),
            ("🇲🇺 Kreol Morisien", "Creole")
        ],
        value="English",
        label="🌍 Select Your Language",
        interactive=True
    )

    # HEADER

    header_title = gr.Markdown("# 🩺 DiaCare AI")

    header_subtitle = gr.Markdown(
        "### Your Health. Your Awareness. Your Future."
    )

    header_description = gr.Markdown(
        "AI-powered diabetes screening prototype"
    )

    with gr.Tabs():

        # HOME

        with gr.Tab("🏠 Home"):

            home_welcome = gr.Markdown(
                "## Welcome to DiaCare AI!"
            )

            home_text = gr.Markdown(
                "Your personal health awareness assistant. "
                "Explore our screening, report analysis, chatbot "
                "and glucose monitoring features."
            )

        # SCREENING

        with gr.Tab("🩺 Screening"):

            screening_title = gr.Markdown(
                "## Diabetes Risk Screening"
            )

            screening_info = gr.Markdown(
                "Answer the questions below to receive an AI-based "
                "screening indication. This is not a medical diagnosis."
            )

            age = gr.Number(
                label="Age",
                minimum=1,
                maximum=120,
                precision=0
            )

            gender = gr.Dropdown(
                choices=[
                    ("Female", "Female"),
                    ("Male", "Male")
                ],
                label="Gender",
                value="Female"
            )

            polyuria = gr.Radio(
                ["Yes", "No"],
                label="Do you experience frequent urination?"
            )

            polydipsia = gr.Radio(
                ["Yes", "No"],
                label="Do you experience excessive thirst?"
            )

            sudden_weight_loss = gr.Radio(
                ["Yes", "No"],
                label="Have you experienced sudden weight loss?"
            )

            weakness = gr.Radio(
                ["Yes", "No"],
                label="Do you experience weakness or fatigue?"
            )

            polyphagia = gr.Radio(
                ["Yes", "No"],
                label="Do you experience increased hunger?"
            )

            genital_thrush = gr.Radio(
                ["Yes", "No"],
                label="Have you experienced genital thrush?"
            )

            visual_blurring = gr.Radio(
                ["Yes", "No"],
                label="Do you experience blurred vision?"
            )

            itching = gr.Radio(
                ["Yes", "No"],
                label="Do you experience itching?"
            )

            irritability = gr.Radio(
                ["Yes", "No"],
                label="Do you experience irritability?"
            )

            delayed_healing = gr.Radio(
                ["Yes", "No"],
                label="Do your wounds heal slowly?"
            )

            partial_paresis = gr.Radio(
                ["Yes", "No"],
                label="Do you experience partial muscle weakness?"
            )

            muscle_stiffness = gr.Radio(
                ["Yes", "No"],
                label="Do you experience muscle stiffness?"
            )

            alopecia = gr.Radio(
                ["Yes", "No"],
                label="Have you experienced hair loss?"
            )

            obesity = gr.Radio(
                ["Yes", "No"],
                label="Are you affected by obesity?"
            )

            with gr.Row():

                clear_button = gr.Button(
                    "Clear Answers",
                    variant="secondary"
                )

                analyse_button = gr.Button(
                    "🔍 Analyse My Responses",
                    variant="primary"
                )

            screening_output = gr.Markdown()

            analyse_button.click(
                fn=predict_diabetes,
                inputs=[
                    age, gender, polyuria, polydipsia,
                    sudden_weight_loss, weakness, polyphagia,
                    genital_thrush, visual_blurring, itching,
                    irritability, delayed_healing, partial_paresis,
                    muscle_stiffness, alopecia, obesity, language
                ],
                outputs=screening_output
            )

            clear_button.click(
                fn=lambda: (
                    None, None, None, None, None, None,
                    None, None, None, None, None, None,
                    None, None, None, None, ""
                ),
                outputs=[
                    age, gender, polyuria, polydipsia,
                    sudden_weight_loss, weakness, polyphagia,
                    genital_thrush, visual_blurring, itching,
                    irritability, delayed_healing, partial_paresis,
                    muscle_stiffness, alopecia, obesity,
                    screening_output
                ]
            )

        # REPORT SCANNING

        with gr.Tab("📄 Scan & Analyse"):

            scan_title = gr.Markdown(
                "## Medical Report Analysis"
            )

            scan_info = gr.Markdown(
                "Upload a PDF or image of your medical report "
                "for text extraction and basic analysis."
            )

            report_file = gr.File(
                label="Upload Your Report",
                file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                type="filepath"
            )

            scan_button = gr.Button(
                "Analyse Report",
                variant="primary"
            )

            report_output = gr.Markdown()

            scan_button.click(
                fn=scan_report,
                inputs=[report_file, language],
                outputs=report_output
            )

        # CHATBOT

        with gr.Tab("💬 Talk to DiaCare"):

            chat_title = gr.Markdown(
                "## Talk to DiaCare"
            )

            chat_info = gr.Markdown(
                "Ask general questions about diabetes, symptoms "
                "and healthy habits."
            )

            chatbot = gr.Chatbot(
                type="messages",
                height=350
            )

            with gr.Row():

                chat_message = gr.Textbox(
                    placeholder="Type your message here",
                    show_label=False,
                    scale=5
                )

                send_button = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1
                )

            send_button.click(
                fn=talk_to_diacare,
                inputs=[chat_message, chatbot, language],
                outputs=[chat_message, chatbot]
            )

            chat_message.submit(
                fn=talk_to_diacare,
                inputs=[chat_message, chatbot, language],
                outputs=[chat_message, chatbot]
            )

        # GLUCOSE MONITORING

        with gr.Tab("📊 Connect & Monitor"):

            monitor_title = gr.Markdown(
                "## Glucose Monitoring"
            )

            glucose_info = gr.Markdown(
                "Record your blood glucose readings and monitor your entries."
            )

            glucose_value = gr.Number(
                label="Blood Glucose (mg/dL)",
                minimum=1,
                maximum=1000
            )

            glucose_button = gr.Button(
                "Add Reading",
                variant="primary"
            )

            glucose_message = gr.Markdown()

            glucose_state = gr.State([])

            glucose_table = gr.Dataframe(
                headers=[
                    "Date",
                    "Glucose Reading (mg/dL)"
                ],
                datatype=["str", "number"],
                interactive=False
            )

            def update_glucose(value, readings, selected_language):

                new_readings, message = add_glucose_reading(
                    value, readings, selected_language
                )

                dataframe = pd.DataFrame(
                    new_readings,
                    columns=[
                        "Date",
                        "Glucose Reading (mg/dL)"
                    ]
                )

                return (
                    new_readings,
                    dataframe,
                    message,
                    None
                )

            glucose_button.click(
                fn=update_glucose,
                inputs=[
                    glucose_value,
                    glucose_state,
                    language
                ],
                outputs=[
                    glucose_state,
                    glucose_table,
                    glucose_message,
                    glucose_value
                ]
            )

    # MEDICAL DISCLAIMER

    disclaimer = gr.Markdown(
        """
        ⚠️ **Disclaimer:** DiaCare AI is an educational screening
        prototype, not a medical diagnostic tool. It cannot replace
        professional medical advice, diagnosis or treatment.
        """
    )

    # LANGUAGE CHANGE EVENT

    language.change(
        fn=update_language,
        inputs=[language],
        outputs=[
            language,
            header_title,
            header_subtitle,
            header_description,
            home_welcome,
            home_text,
            screening_title,
            screening_info,
            age,
            gender,
            analyse_button,
            clear_button,
            screening_output,
            scan_title,
            scan_info,
            report_file,
            scan_button,
            chat_title,
            chat_info,
            chat_message,
            send_button,
            monitor_title,
            glucose_info,
            glucose_value,
            glucose_button,
            disclaimer
        ]
    )


# =========================================================
# 10. LAUNCH APPLICATION
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
