import os
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
    muscle_stiffness, alopecia, obesity, language="English"
):

    translations = {
        "English": {
            "missing": "Please enter your age and gender.",
            "answers": "Please answer all the screening questions.",
            "positive": """## Higher Screening Indication

Your responses indicate a higher screening indication.

Please consult a qualified healthcare professional
for appropriate medical assessment.

**Remember:** This is not a medical diagnosis.""",
            "negative": """## Lower Screening Indication

Your responses indicate a lower screening indication.
This does not rule out diabetes. Consult a healthcare
professional if you have concerns or symptoms.

**Remember:** This is not a medical diagnosis."""
        },

        "French": {
            "missing": "Veuillez saisir votre âge et votre sexe.",
            "answers": "Veuillez répondre à toutes les questions.",
            "positive": """## Indication de dépistage plus élevée

Vos réponses indiquent une possibilité plus élevée de risque.

Veuillez consulter un professionnel de santé
pour une évaluation médicale appropriée.

**Rappel :** Ceci n'est pas un diagnostic médical.""",
            "negative": """## Indication de dépistage plus faible

Vos réponses indiquent une possibilité plus faible de risque.
Cela n'exclut pas le diabète. Consultez un professionnel
de santé si vous avez des inquiétudes ou des symptômes.

**Rappel :** Ceci n'est pas un diagnostic médical."""
        },

        "Creole": {
            "missing": "Silvouple rantre to laz ek to sex.",
            "answers": "Silvouple reponn tou bann kestion depistaz.",
            "positive": """## Indikasion depistaz pli eleve

To bann repons montre enn indikasion depistaz pli eleve.

Silvouple konsilte enn profesione lasante
pou enn evaliasion medikal apropriye.

**Rapel:** Sa pa enn diagnos medikal.""",
            "negative": """## Indikasion depistaz pli ba

To bann repons montre enn indikasion depistaz pli ba.
Sa pa vedir ki to pena diabet. Konsilte enn profesione
lasante si to ena bann traka ouswa bann sinptom.

**Rapel:** Sa pa enn diagnos medikal."""
        }
    }

    text = translations.get(language, translations["English"])

    if age is None or gender is None:
        return text["missing"]

    answers = [
        polyuria, polydipsia, sudden_weight_loss,
        weakness, polyphagia, genital_thrush,
        visual_blurring, itching, irritability,
        delayed_healing, partial_paresis,
        muscle_stiffness, alopecia, obesity
    ]

    if any(answer is None for answer in answers):
        return text["answers"]

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
        return text["positive"]

    return text["negative"]

# ==========================================
# 3. SCAN AND ANALYSE REPORTS
# ==========================================

def scan_report(file, language="English"):

    translations = {
        "English": {
            "missing": "Please upload a report first.",
            "pdf": "No readable text was found in this PDF. Please upload a clear JPG or PNG image instead.",
            "format": "Please upload a PDF, JPG or PNG file.",
            "none": "No glucose or HbA1c references were identified.",
            "title": "Report Extraction Complete!",
            "references": "Possible glucose and HbA1c references",
            "extracted": "Extracted Report Text",
            "important": "Important: This is automated text extraction, not medical interpretation. Please verify extracted values and units against the original report.",
            "error": "Unable to read the report:"
        },

        "French": {
            "missing": "Veuillez d'abord télécharger un rapport.",
            "pdf": "Aucun texte lisible n'a été trouvé dans ce PDF. Veuillez plutôt télécharger une image JPG ou PNG claire.",
            "format": "Veuillez télécharger un fichier PDF, JPG ou PNG.",
            "none": "Aucune référence au glucose ou à l'HbA1c n'a été identifiée.",
            "title": "Extraction du rapport terminée !",
            "references": "Références possibles au glucose et à l'HbA1c",
            "extracted": "Texte extrait du rapport",
            "important": "Important : Il s'agit d'une extraction automatique de texte, et non d'une interprétation médicale. Veuillez vérifier les valeurs et les unités sur le rapport original.",
            "error": "Impossible de lire le rapport :"
        },

        "Creole": {
            "missing": "Silvouple sarz enn rapor avan.",
            "pdf": "Pa finn trouv okenn text ki kapav lir dan sa PDF-la. Silvouple sarz enn zimaz JPG ouswa PNG ki bien kler.",
            "format": "Silvouple sarz enn fichier PDF, JPG ouswa PNG.",
            "none": "Pa finn trouv okenn referans lor glikoz ouswa HbA1c.",
            "title": "Ekstraksion rapor fini !",
            "references": "Referans posib lor glikoz ek HbA1c",
            "extracted": "Text ki finn tire depi rapor",
            "important": "Inportan: Sa se enn ekstraksion text otomatik, pa enn interpretasion medikal. Silvouple verifie bann valer ek inite lor rapor orizinal.",
            "error": "Pa finn kapav lir rapor-la:"
        }
    }

    text = translations.get(language, translations["English"])

    if file is None:
        return text["missing"]

    try:
        filename = file.lower()

        if filename.endswith(".pdf"):

            reader = PdfReader(file)
            extracted_text = ""

            for page in reader.pages:
                extracted_text += page.extract_text() or ""

            if not extracted_text.strip():
                return text["pdf"]

        elif filename.endswith((".png", ".jpg", ".jpeg")):

            image = Image.open(file)
            extracted_text = pytesseract.image_to_string(image)

        else:
            return text["format"]

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
            else text["none"]
        )

        return f"""
## {text["title"]}

### {text["references"]}

{findings}

### {text["extracted"]}

{extracted_text[:4000]}

---

**{text["important"]}**
"""

    except Exception as error:
        return f'{text["error"]} {str(error)}'

# ==========================================
# 4. TALK TO DIACARE CHATBOT
# ==========================================

def talk_to_diacare(message, history, language="English"):

    if not message:
        return history or [], ""

    translations = {
        "English": {
            "emergency": "⚠️ **Please seek emergency medical help immediately.** These symptoms may indicate a serious medical problem. Do not rely on this chatbot for emergency assistance.",
            "symptoms": """### Common Diabetes Symptoms

- Frequent urination
- Excessive thirst
- Increased hunger
- Unexplained weight loss
- Blurred vision
- Unusual tiredness

These symptoms can have other causes too. Consult a healthcare professional for assessment.""",
            "hba1c": """### What is HbA1c?

HbA1c is a blood test that measures average blood glucose over approximately 2–3 months.

A healthcare professional can explain your result using appropriate clinical reference ranges.""",
            "glucose": """### What is Blood Glucose?

Blood glucose is the amount of sugar present in your blood.

It is an important measurement when screening for and monitoring diabetes.""",
            "prevention": """### Healthy Habits

- Eat a balanced diet.
- Stay physically active.
- Maintain a healthy weight.
- Get enough sleep.
- Attend regular health check-ups.""",
            "food": """### Healthy Eating Tips

- Include vegetables and fruits.
- Choose whole grains.
- Include healthy protein sources.
- Limit sugary drinks.
- Pay attention to portion sizes.""",
            "doctor": """### When Should You Consult a Doctor?

Consider consulting a healthcare professional if you experience excessive thirst, frequent urination, unexplained weight loss or persistent tiredness.""",
            "welcome": """Hello! 👋 I'm DiaCare, your diabetes health-awareness assistant.

I can help you learn about:
- Diabetes symptoms
- HbA1c
- Blood glucose
- Prevention
- Healthy eating
- When to consult a doctor

I provide general health information, not medical diagnosis or treatment."""
        },

        "French": {
            "emergency": "⚠️ **Veuillez demander immédiatement une aide médicale d'urgence.** Ces symptômes peuvent indiquer un problème médical grave. Ne comptez pas sur ce chatbot en cas d'urgence.",
            "symptoms": """### Symptômes courants du diabète

- Mictions fréquentes
- Soif excessive
- Faim accrue
- Perte de poids inexpliquée
- Vision floue
- Fatigue inhabituelle

Ces symptômes peuvent avoir d'autres causes. Consultez un professionnel de santé.""",
            "hba1c": """### Qu'est-ce que l'HbA1c ?

L'HbA1c est une analyse sanguine qui mesure la glycémie moyenne sur environ 2 à 3 mois.

Un professionnel de santé peut expliquer votre résultat selon les valeurs de référence appropriées.""",
            "glucose": """### Qu'est-ce que la glycémie ?

La glycémie représente la quantité de sucre présente dans le sang.

Elle est importante pour le dépistage et le suivi du diabète.""",
            "prevention": """### Habitudes saines

- Adoptez une alimentation équilibrée.
- Restez physiquement actif.
- Maintenez un poids sain.
- Dormez suffisamment.
- Effectuez des contrôles médicaux réguliers.""",
            "food": """### Conseils alimentaires

- Mangez des fruits et des légumes.
- Choisissez des céréales complètes.
- Consommez des protéines saines.
- Limitez les boissons sucrées.
- Faites attention aux portions.""",
            "doctor": """### Quand consulter un médecin ?

Consultez un professionnel de santé si vous avez une soif excessive, des mictions fréquentes, une perte de poids inexpliquée ou une fatigue persistante.""",
            "welcome": """Bonjour ! 👋 Je suis DiaCare, votre assistant de sensibilisation au diabète.

Je peux vous renseigner sur :
- Les symptômes du diabète
- L'HbA1c
- La glycémie
- La prévention
- L'alimentation saine
- Les consultations médicales

Je fournis des informations générales, pas de diagnostic ni de traitement médical."""
        },

        "Creole": {
            "emergency": "⚠️ **Silvouple rod led medikal irzan deswit.** Sa bann sinptom-la kapav montre enn problem medikal grav. Pa kont lor sa chatbot-la pou enn irzans.",
            "symptoms": """### Bann sinptom diabet ki souvan trouve

- Pipi souvan
- Swaf boukou
- Fin souvan
- Perdi pwa san explikasion
- Vizyon flou
- Fatig ki pa abitie

Sa bann sinptom-la kapav ena lezot lakoz osi. Konsilte enn profesione lasante.""",
            "hba1c": """### Ki ete HbA1c ?

HbA1c se enn test disan ki mezire to nivo sik mwayin dan disan pandan apepre 2 a 3 mwa.

Enn profesione lasante kapav explik to rezilta avek bann valer referans apropriye.""",
            "glucose": """### Ki ete glikoz dan disan ?

Glikoz dan disan se kantite sik ki ena dan to disan.

Li inportan pou depistaz ek swivi diabet.""",
            "prevention": """### Bann bon labitid

- Manz enn manze ekilibre.
- Fer aktivite fizik.
- Gard enn pwa ki bon pou lasante.
- Dormi ase.
- Fer bann kontrol medikal regilier.""",
            "food": """### Konsey pou manze bien

- Manz legim ek fri.
- Swazir sereyal konplet.
- Manz bann protein ki bon pou lasante.
- Limit bann bwar ki ena boukou disik.
- Fer atansion ar kantite manze.""",
            "doctor": """### Kan bizin al get dokter ?

Konsilte enn profesione lasante si to swaf boukou, al pipi souvan, perdi pwa san explikasion ouswa santi fatig toultan.""",
            "welcome": """Bonzour ! 👋 Mo apel DiaCare. Mo la pou ed twa konn plis lor diabet.

Mo kapav ed twa konpran :
- Bann sinptom diabet
- HbA1c
- Glikoz dan disan
- Prevansion
- Manze ki bon pou lasante
- Kan bizin al get dokter

Mo donn informasion zeneral, pa fer diagnos ouswa tretman medikal."""
        }
    }

    text = translations.get(language, translations["English"])
    question = message.lower()

    if any(word in question for word in [
        "chest pain", "difficulty breathing", "unconscious",
        "fainting", "douleur thoracique", "difficulté à respirer",
        "perte de connaissance", "douleur poitrine",
        "difikilte respire", "perdi konesans"
    ]):
        answer = text["emergency"]

    elif any(word in question for word in [
        "symptom", "sign", "symptôme", "sinptom"
    ]):
        answer = text["symptoms"]

    elif any(word in question for word in [
        "hba1c", "a1c"
    ]):
        answer = text["hba1c"]

    elif any(word in question for word in [
        "glucose", "blood sugar", "glycémie", "sucre dans le sang",
        "glikoz", "sik dan disan"
    ]):
        answer = text["glucose"]

    elif any(word in question for word in [
        "prevent", "prevention", "prévention", "prevention",
        "prevansion"
    ]):
        answer = text["prevention"]

    elif any(word in question for word in [
        "food", "eat", "manger", "alimentation",
        "manze", "manze"
    ]):
        answer = text["food"]

    elif any(word in question for word in [
        "doctor", "consult", "médecin", "docteur",
        "dokter"
    ]):
        answer = text["doctor"]

    else:
        answer = text["welcome"]

    history = history or []

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return history, ""

# ==========================================
# 5. CONNECT AND MONITOR
# ==========================================

def add_glucose_reading(value, unit, note, history, language="English"):

    translations = {
        "English": {
            "invalid": "Please enter a valid glucose reading.",
            "success": "Your glucose reading has been added successfully!",
            "no_note": "No note"
        },
        "French": {
            "invalid": "Veuillez saisir une mesure de glucose valide.",
            "success": "Votre mesure de glucose a été ajoutée avec succès !",
            "no_note": "Aucune remarque"
        },
        "Creole": {
            "invalid": "Silvouple rantre enn mezir glikoz ki valid.",
            "success": "To mezir glikoz finn azoute avek sikse !",
            "no_note": "Okenn not"
        }
    }

    text = translations.get(language, translations["English"])

    history = history or []

    if value is None or value <= 0:
        return text["invalid"], history, pd.DataFrame(history)

    reading = {
        "Date": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "Glucose Reading": value,
        "Unit": unit,
        "Note": note if note else text["no_note"]
    }

    history.append(reading)

    return (
        text["success"],
        history,
        pd.DataFrame(history)
    )

# ==========================================
# 6. APPLICATION DESIGN
# ==========================================
# ==========================================
# LANGUAGE TRANSLATIONS
# ==========================================

TRANSLATIONS = {
    "English": {
        "language": "🌍 Select Your Language",
        "home": "Home",
        "screening": "Screening",
        "scan": "Scan & Analyse",
        "chat": "Talk to DiaCare",
        "monitor": "Connect & Monitor",
        "welcome": "Welcome to DiaCare AI! 💚",
        "tagline": "Your diabetes health-awareness companion.",
        "analyse": "Analyse My Responses",
        "scan_button": "Scan Report",
        "upload": "Upload Medical Report",
        "send": "Send",
        "message": "Your Message",
        "add": "Add Reading",
        "reading": "Glucose Reading",
        "history": "Reading History",
        "note": "Note",
        "unit": "Unit",
        "age": "Age",
        "gender": "Gender",
        "frequent_urination": "Frequent urination",
        "excessive_thirst": "Excessive thirst",
        "sudden_weight_loss": "Sudden weight loss",
        "weakness": "Weakness",
        "excessive_hunger": "Excessive hunger",
        "genital_thrush": "Genital thrush",
        "blurred_vision": "Blurred vision",
        "itching": "Itching",
        "irritability": "Irritability",
        "delayed_healing": "Delayed healing",
        "partial_paresis": "Partial paresis",
        "muscle_stiffness": "Muscle stiffness",
        "hair_loss": "Hair loss",
        "obesity": "Obesity",
        "yes": "Yes",
        "no": "No",
        "disclaimer": "DiaCare AI is an educational screening prototype. It does not diagnose diabetes or replace professional medical advice.",
        "header_content": """
# 💚 DiaCare AI

### AI-Powered Diabetes Risk Screening and Awareness

Helping people understand diabetes risk indicators
and encouraging early health awareness.

**Disclaimer:** DiaCare AI is an educational screening
prototype. It does not diagnose diabetes or replace
professional medical advice.
""",
        "home_content": """
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
""",
        "screening_content": """
## Diabetes Risk Screening

Please answer all questions below.
""",
        "scan_content": """
## Scan & Analyse Your Report

Upload a PDF or image of a report to extract
possible glucose and HbA1c references.
""",
        "chat_content": """
## Talk to DiaCare 💬

Ask questions about diabetes awareness,
symptoms, HbA1c, glucose and prevention.
""",
        "monitor_content": """
## Connect & Monitor 📊

Record your glucose readings and view your
entries during this application session.
"""
    },

    "French": {
        "language": "🌍 Choisissez votre langue",
        "home": "Accueil",
        "screening": "Dépistage",
        "scan": "Analyser un rapport",
        "chat": "Discuter avec DiaCare",
        "monitor": "Suivi du glucose",
        "welcome": "Bienvenue sur DiaCare AI ! 💚",
        "tagline": "Votre compagnon de sensibilisation au diabète.",
        "analyse": "Analyser mes réponses",
        "scan_button": "Analyser le rapport",
        "upload": "Télécharger un rapport médical",
        "send": "Envoyer",
        "message": "Votre message",
        "add": "Ajouter une mesure",
        "reading": "Mesure de glucose",
        "history": "Historique des mesures",
        "note": "Remarque",
        "unit": "Unité",
        "age": "Âge",
        "gender": "Sexe",
        "frequent_urination": "Mictions fréquentes",
        "excessive_thirst": "Soif excessive",
        "sudden_weight_loss": "Perte de poids soudaine",
        "weakness": "Faiblesse",
        "excessive_hunger": "Faim excessive",
        "genital_thrush": "Mycose génitale",
        "blurred_vision": "Vision floue",
        "itching": "Démangeaisons",
        "irritability": "Irritabilité",
        "delayed_healing": "Cicatrisation lente",
        "partial_paresis": "Parésie partielle",
        "muscle_stiffness": "Raideur musculaire",
        "hair_loss": "Perte de cheveux",
        "obesity": "Obésité",
        "yes": "Oui",
        "no": "Non",
        "disclaimer": "DiaCare AI est un prototype éducatif de dépistage. Il ne diagnostique pas le diabète et ne remplace pas les conseils d'un professionnel de santé.",
                "header_content": """
# 💚 DiaCare AI

### Dépistage et sensibilisation au diabète grâce à l'IA

Aider chacun à comprendre les indicateurs de risque du diabète
et encourager une meilleure sensibilisation à la santé.

**Avertissement :** DiaCare AI est un prototype éducatif de dépistage.
Il ne diagnostique pas le diabète et ne remplace pas les conseils médicaux.
""",
        "home_content": """
# Bienvenue sur DiaCare AI ! 💚

Votre compagnon de sensibilisation au diabète.

### Que peut faire DiaCare AI ?

🩺 **Dépistage :** Répondez à un questionnaire de santé interactif.

📄 **Analyser un rapport :** Extraire les références possibles au glucose et à l'HbA1c.

💬 **Discuter avec DiaCare :** Posez des questions générales sur le diabète.

📊 **Suivi :** Enregistrez vos mesures de glucose.

### Notre mission

Encourager la sensibilisation précoce au diabète et faciliter
l'accès aux informations de santé.
""",
        "screening_content": """
## Dépistage du risque de diabète

Veuillez répondre à toutes les questions ci-dessous.
""",
        "scan_content": """
## Analyser votre rapport

Téléchargez un rapport PDF ou une image pour extraire
les références possibles au glucose et à l'HbA1c.
""",
        "chat_content": """
## Discutez avec DiaCare 💬

Posez des questions sur le diabète, ses symptômes,
l'HbA1c, la glycémie et la prévention.
""",
        "monitor_content": """
## Suivi du glucose 📊

Enregistrez vos mesures de glucose et consultez
vos entrées pendant cette session.
"""
    },
    "Creole": {
        "language": "🌍 Swazir to lang",
        "home": "Lakaz",
        "screening": "Depistaz",
        "scan": "Analiz Rapor",
        "chat": "Koze avek DiaCare",
        "monitor": "Swivi Glikoz",
        "welcome": "Bienveni lor DiaCare AI ! 💚",
        "tagline": "To konpagnon pou konn plis lor diabet.",
        "analyse": "Analiz mo bann repons",
        "scan_button": "Analiz Rapor",
        "upload": "Sarz enn rapor medikal",
        "send": "Avoy",
        "message": "To mesaz",
        "add": "Azout enn mezir",
        "reading": "Mezir Glikoz",
        "history": "Listwar bann mezir",
        "note": "Not",
        "unit": "Inite",
        "age": "Laz",
        "gender": "Sex",
        "frequent_urination": "Pipi souvan",
        "excessive_thirst": "Swaf boukou",
        "sudden_weight_loss": "Perdi pwa enn kout",
        "weakness": "Febles",
        "excessive_hunger": "Fin boukou",
        "genital_thrush": "Mykoz jenital",
        "blurred_vision": "Vizyon flou",
        "itching": "Demanzaison",
        "irritability": "Iritabilite",
        "delayed_healing": "Blesir pran letan pou geri",
        "partial_paresis": "Paralizi parsyel",
        "muscle_stiffness": "Ridir misk",
        "hair_loss": "Perdi seve",
        "obesity": "Lobezite",
        "yes": "Wi",
        "no": "Non",
        "disclaimer": "DiaCare AI se enn prototip edikatif pou depistaz. Li pa fer diagnos diabet ek li pa ranplas konsey enn profesione lasante.",
                "header_content": """
# 💚 DiaCare AI

### Depistaz ek sansibilizasion lor diabet avek IA

Ed dimounn konpran bann sign ki kapav montre risk diabet
ek ankouraz zot pran swin zot lasante.

**Avertisman:** DiaCare AI se enn prototip edikatif pou depistaz.
Li pa fer diagnos diabet ek li pa ranplas konsey medikal.
""",
        "home_content": """
# Bienveni lor DiaCare AI ! 💚

To konpagnon pou konn plis lor diabet.

### Ki DiaCare AI kapav fer ?

🩺 **Depistaz :** Ranpli enn kestioner lor to lasante.

📄 **Analiz rapor :** Tir bann referans posib lor glikoz ek HbA1c dan bann rapor.

💬 **Koze avek DiaCare :** Poz kestion zeneral lor diabet.

📊 **Swivi :** Anrezistre to bann mezir glikoz.

### Nou misyon

Ankouraz dimounn pran konsians diabet boner ek rann
informasion lor lasante pli fasil pou gagne.
""",
        "screening_content": """
## Depistaz risk diabet

Silvouple reponn tou bann kestion anba.
""",
        "scan_content": """
## Analiz to rapor

Sarz enn rapor PDF ouswa enn zimaz pou tir
bann referans posib lor glikoz ek HbA1c.
""",
        "chat_content": """
## Koze avek DiaCare 💬

Poz kestion lor diabet, so bann sinptom,
HbA1c, glikoz ek prevansion.
""",
        "monitor_content": """
## Swivi Glikoz 📊

Anrezistre to bann mezir glikoz ek get
to bann lanrezistreman pandan sa sesion-la.
"""
    }
}
def translate(language, key):
    return TRANSLATIONS.get(
        language,
        TRANSLATIONS["English"]
    ).get(
        key,
        TRANSLATIONS["English"].get(key, key)
    )


def update_language(language):

    yes = translate(language, "yes")
    no = translate(language, "no")

    gender_choices = [
        ({"English": "Female", "French": "Femme", "Creole": "Fam"}[language], "Female"),
        ({"English": "Male", "French": "Homme", "Creole": "Zom"}[language], "Male")
    ]

    answer_choices = [
        (yes, "Yes"),
        (no, "No")
    ]

    return (
        gr.update(label=translate(language, "age")),
        gr.update(label=translate(language, "gender"), choices=gender_choices),

        gr.update(label=translate(language, "frequent_urination"), choices=answer_choices),
        gr.update(label=translate(language, "excessive_thirst"), choices=answer_choices),
        gr.update(label=translate(language, "sudden_weight_loss"), choices=answer_choices),
        gr.update(label=translate(language, "weakness"), choices=answer_choices),
        gr.update(label=translate(language, "excessive_hunger"), choices=answer_choices),
        gr.update(label=translate(language, "genital_thrush"), choices=answer_choices),
        gr.update(label=translate(language, "blurred_vision"), choices=answer_choices),
        gr.update(label=translate(language, "itching"), choices=answer_choices),
        gr.update(label=translate(language, "irritability"), choices=answer_choices),
        gr.update(label=translate(language, "delayed_healing"), choices=answer_choices),
        gr.update(label=translate(language, "partial_paresis"), choices=answer_choices),
        gr.update(label=translate(language, "muscle_stiffness"), choices=answer_choices),
        gr.update(label=translate(language, "hair_loss"), choices=answer_choices),
        gr.update(label=translate(language, "obesity"), choices=answer_choices),

        gr.update(value=translate(language, "analyse")),
        gr.update(label=translate(language, "upload")),
        gr.update(value=translate(language, "scan_button")),
        gr.update(label=translate(language, "message")),
        gr.update(value=translate(language, "send")),
        gr.update(label=translate(language, "reading")),
        gr.update(label=translate(language, "unit")),
        gr.update(label=translate(language, "note")),
        gr.update(value=translate(language, "add")),

        gr.update(label={
            "English": "DiaCare Assistant",
            "French": "Assistant DiaCare",
            "Creole": "Asistan DiaCare"
        }[language]),

        gr.update(
            label=translate(language, "history")
        ),

        gr.update(value=translate(language, "header_content")),
        gr.update(value=translate(language, "home_content")),
        gr.update(value=translate(language, "screening_content")),
        gr.update(value=translate(language, "scan_content")),
        gr.update(value=translate(language, "chat_content")),
        gr.update(value=translate(language, "monitor_content"))
    )
    
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
    # MAIN HEADER
    header_content = gr.Markdown("""
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

             home_content = gr.Markdown("""
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

           screening_content = gr.Markdown("""
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
                   muscle_stiffness, alopecia, obesity, language
                ],
                outputs=screening_result
            )

        # SCAN AND ANALYSE
        with gr.Tab("Scan & Analyse"):

           scan_content = gr.Markdown("""
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
                inputs=[report_file, language],
                outputs=scan_result
            )

        # TALK TO DIACARE
        with gr.Tab("Talk to DiaCare"):

           chat_content = gr.Markdown("""
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
                inputs=[message, chatbot, language],
                outputs=[chatbot, message]
            )

            message.submit(
                talk_to_diacare,
                inputs=[message, chatbot, language],
                outputs=[chatbot, message]
            )

        # CONNECT AND MONITOR
        with gr.Tab("Connect & Monitor"):

           monitor_content = gr.Markdown("""
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
                    glucose_state,
                    language
                ],
                outputs=[
                    monitor_message,
                    glucose_state,
                    glucose_table
                ]
            )

     language.change(
         fn=update_language,
         inputs=[language],
         outputs=[
             age,
             gender,
             polyuria,
             polydipsia
             sudden_weight_loss,
             weakness,
             polyphagia,
             genital_thrush,
             visual_blurring,
             itching,
             irritability,
             delayed_healing,
             partial_paresis
             muscle_stiffness,
             alopecia,
             obesity,
             screen_button,
             report_file,
             scan_button,
             message,
             send_button,
             glucose_value,
             glucose_unit,
             glucose_note,
             monitor_button,
             chatbot,
             glucose_table,
             header_content,
             home_content,
             screening_content,
             scan_content,
             chat_content,
             monitor_content
         ]
     )




# ==========================================
# 8. LAUNCH APPLICATION
# ==========================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="teal",
            neutral_hue="slate"
        ),
        css=custom_css
    )
