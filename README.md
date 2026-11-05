# DiaCare AI 🩺💚
### AI-Powered Diabetes Risk Screening and Awareness

## 1. About the Project

DiaCare AI is an AI-powered diabetes screening and awareness application designed to help people understand potential diabetes risk indicators and encourage early health awareness.

The application provides an accessible interface for screening, medical report text extraction, health-related questions, and glucose reading monitoring.

**Important:** DiaCare AI is an educational screening prototype. It does not diagnose diabetes or replace professional medical advice.

## 2. Key Features

### 🩺 Diabetes Screening
- Interactive health questionnaire.
- Uses a trained Random Forest machine learning model.
- Displays a screening indication based on user responses.

### 📄 Scan & Analyse
- Upload PDF, JPG, or PNG reports.
- Extracts report text.
- Identifies possible glucose and HbA1c references.

### 💬 Talk to DiaCare
- Answers basic diabetes awareness questions.
- Provides general information about symptoms, prevention, food, and blood glucose.

### 📊 Connect & Monitor
- Allows users to enter glucose readings.
- Displays recorded readings in a table.
- Supports basic personal monitoring during the application session.

## 3. Technologies Used

- Python
- Gradio
- Pandas
- NumPy
- Scikit-learn
- Joblib
- PyPDF
- Pytesseract
- Pillow
- Google Colab

## 4. Machine Learning Model

DiaCare AI uses a Random Forest Classifier.

The model was trained using the UCI Early Stage Diabetes Risk Prediction dataset.

The dataset contains 520 original records. After removing duplicate records, 251 unique records remained.

### Preliminary Model Results

- Random Forest test accuracy: 94.12%
- Cross-validation mean accuracy: 93.50%
- Positive-class recall: 94.95% (cross-validation mean)

These results are preliminary and are not evidence of clinical accuracy. The dataset is relatively small, originates from Bangladesh, and requires further independent and local validation.

## 5. How to Run

The application was developed using Google Colab.

1. Open the DiaCare AI notebook in Google Colab.
2. Upload the trained model file `diacare_rf_model.pkl`.
3. Install the required libraries.
4. Run the notebook cells in order.
5. Launch the Gradio application.
6. Open the generated public demo link.

## 6. Ethical Considerations

- The application is not a medical diagnostic tool.
- Users should consult qualified healthcare professionals for medical advice.
- The model requires further testing and validation before real-world clinical use.
- No real patient data should be used for the hackathon demonstration.
- Screening results should not be treated as confirmed medical conditions.

## 7. Future Improvements

- Support for English, French, and Mauritian Creole.
- Improved medical report extraction.
- Voice-based interaction.
- Persistent glucose monitoring.
- Local validation using appropriately approved datasets.

## 8. Project

**Project Name:** DiaCare AI  
**Hackathon:** Tech Wave 2026  
**Track:** Diabetes Risk Screening

Developed as an AI-powered prototype to encourage early diabetes awareness and screening.
