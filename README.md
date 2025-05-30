# 🧠 Disease Prediction and Early Diagnosis

This project demonstrates the use of supervised machine learning models to predict the risk of **Cardiovascular Disease** and **Diabetes**, based on patient health data.

The app also supports **automated extraction of lab values** (e.g. glucose, cholesterol) from uploaded **medical reports (PDF or image)** using OCR (`EasyOCR`), enhancing accessibility and real-world utility.

---

## 🎯 Project Objectives

- Predict risk for Cardiovascular Disease and Diabetes based on clinical data
- Extract medical values automatically from scanned lab results
- Provide clean UI, visual risk indicators, and PDF report downloads
- Combine medical insight with intuitive user experience using **Streamlit**

---

## 📁 Datasets Used

### 1. [Cardiovascular Disease Dataset – Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- 70,000+ rows
- Features: age, gender, cholesterol, glucose, smoking, physical activity, etc.

### 2. [Diabetes Dataset – Kaggle](https://www.kaggle.com/datasets/nancyalaswad90/review)
- Over 1,000 rows
- Features: pregnancies, glucose, insulin, BMI, DPF, etc.

---

## ⚙️ Features

- Manual data input via web form
- Optional OCR-based value extraction (Glucose, Cholesterol)
- Risk visualization (progress bar + classification)
- PDF report generation with summary of input and prediction
- Modular code structure for easy scaling and debugging

---

## ⚙️ Techniques Used

- Preprocessing:
  - Handling missing values, BMI calculation
  - One-hot encoding for categorical variables
  - Feature scaling with `StandardScaler`
- Models:
  - **Logistic Regression**
  - **Decision Tree**
  - **Random Forest** (selected as final model)
- OCR Extraction:
  - `EasyOCR` with `pdf2image` and `PIL`
- Web UI:
  - Built with **Streamlit**
  - Generates personalized **PDF reports** with `FPDF`
- Evaluation Metrics:
  - Accuracy, F1-Score, Precision, Recall
  - **ROC AUC Score**

---

## 🧱 Project Structure

```
disease-prediction-app/
│
├── app.py                   # Streamlit frontend application
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
│
├── models/
│   ├── train_models.py      # Script for training & saving models
│   └── (models generated locally after running train_models.py)
```

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/mgedna/disease-prediction-app.git
   cd disease-prediction-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate model files locally**
   If the `models/` directory does not contain `.pkl` files, generate them by running:
   ```bash
   python models/train_models.py
   ```

   This will generate:
   ```
   models/
   ├── model_cardio.pkl
   ├── model_diabetes.pkl
   ├── scaler_cardio.pkl
   └── scaler_diabetes.pkl
   ```

4. **Launch the Streamlit app**
   ```bash
   streamlit run app.py
   ```

> Optional: You must have [Poppler](http://blog.alivate.com.au/poppler-windows/) installed for PDF OCR functionality via `pdf2image`.

---

## 🖼 Example Inputs

- Manual entry of health metrics via sliders and dropdowns
- Upload Synevo-style lab reports (e.g. "glucoză", "colesterol") for auto extraction
- Supports both `.pdf` and image files (`.jpg`, `.png`)

---

## 👤 Author

**Edna Memedula**  
📫 [LinkedIn](https://www.linkedin.com/in/edna-memedula-24b519245) • [GitHub](https://github.com/mgedna)
