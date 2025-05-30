import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)

def compare_models(X_train, X_test, y_train, y_test, dataset_name):
    models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        print(f"\n--- {dataset_name} - {name} ---")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC: {auc:.4f}")

        results.append({
            "Dataset": dataset_name,
            "Model": name,
            "ROC AUC": auc
        })
    return results

print("Loading Cardiovascular dataset...")
cv_df = pd.read_csv("cardio_train.csv", sep=';')
cv_df.drop(columns=['id'], inplace=True)
cv_df['age'] = (cv_df['age'] / 365).astype(int)
cv_df['bmi'] = cv_df.apply(lambda row: calculate_bmi(row['weight'], row['height']), axis=1)

cv_df = pd.get_dummies(cv_df, columns=['cholesterol', 'gluc'], drop_first=True)
X_cv = cv_df.drop(columns=['cardio'])
y_cv = cv_df['cardio']

scaler_cv = StandardScaler()
X_cv_scaled = scaler_cv.fit_transform(X_cv)
X_cv_train, X_cv_test, y_cv_train, y_cv_test = train_test_split(X_cv_scaled, y_cv, test_size=0.2, random_state=42, stratify=y_cv)

model_cv = RandomForestClassifier(n_estimators=100, random_state=42)
model_cv.fit(X_cv_train, y_cv_train)
y_cv_pred = model_cv.predict(X_cv_test)
y_cv_proba = model_cv.predict_proba(X_cv_test)[:, 1]

print("\nCardiovascular Model Report (Final Random Forest):")
print(classification_report(y_cv_test, y_cv_pred))
print("ROC AUC:", roc_auc_score(y_cv_test, y_cv_proba))

joblib.dump(model_cv, "model_cardio.pkl")
joblib.dump(scaler_cv, "scaler_cardio.pkl")

cv_results = compare_models(X_cv_train, X_cv_test, y_cv_train, y_cv_test, "Cardiovascular")

print("\nLoading Diabetes dataset...")
diabetes_df = pd.read_csv("diabetes.csv")
diabetes_df = diabetes_df.rename(columns={'BMI': 'bmi'})
X_db = diabetes_df.drop(columns=['Outcome'])
y_db = diabetes_df['Outcome']

scaler_db = StandardScaler()
X_db_scaled = scaler_db.fit_transform(X_db)
X_db_train, X_db_test, y_db_train, y_db_test = train_test_split(X_db_scaled, y_db, test_size=0.2, random_state=42, stratify=y_db)

model_db = RandomForestClassifier(n_estimators=100, random_state=42)
model_db.fit(X_db_train, y_db_train)
y_db_pred = model_db.predict(X_db_test)
y_db_proba = model_db.predict_proba(X_db_test)[:, 1]

print("\nDiabetes Model Report (Final Random Forest):")
print(classification_report(y_db_test, y_db_pred))
print("ROC AUC:", roc_auc_score(y_db_test, y_db_proba))

joblib.dump(model_db, "model_diabetes.pkl")
joblib.dump(scaler_db, "scaler_diabetes.pkl")

db_results = compare_models(X_db_train, X_db_test, y_db_train, y_db_test, "Diabetes")

