
# VitalPredict - Random Forest Diabetes Risk Prediction

Academic mini-project based on the supplied VitalPredict report.

## Architecture

Browser (HTML/CSS/JavaScript)
        |
        v
Flask REST API (/predict)
        |
        v
Preprocessing -> StandardScaler
        |
        v
Random Forest Classifier
        |
        v
Risk result + feature importance
        |
        v
Dashboard

## Project structure

VitalPredict_RF_Project/
|-- app.py
|-- train_model.py
|-- requirements.txt
|-- README.md
|-- data/
|   `-- diabetes.csv
|-- model/
|   `-- generated model files
|-- templates/
|   `-- index.html
`-- static/
    |-- css/style.css
    `-- js/app.js

## VS Code execution

1. Install Python 3.10+.
2. Open this folder in VS Code.
3. Open Terminal.
4. Create virtual environment:
   Windows:
       python -m venv venv
       venv\Scripts\activate
   macOS/Linux:
       python3 -m venv venv
       source venv/bin/activate

5. Install packages:
       pip install -r requirements.txt

6. Train the model:
       python train_model.py

7. Start Flask:
       python app.py

8. Open:
       http://127.0.0.1:5000

## Dataset

The included data/diabetes.csv is a SMALL DEMO DATASET so the website can execute immediately.
For your final academic evaluation, replace it with the Pima Indians Diabetes Dataset and retrain:
       python train_model.py

Required columns:
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
BMI, DiabetesPedigreeFunction, Age, Outcome

Outcome:
0 = no diabetes in the original Pima dataset
1 = diabetes in the original Pima dataset

The dashboard converts the model probability into the project's two screening bands:
Low Risk / Medium Risk.

This is a student research/demo system, not a medical diagnostic device.
