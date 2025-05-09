from flask import Flask, render_template, request
import joblib
import numpy as np
import json
from utils import transform_user_input

# Carica modello e altri oggetti
model = joblib.load('C:/Users/Ciro/Desktop/CorsoPython/pythonProgettiGruppo/PredizioneDellObesita/SvModelli/model_xgb.pkl')
scaler = joblib.load('C:/Users/Ciro/Desktop/CorsoPython/pythonProgettiGruppo/PredizioneDellObesita/SvModelli/scaler.pkl')
feature_names = joblib.load('C:/Users/Ciro/Desktop/CorsoPython/pythonProgettiGruppo/PredizioneDellObesita/SvModelli/feature_names.pkl')
with open('C:/Users/Ciro/Desktop/CorsoPython/pythonProgettiGruppo/PredizioneDellObesita/SvModelli/class_map.json') as f:
    class_map = {int(v): k for k, v in json.load(f).items()}

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", title="Home")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        try:
            data_scaled = transform_user_input(request.form, scaler, feature_names)
            pred = model.predict(data_scaled)[0]
            prediction = class_map.get(pred, "Unknown")
        except Exception as e:
            print(f"[Errore Predict]: {e}")
            prediction = f"Errore: {e}"
    return render_template("predict.html", prediction=prediction, features=feature_names)

@app.route("/data")
def data():
    return render_template("data.html", title="Analisi Dati")

if __name__ == "__main__":
    app.run(debug=True)

