# Importazione delle librerie necessarie
from flask import Flask, render_template, request
import joblib
import numpy as np
import json
from utils import transform_user_input  # Funzione personalizzata per trasformare l'input dell'utente

# Caricamento del modello di machine learning e degli oggetti necessari (scaler, nomi delle feature, mappa delle classi)
model = joblib.load('SvModelli/model_xgb.pkl')
scaler = joblib.load('SvModelli/scaler.pkl')
feature_names = joblib.load('SvModelli/feature_names.pkl')

# Caricamento della mappa che associa i numeri di classe alle etichette testuali
with open('SvModelli/class_map.json') as f:
    class_map = {int(v): k for k, v in json.load(f).items()}

# Creazione dell'app Flask, specificando le cartelle per i template e i file statici
app = Flask(__name__, template_folder="templates", static_folder="static")

# Rotta per la home page
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", title="Home")  # Ritorna la pagina home.html

# Rotta per la pagina di predizione
@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None  # Inizializza la variabile della predizione
    if request.method == "POST":
        try:
            # Trasforma l'input dell'utente e lo scala
            data_scaled = transform_user_input(request.form, scaler, feature_names)
            # Effettua la predizione con il modello
            pred = model.predict(data_scaled)[0]
            # Converte la predizione numerica in etichetta testuale
            prediction = class_map.get(pred, "Unknown")
        except Exception as e:
            # Stampa l'errore e mostra un messaggio nella pagina
            print(f"[Errore Predict]: {e}")
            prediction = f"Errore: {e}"
    # Ritorna la pagina predict.html con il risultato della predizione
    return render_template("predict.html", prediction=prediction, features=feature_names)

# Rotta per la pagina di analisi dati
@app.route("/data")
def data():
    return render_template("data.html", title="Analisi Dati")  # Ritorna la pagina data.html

# Esecuzione dell'app in modalità debug (utile per lo sviluppo)
if __name__ == "__main__":
    app.run(debug=True)


