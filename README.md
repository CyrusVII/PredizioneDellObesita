# 🧠 Obesity Predictor Web App

Welcome to the **Obesity Prediction** project, developed by **Maresca Ciro** and **Filippo Giorgio Rondó**!  
This web app estimates an individual's obesity level based on personal data and habits, using an XGBoost model trained on real-world data.

---

## 🚀 Features

- 🧾 Interactive user data input form  
- 🤖 Automatic prediction of obesity level  
- 📊 Charts section with dataset and model analysis  
- 🌐 Multilanguage support (ITA/ENG)  
- 💾 Scalable architecture with saved models and separated backend/frontend  
---

## 📸 English 🇬🇧
Here are some screenshots of the application:

![Screenshot 2025-05-12 150338](https://github.com/user-attachments/assets/709906cd-04e3-4cfe-8498-978dc2e4f63f)

---

## 📁 Project Structure

```
PredizioneDellObesita/
│
├── DatiCsv/ # Dataset originale + previsioni salvate
│ ├── train.csv
│ ├── test.csv
│ └── predizioni_obesita.csv
│
├── SvModelli/ # Modelli e oggetti salvati (PKL/JSON)
│ ├── model_xgb.pkl
│ ├── scaler.pkl
│ ├── feature_names.pkl
│ ├── class_map.json
│ ├── Gender_encoder.pkl
│ ├── CAEC_encoder.pkl
│ ├── CALC_encoder.pkl
│ └── MTRANS_encoder.pkl
│
├── static/ # CSS e immagini statiche
│ ├── style.css
│ └── BackGround.css
│
├── templates/ # Template HTML (Jinja2)
│ ├── base.html
│ ├── home.html
│ ├── predict.html
│ ├── data.html
│ └── form.html
│
├── Img/ # Immagini per grafici o sfondi
│
├── app.py # Backend Flask
├── utils.py # Funzioni per la normalizzazione e encoding
├── prevObesita.py # Script di training e salvataggio modello
├── cleanerdata.ipynb # Notebook per analisi preliminare dei dati
└── README.md # Questo file
```

---

## 🧪 Model & Dataset

- Model: `XGBoostClassifier`  
- Feature engineering: categorical encoding, standard scaling  
- Target: multiclass classification (7 classes from "Insufficient Weight" to "Obesity Type III")  

---

## 🛠️ Quick Setup

```bash
git clone https://github.com/your-user/PredizioneDellObesita.git
cd PredizioneDellObesita
pip install -r requirements.txt
python app.py
```

## 👨‍💻 Authors

- [Ciro Maresca](https://github.com/CyrusVII)
  
- [Filippo Giorgio Rondó](https://linktr.ee/filippogiorgiorondo)

## ⚠️ Disclaimer
-- This project is for educational purposes only and is not a substitute for professional medical evaluation.



# ----------------------------

# 🧠 Obesity Predictor Web App

Benvenuti nel progetto **Predizione dell'Obesità**, realizzato da **Maresca Ciro** e **Filippo Giorgio Rondó**!  
Questa web app consente di stimare il livello di obesità di un individuo sulla base di dati personali e abitudini, utilizzando un modello XGBoost addestrato su un dataset reale.



## 🚀 Funzionalità

- 🧾 Inserimento dati utente tramite form interattivo
- 🤖 Predizione automatica del livello di obesità
- 📊 Sezione grafici con analisi del dataset e modello
- 🌐 Supporto multilingua (ITA/ENG)
- 💾 Architettura scalabile con modelli salvati e separazione backend/frontend
---
## 📸 Screenshot 
Ecco alcune schermate dell'applicazione:

![Screenshot 2025-05-12 150338](https://github.com/user-attachments/assets/709906cd-04e3-4cfe-8498-978dc2e4f63f)
---

## 📁 Struttura del progetto

# PredizioneDellObesita
```
PredizioneDellObesita/
│
├── DatiCsv/ # Dataset originale + previsioni salvate
│ ├── train.csv
│ ├── test.csv
│ └── predizioni_obesita.csv
│
├── SvModelli/ # Modelli e oggetti salvati (PKL/JSON)
│ ├── model_xgb.pkl
│ ├── scaler.pkl
│ ├── feature_names.pkl
│ ├── class_map.json
│ ├── Gender_encoder.pkl
│ ├── CAEC_encoder.pkl
│ ├── CALC_encoder.pkl
│ └── MTRANS_encoder.pkl
│
├── static/ # CSS e immagini statiche
│ ├── style.css
│ └── BackGround.css
│
├── templates/ # Template HTML (Jinja2)
│ ├── base.html
│ ├── home.html
│ ├── predict.html
│ ├── data.html
│ └── form.html
│
├── Img/ # Immagini per grafici o sfondi
│
├── app.py # Backend Flask
├── utils.py # Funzioni per la normalizzazione e encoding
├── prevObesita.py # Script di training e salvataggio modello
├── cleanerdata.ipynb # Notebook per analisi preliminare dei dati
└── README.md # Questo file
```

---

## 🧪 Modello e Dataset

- Modello: `XGBoostClassifier`
- Feature engineering: encoding categorico, scaler standard
- Target: classificazione multiclasse (7 classi da "Insufficient Weight" a "Obesity Type III")

---

## 🛠️ Setup rapido

```bash
git clone https://github.com/tuo-utente/PredizioneDellObesita.git
cd PredizioneDellObesita
pip install -r requirements.txt
python app.py
```

## 👨‍💻 Autori

- [Ciro Maresca](https://github.com/CyrusVII)

- [Filippo Giorgio Rondó](https://linktr.ee/filippogiorgiorondo)

## ⚠️ Disclaimer

- Questo progetto ha uno scopo puramente didattico e non sostituisce in alcun modo una valutazione medica professionale.

