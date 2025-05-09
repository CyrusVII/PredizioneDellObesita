# 🧠 Obesity Predictor Web App

Benvenuti nel progetto **Predizione dell'Obesità**, realizzato da **Maresca Ciro** e **Filippo Giorgio Rondó**!  
Questa web app consente di stimare il livello di obesità di un individuo sulla base di dati personali e abitudini, utilizzando un modello XGBoost addestrato su un dataset reale.

---

## 🚀 Funzionalità

- 🧾 Inserimento dati utente tramite form interattivo
- 🤖 Predizione automatica del livello di obesità
- 📊 Sezione grafici con analisi del dataset e modello
- 🌐 Supporto multilingua (ITA/ENG)
- 💾 Architettura scalabile con modelli salvati e separazione backend/frontend

---

## 📁 Struttura del progetto

# PredizioneDellObesita
'''
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
'''

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

- Filippo Giorgio Rondó

## ⚠️ Disclaimer

- Questo progetto ha uno scopo puramente didattico e non sostituisce in alcun modo una valutazione medica professionale.

