import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render
from .forms import InputForm

# Carica il modello, lo scaler, l'encoder e i nomi delle colonne
model = joblib.load('C:/Users/Ciro/Desktop/CorsoPython/pythonProgettiGruppo/PredizioneDellObesita/JangoData/xgb_obesity_model.pkl')
scaler = joblib.load('C:/Users/Ciro/Desktop/CorsoPython/pythonProgettiGruppo/PredizioneDellObesita/JangoData/scaler.pkl')
feature_names = joblib.load('C:/Users/Ciro/Desktop/CorsoPython/pythonProgettiGruppo/PredizioneDellObesita/JangoData/features.pkl')  # lista delle colonne
inverse_map = joblib.load('C:/Users/Ciro/Desktop/CorsoPython/pythonProgettiGruppo/PredizioneDellObesita/JangoData/inverse_map.pkl')
caec_encoder = joblib.load('C:/Users/Ciro/Desktop/CorsoPython/pythonProgettiGruppo/PredizioneDellObesita/JangoData/CAEC_encoder.pkl')
calc_encoder = joblib.load('C:/Users/Ciro/Desktop/CorsoPython/pythonProgettiGruppo/PredizioneDellObesita/JangoData/CALC_encoder.pkl')
mtrans_encoder = joblib.load('C:/Users/Ciro/Desktop/CorsoPython/pythonProgettiGruppo/PredizioneDellObesita/JangoData/MTRANS_encoder.pkl')


def map_values(user_input, col_name, mapping):
    user_input[col_name] = user_input[col_name].map(mapping)
    return user_input

def predict(request):
    prediction = None
    # Mappatura per CAEC e calc
    caec_calc_map = {
        0: 'No',
        1: 'Sometimes',
        2: 'Frequently',
        3: 'Always'
    }

    # Mappatura per MTRANS
    mtrans_map = {
        0: 'Automobile',
        1: 'Motorbike',
        2: 'Bike',
        3: 'Public_Transportation',
        4: 'Walking'
    }

    if request.method == 'POST':
        form = InputForm(request.POST)
        if form.is_valid():
            # Ottieni i dati in formato dizionario
            cleaned_data = form.cleaned_data

            # Crea un DataFrame da una riga
            user_input = pd.DataFrame([cleaned_data])

            # Assicura che tutte le colonne usate nel training siano presenti
            for col in feature_names:
                if col not in user_input.columns:
                    user_input[col] = 0  # Imposta 0 per i valori mancanti

            # Ordina le colonne nella giusta sequenza
            user_input = user_input[feature_names]

            # Mappiamo i dati dell'utente
            user_input = map_values(user_input, 'CAEC', caec_calc_map)
            user_input = map_values(user_input, 'CALC', caec_calc_map)
            user_input = map_values(user_input, 'MTRANS', mtrans_map)
            #encodiamo 
            user_input['CAEC'] = caec_encoder.transform(user_input['CAEC'])
            user_input['CALC'] = calc_encoder.transform(user_input['CALC'])
            user_input['MTRANS'] = mtrans_encoder.transform(user_input['MTRANS'])


            # 2. Applica lo scaling ai dati numerici
            X_scaled = scaler.transform(user_input)

            # 3. Predizione
            pred = model.predict(X_scaled)
            pred = np.clip(pred, 0, 6).astype(int)
            prediction = inverse_map.get(int(pred[0]), "Etichetta sconosciuta")
            
    else:
        form = InputForm()

    return render(request, 'predizione/index.html', {'form': form, 'prediction': prediction})



