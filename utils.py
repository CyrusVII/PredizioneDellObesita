import joblib
import numpy as np
import pandas as pd

# Ricarica LabelEncoders usati per l'encoding (puoi anche salvarli e ricaricarli da file)
label_encoders = {}

# === Convertitore utente → dati modello ===
def transform_user_input(form_data, scaler, feature_names):
    # Costruisci DataFrame a partire dai dati del form
    input_dict = {key: [form_data[key]] for key in feature_names}
    df = pd.DataFrame(input_dict)
    
    # Normalize form values (remove capitalization inconsistencies)
    df = df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)
    
    # Capitalizzazione coerente con il training
    capitalized_cols = ['Gender', 'CAEC', 'CALC', 'MTRANS']
    for col in capitalized_cols:
      df[col] = df[col].apply(lambda x: x if x in ['yes', 'no'] else x.capitalize())
    
    # Conversione speciale per MTRANS (es. public_transportation → Public_Transportation)
    if 'MTRANS' in df.columns:
      df['MTRANS'] = df['MTRANS'].apply(lambda x: x.replace('_', ' ').title().replace(' ', '_'))

    # Eccezioni particolari
    df['MTRANS'] = df['MTRANS'].apply(lambda x: x.replace('_', ' ').title().replace(' ', '_'))
    
    # === Booleani ===
    for col in ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC','CALC']:
        df[col] = df[col].map({'yes': True, 'no': False})

    # === Label encoding per categorici ===
    label_cols = ['Gender', 'CAEC', 'CALC', 'MTRANS']
    for col in label_cols:
        if col not in label_encoders:
            le = joblib.load(f'SvModelli/{col}_encoder.pkl')  # li devi salvare tu nel pre-processing
            label_encoders[col] = le
        df[col] = label_encoders[col].transform(df[col].astype(str))

    # === Ordina le colonne ===
    df = df[feature_names]

    # === Scala ===
    scaled = scaler.transform(df)

    return scaled


