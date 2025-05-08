import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === 1. Caricamento dei dati ===
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# === 2. Pulizia valori nulli ===
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# === 3. Mappatura della colonna target 'NObeyesdad' ===
weight_map = {
    'Normal_Weight': 0,
    'Insufficient_Weight': -1,
    'Overweight_Level_I': 1,
    'Overweight_Level_II': 2,
    'Obesity_Type_I': 3,
    'Obesity_Type_II': 4,
    'Obesity_Type_III': 5
}
train_df['NObeyesdad'] = train_df['NObeyesdad'].map(weight_map)

# === 4. Encoding yes/no → True/False ===
for df in [train_df, test_df]:
    df.replace({'yes': True, 'no': False}, inplace=True)

# === 5. Label Encoding delle colonne categoriche (esclusa la colonna target) ===
categorical_cols = train_df.select_dtypes(include='object').columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    
    # Fit su dati combinati per evitare problemi con categorie nuove
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(combined)
    
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    
    label_encoders[col] = le

# === 6. Separazione input e target ===
X = train_df.drop(columns=['id', 'NObeyesdad'])  # Rimuove id e target
y = train_df['NObeyesdad']

# === 7. Train/validation split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
