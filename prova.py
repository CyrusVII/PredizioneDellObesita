import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# 1. Carica i dati
df = pd.read_csv('train.csv')

# 2. Separazione X e y
X = df.drop(columns=['NObeyesdad'])
y = df['NObeyesdad']

# 3. Codifica variabili categoriche
# Applica LabelEncoder a tutte le colonne object e bool
label_encoders = {}
for col in X.select_dtypes(include=['object', 'bool']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Codifica anche la variabile target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# 4. Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modello
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Valutazione
y_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred, target_names=target_encoder.classes_))
