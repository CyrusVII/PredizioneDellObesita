import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance,XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt


def pre_processing(train_path, test_path):
    """
    Funzione per il pre-processing del dataset sull'obesità.
    
    Args:
        train_path: percorso al file CSV di training
        test_path: percorso al file CSV di test
        
    Returns:
        X_train_selected: features di training processate
        y_train: target di training (convertito in interi)
        X_val: features di validation
        y_val: target di validation (convertito in interi)
        X_test_selected: features di test processate
        test_df: dataframe originale di test (per mantenere gli ID)
        weight_map: mappatura delle classi
    """
    # === 1. Caricamento dati ===
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # === 2. Pulizia dati ===
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    
    # === 3. Mappatura target ===
    weight_map = {
        'Normal_Weight': 1,
        'Insufficient_Weight': 0,
        'Overweight_Level_I': 2,
        'Overweight_Level_II': 3,
        'Obesity_Type_I': 4,
        'Obesity_Type_II': 5,
        'Obesity_Type_III': 6
    }
    
    train_df['NObeyesdad'] = train_df['NObeyesdad'].map(weight_map).astype(int)  # Forziamo interi
    
    # === 4. Codifica colonne booleane (yes/no -> True/False) ===
    for df in [train_df, test_df]:
        df.replace({'yes': True, 'no': False}, inplace=True)
        df.infer_objects(copy=False)
        
        # === 5. Creazione feature BMI ===
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)  # Altezza già in metri nel dataset
    
    # === 6. Codifica colonne categoriche ===
    categorical_cols = train_df.select_dtypes(include='object').columns
    label_encoders = {}
    
    # Codifica colonne presenti sia in train che test
    common_categorical_cols = [col for col in categorical_cols if col in test_df.columns]
    
    for col in common_categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        
        # Gestione valori sconosciuti in test
        test_df[col] = test_df[col].astype(str)
        test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else 'unknown')
        
        if 'unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'unknown')
        test_df[col] = le.transform(test_df[col])
        label_encoders[col] = le
    
    # === 7. Separazione features/target ===
    X = train_df.drop(columns=['id', 'NObeyesdad'])
    y = train_df['NObeyesdad'].astype(int)  # Garantiamo che sia intero
    
    # === 8. Split train/validation ===
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    
    # === 9. Normalizzazione ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Processamento test set
    X_test = test_df.drop(columns=['id', 'NObeyesdad'], errors='ignore')
    X_test_scaled = scaler.transform(X_test)
    
    # === 10. Preparazione output ===
    # Restituiamo i dati NON scalati per XGBoost (lavora meglio con dati non normalizzati)
    X_train_selected = X_train.copy()
    X_test_selected = X_test[X_train_selected.columns]  # Allineamento colonne
    
    # Verifica finale
    assert not y_train.isnull().any(), "Error: y_train contains NaN values"
    assert not y_val.isnull().any(), "Error: y_val contains NaN values"
    
    return X_train_selected, y_train, X_val, y_val, X_test_selected, test_df, weight_map

def modelling(X_train_selected, y_train, X_val, y_val, X_test_selected, test_df, weight_map, output_path=None):
    """
    Trains an XGBoost classifier using GridSearchCV, evaluates on validation set, and predicts on test set.
    """
    
    # Define hyperparameter grid for XGBoost
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    # Define XGBoost model with multi-class classification parameters
    xgb_model = XGBClassifier(
        objective='multi:softmax',  # For multi-class classification
        num_class=7,                # Number of target classes
        eval_metric='mlogloss',
        use_label_encoder=False,    # Prevent automatic label encoding
        random_state=42
    )

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1
    )

    grid_search.fit(X_train_selected, y_train)
    best_model = grid_search.best_estimator_

    print(f"\n🎯 Best hyperparameters found: {grid_search.best_params_}")

    # === Evaluate on validation set ===
    y_pred_xgb = best_model.predict(X_val)

    # No need for probability handling here, since `multi:softmax` will directly give predicted class labels
    accuracy = accuracy_score(y_val, y_pred_xgb)

    # Prepare readable class names
    inverse_map = {v: k for k, v in weight_map.items()}
    target_names = [inverse_map[i] for i in sorted(inverse_map.keys())]

    print("\n📄 Classification Report:\n")
    print(classification_report(y_val, y_pred_xgb, target_names=target_names))

    # === Predict on test set ===
    predictions = best_model.predict(X_test_selected)
    predictions = np.clip(predictions, 0, 6).astype(int)  # Ensure predictions are within the correct range

    pred_labels = [inverse_map.get(p, 'Unknown') for p in predictions]

    output = test_df[['id']].copy()
    output['Predicted_Obesity_Level'] = pred_labels

    if output_path:
        output.to_csv(output_path, index=False)
        print(f"\n💾 Predictions saved to: {output_path}")

    print("\n📊 Sample predictions (first 10 rows):")
    print(output.head(10))

    return output, best_model


# Example usage of functions
if __name__ == "__main__":
    # Define file paths
    train_path = "train.csv"
    test_path = "test.csv"
    output_path = "predizioni_obesita.csv"
    
    # Pre-processing
    X_train, y_train, X_val, y_val, X_test, test_df, weight_map = pre_processing(train_path, test_path)

    # Modellazione e predizione
    output, model = modelling(X_train, y_train, X_val, y_val, X_test, test_df, weight_map, output_path)
    
    print("Processo completato con successo!")
    