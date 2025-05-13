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
import joblib
import json


def pre_processing(train_path, test_path):
    """
    Performs preprocessing for the obesity dataset.
    
    Args:
        train_path: Path to training CSV file
        test_path: Path to test CSV file
        
    Returns (in this exact order):
        X_train_scaled: Scaled training features (StandardScaler)
        y_train: Training target (converted to integers)
        X_val_scaled: Scaled validation features  
        y_val: Validation target (converted to integers)
        X_test_scaled: Scaled test features
        test_df: Original test dataframe (to preserve IDs)
        weight_map: Class label mapping dictionary
    """
    # === 1. Data Loading ===
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # === 2. Null Value Handling ===
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    
    # === 3. Target Encoding ===
    weight_map = {
        'Normal_Weight': 1,
        'Insufficient_Weight': 0, 
        'Overweight_Level_I': 2,
        'Overweight_Level_II': 3,
        'Obesity_Type_I': 4,
        'Obesity_Type_II': 5,
        'Obesity_Type_III': 6
    }
    # Force integer conversion for classification
    train_df['NObeyesdad'] = train_df['NObeyesdad'].map(weight_map).astype(int)  

    # === 4. Boolean Encoding (yes/no → True/False) ===
    for df in [train_df, test_df]:
        df.replace({'yes': True, 'no': False}, inplace=True)
        df.infer_objects(copy=False)  # Auto-type conversion

    # === 5. Categorical Feature Encoding ===
    categorical_cols = train_df.select_dtypes(include='object').columns
    
    for col in categorical_cols:
        if col in test_df.columns:  # Only encode columns present in both datasets
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col].astype(str))
            
            # Handle unseen test categories (e.g., 'unknown')
            test_df[col] = test_df[col].astype(str)
            mask = ~test_df[col].isin(le.classes_)
            test_df.loc[mask, col] = 'unknown'
            
            if 'unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'unknown')
            test_df[col] = le.transform(test_df[col])
            
            # Salva encoder
            joblib.dump(le, f'C:/Users/Ciro/Desktop/CorsoPython/pythonProgettiGruppo/PredizioneDellObesita/SvModelli/{col}_encoder.pkl')

    # === 6. Feature/Target Separation ===
    X = train_df.drop(columns=['id', 'NObeyesdad'])
    y = train_df['NObeyesdad'].astype(int)  # Ensure integer type
    feature_names = X.columns.tolist()
    # === 7. Train/Validation Split ===
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )

    # === 8. Feature Scaling (StandardScaler) ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training data
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(test_df.drop(columns=['id', 'NObeyesdad'], errors='ignore'))
    joblib.dump(scaler, 'C:/Users/Ciro/Desktop/CorsoPython/pythonProgettiGruppo/PredizioneDellObesita/SvModelli/scaler.pkl')
    
    # === 9. Test Feature Alignment ===
    # Ensure test data has same columns as training data after encoding
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)

    # === 10. Final Checks ===
    assert not y_train.isnull().any(), "Error: y_train contains NaN values"
    assert not y_val.isnull().any(), "Error: y_val contains NaN values"
    
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, test_df, weight_map,feature_names

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
    train_path = "DatiCsv/train.csv"
    test_path = "DatiCsv/test.csv"
    output_path = "DatiCsv/predizioni_obesita.csv"
    
    # Pre-processing
    X_train, y_train, X_val, y_val, X_test, test_df, weight_map,feature_names = pre_processing(train_path, test_path)

    # Modellazione e predizione
    output, model = modelling(X_train, y_train, X_val, y_val, X_test, test_df, weight_map, output_path)
    
    # Salvataggio
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit con dati di training
    
    joblib.dump(model, 'SvModelli/model_xgb.pkl')
    joblib.dump(feature_names, 'SvModelli/feature_names.pkl')
    with open('SvModelli/class_map.json', 'w') as f:
        json.dump({str(k): v for k, v in weight_map.items()}, f)
        
    print("Process succesfully completed!")