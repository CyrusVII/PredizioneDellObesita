# 📚 Project Documentation

## 1. General Description

The application predicts a user's obesity level based on manually entered data. The core of the system is an XGBoost model trained on real-world data. The frontend is built using Flask, Bootstrap, and Jinja2.

---

## 2. Dataset

- Source: Public dataset on obesity  
- Features: 17 variables (numerical and categorical)  
- Target: `NObeyesdad` (7 classes)

---

## 3. Pre-processing Pipeline

- Conversion of "yes"/"no" strings → boolean values  
- Encoding of categorical variables using `LabelEncoder`  
- Numerical scaling using `StandardScaler`  
- Train-test-validation split with stratification  
- Object saving (`model.pkl`, `scaler.pkl`, `encoder.pkl`)

---

## 4. Model

- Type: `XGBoostClassifier`  
- Evaluation: Accuracy, Confusion Matrix, Cross-validation  
- Feature importance visualized through graph

---

## 5. Web App (Flask)

- `/`: home page with introduction  
- `/predict`: prediction form  
- `/data`: section with charts and insights  
- Language toggle stored in `localStorage`  
- Responsive layout (Bootstrap)

---

## 6. Charts

- Distribution of target classes  
- Feature importance  
- Accuracy comparison: Train vs Validation

---

## 7. Requirements

- Python ≥ 3.8  
- Flask, XGBoost, pandas, scikit-learn

---

## 8. Credits

Project developed by:  
- **Maresca Ciro**  
- **Filippo Giorgio Rondó**
