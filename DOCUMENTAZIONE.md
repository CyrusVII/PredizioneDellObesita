
---

## ✅ 3. Documentazione del progetto

```md
# 📚 Documentazione del progetto

## 1. Descrizione generale

L'applicazione consente di prevedere il livello di obesità di un utente partendo da dati inseriti manualmente. Il cuore del sistema è un modello XGBoost addestrato su dati reali. Il frontend è stato realizzato con Flask, Bootstrap e Jinja2.

---

## 2. Dataset

- Fonte: Dataset pubblico sull’obesità
- Features: 17 variabili (numeriche e categoriche)
- Target: `NObeyesdad` (7 classi)

---

## 3. Pipeline di Pre-processing

- Conversione stringhe "yes"/"no" → booleani
- Encoding con `LabelEncoder` per variabili categoriche
- Scaling numerico con `StandardScaler`
- Train-test-validation split con stratificazione
- Salvataggio oggetti (`model.pkl`, `scaler.pkl`, `encoder.pkl`)

---

## 4. Modello

- Tipo: `XGBoostClassifier`
- Valutazione: Accuracy, Confusion Matrix, Cross-validation
- Feature importance visualizzata con grafico

---

## 5. Web App (Flask)

- `/`: home page con presentazione
- `/predict`: form per la predizione
- `/data`: sezione con grafici e insight
- Toggle lingua salvato in `localStorage`
- Layout responsive (Bootstrap)

---

## 6. Grafici

- Distribuzione delle classi target
- Feature importance
- Confronto accuracy Train vs Validation

---

## 7. Requisiti

- Python ≥ 3.8
- Flask, XGBoost, pandas, scikit-learn

---

## 8. Credits

Progetto realizzato da:
- **Maresca Ciro**
- **Filippo Giorgio Rondó**
