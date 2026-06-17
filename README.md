# 🏥 Cardiovascular Risk Classification Pipeline

An end-to-end clinical machine learning pipeline designed to predict cardiovascular risk profiles using patient diagnostics from the OpenML `heart-h` dataset. This project showcases structured data cleaning, robust imputation strategies, and unified preprocessing using Scikit-Learn pipelines.

---

## 🏗️ System Architecture & Data Flow

To ensure high data integrity and completely prevent data leakage between evaluation sets, data is funneled through a strict operational sequence:

1. **Initial Data Ingestion:** Collects 294 raw clinical patient records.
2. **Missing Value Imputation:**
   - **Continuous Features:** Missing blocks are cleanly imputed using data column medians.
   - **Categorical Features:** Missing data entries are filled using column modes.
3. **Stratified Split:** Data is divided into an 80% Training set (235 records) and a 20% Test validation set (59 records).
4. **Encapsulated Preprocessing Pipeline (`ColumnTransformer`):**
   - **Numerical Path:** Automatically standardizes continuous columns via `StandardScaler`.
   - **Categorical Path:** Transforms nominal text parameters into binary tracks via `OneHotEncoder`.
5. **Model Estimation:** Feeds the resulting mathematical matrix directly into an instantiated `RandomForestClassifier`.

---

## 📊 Model Performance Metrics

By encapsulating data scaling, string transformations, and model training within a single pipeline workflow, the architecture safely resolved initial text conversion blocks and achieved a strong baseline tracking profile.

### 📈 Final Validation Accuracy: **86.44%**

### 📋 Classification Report
| Target Risk Profile | Precision | Recall | F1-Score | Patients (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Low Risk Profile)** | 0.89 | 0.89 | 0.89 | 38 |
| **1 (High Risk Profile)** | 0.81 | 0.81 | 0.81 | 21 |
| **System Accuracy** | | | **0.86** | **59** |
| **Macro Average** | 0.85 | 0.85 | 0.85 | 59 |
| **Weighted Average** | 0.86 | 0.86 | 0.86 | 59 |

### 📉 Confusion Matrix Breakdown
- **True Negatives (TN):** 34 patients accurately predicted as low risk.
- **True Positives (TP):** 17 patients accurately flagged as high risk.
- **False Negatives (FN):** 4 high-risk patients missed during testing.
- **False Positives (FP):** 4 low-risk patients flagged with warning metrics.

---

## 🏆 Top Clinical Feature Importances

Leveraging Scikit-Learn's internal step-tracking API (`pipeline.named_steps[]`), we extracted the internal Gini weights to rank the key diagnostic indicators driving risk determinations:

1. **`oldpeak` (15.76%)** — ST depression induced by exercise relative to rest.
2. **`chol` (11.61%)** — Serum cholesterol levels.
3. **`thalach` (11.29%)** — Maximum heart rate achieved during stress tracking.
4. **`chest_pain_asympt` (10.29%)** — Silent, asymptomatic clinical pain presentations.

---

## 📁 Repository Structure Map

- `.gitignore` — Scope configurations excluding local environments (`ml_env/`).
- `README.md` — Project summary, performance matrix, and engineering architectural map.
- `CHANGELOG.md` — Chronological engineering journal documenting structural fixes and pivots.
- `cardio_risk_pipeline.ipynb` — Executable Jupyter Notebook holding baseline and pipeline nodes.