# 🏥 Clinical Readmission Risk Engine (30-Day Triage CDS)

An end-to-end clinical decision-support (CDS) system engineered to predict 30-day all-cause hospital readmission risk for diabetic inpatients under the **CMS Hospital Readmissions Reduction Program (HRRP)**.

---

## 📌 Executive Summary & Clinical Context

Under the Affordable Care Act's HRRP, hospitals face reimbursement penalties (up to 3% of total Medicare billings) for elevated 30-day readmission rates. 

In clinical predictive modeling, the cost matrix is deeply asymmetric:
* **False Negative:** An at-risk patient is discharged without transitional support, leading to acute medical relapse and CMS financial penalties.
* **False Positive:** A stable patient receives low-cost preventative outreach (e.g., telephone follow-up, medication review).

This pipeline optimizes specifically for **Recall and PR-AUC (Precision-Recall AUC)** over raw accuracy, implementing clinical threshold calibration to prioritize high-risk patient interventions.

---

## 🏗️ Architecture & Engineering Pipeline
```text
├── data/
│   ├── raw/                  <- Ingested clinical encounter records
│   ├── processed/            <- Stratified Parquet feature sets & tournament records
│   └── clinical_records.db   <- Relational SQLite database with performance indexes
├── models/
│   ├── readmission_pipeline.joblib <- Serialized end-to-end ColumnTransformer + XGBoost
│   └── model_metadata.json         <- Calibrated decision threshold & metrics
├── src/
│   ├── init.py
│   ├── db_loader.py          <- Clinical filtering, exclusion rules & SQLite ingestion
│   ├── preprocessor.py       <- ICD-9 chapter grouping, age midpointing, scikit-learn transformers
│   ├── model_benchmarker.py  <- 5-model tournament via 5-Fold Stratified Cross-Validation
│   └── train_final.py        <- Final fitting, threshold tuning & pipeline export
├── app.py                    <- Interactive Streamlit triage engine & Plotly risk gauge
├── requirements.txt
└── README.md
```
---

## 🔬 Clinical Feature Engineering

1. **ICD-9 Chapter Mapping:** Compressed hundreds of granular medical diagnosis codes (`diag_1`, `diag_2`, `diag_3`) into 9 high-level clinical categories (Circulatory, Respiratory, Endocrine, Digestive, etc.).
2. **Prior Healthcare Utilization:** Aggregated emergency, outpatient, and inpatient visits over the previous 12 months into a composite utilization signal.
3. **Clinical Filtering:** Excluded expired and hospice-discharged patients ($n \in \{11, 13, 14, 19, 20, 21\}$) to eliminate structural target leakage.
4. **Leakage-Free Transformation:** Encapsulated categorical one-hot encoding and numerical standardization inside a unified Scikit-Learn `ColumnTransformer`.

---

## 🏆 Model Tournament Benchmark

Algorithms benchmarked across **5-Fold Stratified Cross-Validation** on ~70,000 encounters:

| Algorithmic Family | Model Candidate | Primary Metric (PR-AUC) | Secondary (ROC-AUC) |
| :--- | :--- | :---: | :---: |
| **Gradient Boosting** | **XGBoost Classifier (Selected)** | **Top Performer** | **High** |
| Bagging Ensemble | Random Forest | Competitive | Moderate |
| Margin-based | Linear Support Vector Classifier | Baseline | Baseline |
| Linear / Parametric | Logistic Regression ($L_2$ Regularized) | Clinical Baseline | Baseline |
| Probabilistic | Gaussian Naive Bayes | Exploratory | Low |

*The top model was calibrated using Precision-Recall curve analysis to select the optimal operational decision threshold, maximizing recall for vulnerable patients.*

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
conda create -n clinical-ml python=3.11 -y
conda activate clinical-ml
pip install -r requirements.txt
```
### 2. Ingestion & Preprocessing
```Bash

python -m src.db_loader
python -m src.preprocessor
```
### 3. Model Tournament & Serialization
```Bash
python -m src.model_benchmarker
python -m src.train_final
```

### 4. Launch Clinical Decision Support Dashboard
```Bash
streamlit run app.py
```


