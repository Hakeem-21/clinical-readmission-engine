# src/preprocessor.py
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "clinical_records.db"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def map_icd9_to_category(val):
    """Maps raw ICD-9 codes to high-level clinical disease chapters."""
    if pd.isna(val) or val == "?" or str(val).strip() == "":
        return "Missing"

    val_str = str(val).strip()

    # External causes / Supplementary classifications
    if val_str.startswith(("V", "E")):
        return "Other"

    try:
        code = float(val_str)
    except ValueError:
        return "Other"

    if (390 <= code <= 459) or code == 785:
        return "Circulatory"
    elif (460 <= code <= 519) or code == 786:
        return "Respiratory"
    elif (520 <= code <= 579) or code == 787:
        return "Digestive"
    elif np.floor(code) == 250:
        return "Diabetes"
    elif 800 <= code <= 999:
        return "Injury"
    elif 710 <= code <= 739:
        return "Musculoskeletal"
    elif (580 <= code <= 629) or code == 788:
        return "Genitourinary"
    elif 140 <= code <= 239:
        return "Neoplasms"
    else:
        return "Other"


def clean_age(age_str):
    """Converts bracketed age ranges like '[70-80)' into the range midpoint."""
    if pd.isna(age_str):
        return 55.0
    mapping = {
        "[0-10)": 5.0,
        "[10-20)": 15.0,
        "[20-30)": 25.0,
        "[30-40)": 35.0,
        "[40-50)": 45.0,
        "[50-60)": 55.0,
        "[60-70)": 65.0,
        "[70-80)": 75.0,
        "[80-90)": 85.0,
        "[90-100)": 95.0,
    }
    return mapping.get(str(age_str).strip(), 55.0)


def load_and_transform_features():
    """Extracts raw table from SQLite and applies clinical transformations."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM encounters", conn)
    conn.close()

    print(f"Loaded {len(df):,} records from database for feature pipeline.")

    # 1. Feature Engineering
    df["age_num"] = df["age"].apply(clean_age)
    df["diag_1_group"] = df["diag_1"].apply(map_icd9_to_category)
    df["diag_2_group"] = df["diag_2"].apply(map_icd9_to_category)
    df["diag_3_group"] = df["diag_3"].apply(map_icd9_to_category)

    # 2. Total prior healthcare utilization count
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )

    # 3. Define feature columns
    numerical_cols = [
        "age_num",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "total_prior_visits",
    ]

    categorical_cols = [
        "race",
        "gender",
        "admission_type_id",
        "admission_source_id",
        "max_glu_serum",
        "A1Cresult",
        "metformin",
        "glipizide",
        "glyburide",
        "pioglitazone",
        "rosiglitazone",
        "insulin",
        "change",
        "diabetesMed",
        "diag_1_group",
        "diag_2_group",
        "diag_3_group",
    ]

    for col in categorical_cols:
        df[col] = df[col].fillna("Missing").astype(str)

    target_col = "readmitted_binary"

    X = df[numerical_cols + categorical_cols]
    y = df[target_col]

    return X, y, numerical_cols, categorical_cols


def build_pipeline_transformer(numerical_cols, categorical_cols):
    """Creates a reusable scikit-learn ColumnTransformer."""
    numeric_pipe = StandardScaler()
    categorical_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numerical_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    return preprocessor


def prepare_and_save_data():
    """Generates train/test split and saves processed artifacts."""
    X, y, num_cols, cat_cols = load_and_transform_features()

    print("Splitting dataset (80% Train, 20% Test) with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    # Save to disk for model training
    X_train.to_parquet(PROCESSED_DIR / "X_train.parquet", index=False)
    X_test.to_parquet(PROCESSED_DIR / "X_test.parquet", index=False)
    y_train.to_frame().to_parquet(PROCESSED_DIR / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(PROCESSED_DIR / "y_test.parquet", index=False)

    print(f"Done")


if __name__ == "__main__":
    prepare_and_save_data()