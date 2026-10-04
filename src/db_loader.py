# src/db_loader.py
import sqlite3
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV = BASE_DIR / "data" / "raw" / "diabetic_data.csv"
DB_PATH = BASE_DIR / "data" / "clinical_records.db"

# Discharge IDs indicating death or hospice (cannot be readmitted)
EXCLUDED_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}


def initialize_database():
    """Reads raw CSV, applies clinical validity filtering, and ingests into SQLite."""
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_CSV}")

    print("Reading raw clinical encounter data...")
    # Dataset represents missing values with '?'
    df = pd.read_csv(RAW_CSV, na_values="?", low_memory=False)
    initial_rows = len(df)
    print(f"Loaded {initial_rows:,} raw records.")

    # 1. Remove expired / hospice patients
    df = df[~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)].copy()
    print(f"Filtered out deceased/hospice cases: {len(df):,} records remaining.")

    # 2. Retain first clinical encounter per unique patient
    df = df.drop_duplicates(subset=["patient_nbr"], keep="first").copy()
    print(f"Retained first encounter per unique patient: {len(df):,} records remaining.")

    # 3. Create binary readmission label (<30 days vs rest)
    df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)
    pos_rate = df["readmitted_binary"].mean() * 100
    print(f"Target distribution: {pos_rate:.2f}% readmitted within 30 days (Class 1).")

    # 4. Ingest into SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print(f"Ingesting into SQLite database at {DB_PATH.name}...")
    df.to_sql("encounters", conn, if_exists="replace", index=False)

    # 5. Create performance indexes for querying
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_patient ON encounters(patient_nbr);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_target ON encounters(readmitted_binary);")
    conn.commit()
    conn.close()

    print("Done")


if __name__ == "__main__":
    initialize_database()  