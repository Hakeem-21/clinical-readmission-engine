# src/train_final.py
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessor import (
    build_pipeline_transformer,
    load_and_transform_features,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_and_export_production_model():
    # 1. Load split sets
    print("Loading preprocessed training and test partitions...")
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["readmitted_binary"]
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["readmitted_binary"]

    _, _, numerical_cols, categorical_cols = load_and_transform_features()

    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Applying scale_pos_weight: {scale_weight:.2f}")

    # 2. Build full production pipeline
    transformer = build_pipeline_transformer(numerical_cols, categorical_cols)
    final_classifier = XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", transformer),
        ("classifier", final_classifier)
    ])

    print("\nFitting full production pipeline on complete training set...")
    pipeline.fit(X_train, y_train)

    # 3. Test set evaluation
    print("Evaluating against holdout test set...")
    y_test_probs = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_test_probs)
    pr_auc = average_precision_score(y_test, y_test_probs)

    print(f"--- Holdout Performance ---")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Test PR-AUC:  {pr_auc:.4f}")

    # 4. Optimal Clinical Decision Threshold Selection
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_probs)
    # Find threshold maximizing F1 score
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.50

    print(f"\nOptimal Clinical Decision Cutoff: {optimal_threshold:.4f}")
    print(f"At this cutoff -> Precision: {precisions[best_idx]:.4f} | Recall: {recalls[best_idx]:.4f} | F1: {f1_scores[best_idx]:.4f}")

    # 5. Confusion matrix under optimal threshold
    y_test_pred_custom = (y_test_probs >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_test_pred_custom)
    print("\nTest Confusion Matrix:")
    print(f"TN: {cm[0, 0]:,} | FP: {cm[0, 1]:,}")
    print(f"FN: {cm[1, 0]:,} | TP: {cm[1, 1]:,}")

    # 6. Export serialized pipeline and metadata
    pipeline_path = MODEL_DIR / "readmission_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    print(f"\n[Saved] End-to-end inference pipeline -> {pipeline_path}")

    metadata = {
        "model_type": "XGBClassifier",
        "optimal_threshold": round(optimal_threshold, 4),
        "test_roc_auc": round(float(roc_auc), 4),
        "test_pr_auc": round(float(pr_auc), 4),
        "test_recall": round(float(recalls[best_idx]), 4),
        "test_precision": round(float(precisions[best_idx]), 4),
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
    }

    metadata_path = MODEL_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[Saved] Metadata & thresholds -> {metadata_path}")
    print("\nready for deployment.")


if __name__ == "__main__":
    train_and_export_production_model()