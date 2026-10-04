# src/model_benchmarker.py
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.preprocessor import (
    build_pipeline_transformer,
    load_and_transform_features,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_train_data():
    """Loads stratified training sets saved from Milestone 2."""
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")[
        "readmitted_binary"
    ]
    return X_train, y_train


def get_candidate_models(scale_pos_weight: float):
    """Instantiates 5 distinct algorithm candidates with class weighting."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Linear SVM (Calibrated)": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=42, dual="auto")
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }


def run_tournament():
    X_train, y_train = load_train_data()
    _, _, numerical_cols, categorical_cols = load_and_transform_features()

    # Calculate class imbalance weighting factor
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_weight = negative_count / positive_count

    print(
        f"Starting Model Tournament on {len(X_train):,} training records..."
    )
    print(
        f"Class balance: {negative_count:,} negative / {positive_count:,} positive (Ratio: {scale_weight:.2f}:1)\n"
    )

    models = get_candidate_models(scale_pos_weight=scale_weight)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    for name, model in models.items():
        print(f"--> Benchmarking: {name} across 5 folds...")

        fold_roc_aucs = []
        fold_pr_aucs = []
        fold_f1s = []

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X_train, y_train), 1
        ):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Build full end-to-end pipeline inside fold to avoid data leakage
            transformer = build_pipeline_transformer(
                numerical_cols, categorical_cols
            )
            pipeline = Pipeline(
                [("preprocessor", transformer), ("classifier", model)]
            )

            pipeline.fit(X_tr, y_tr)

            # Predict probabilities for positive class (readmission)
            y_probs = pipeline.predict_proba(X_val)[:, 1]
            y_preds = (y_probs >= 0.50).astype(int)

            fold_roc_aucs.append(roc_auc_score(y_val, y_probs))
            fold_pr_aucs.append(average_precision_score(y_val, y_probs))
            fold_f1s.append(f1_score(y_val, y_preds, zero_division=0))

        mean_roc = np.mean(fold_roc_aucs)
        mean_pr = np.mean(fold_pr_aucs)
        mean_f1 = np.mean(fold_f1s)

        results.append(
            {
                "Model": name,
                "Mean ROC-AUC": round(mean_roc, 4),
                "Mean PR-AUC": round(mean_pr, 4),
                "Mean F1": round(mean_f1, 4),
            }
        )
        print(
            f"    [Result] {name} => PR-AUC: {mean_pr:.4f} | ROC-AUC: {mean_roc:.4f} | F1: {mean_f1:.4f}\n"
        )

    # Rank leaderboard by PR-AUC (primary metric for imbalanced clinical readmission)
    leaderboard = pd.DataFrame(results).sort_values(
        by="Mean PR-AUC", ascending=False
    )
    leaderboard_path = PROCESSED_DIR / "tournament_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    print("=" * 60)
    print("           FINAL MODEL TOURNAMENT LEADERBOARD           ")
    print("=" * 60)
    print(leaderboard.to_string(index=False))
    print(f"\nLeaderboard successfully saved to: {leaderboard_path}")


if __name__ == "__main__":
    run_tournament()