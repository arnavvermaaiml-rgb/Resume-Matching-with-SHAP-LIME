"""
Train a Logistic Regression model on the engineered features (slide 5) and
report standard classification metrics (slide 8).
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

FEATURES = [
    "required_skill_match",
    "preferred_skill_match",
    "experience_match",
    "education_match",
    "technology_match",
    "project_relevance",
    "semantic_similarity",
]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_DIR / "dataset.csv")
    X = df[FEATURES]
    y = df["good_match"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cv = cross_validate(
        LogisticRegression(max_iter=1000), X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    )
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        # 5-fold CV over the full dataset: a steadier estimate than one hold-out split
        "cv_5fold": {
            name: {"mean": float(cv[f"test_{name}"].mean()), "std": float(cv[f"test_{name}"].std())}
            for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]
        },
        "coefficients": dict(zip(FEATURES, map(float, model.coef_[0]))),
    }

    print(json.dumps(metrics, indent=2))

    joblib.dump(model, OUT_DIR / "model.pkl")
    X_test.to_csv(OUT_DIR / "X_test.csv", index=False)
    y_test.to_csv(OUT_DIR / "y_test.csv", index=False)
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {OUT_DIR / 'model.pkl'}")


if __name__ == "__main__":
    main()
