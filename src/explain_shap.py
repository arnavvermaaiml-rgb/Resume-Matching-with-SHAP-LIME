"""
SHAP explainability for the trained Logistic Regression model (slides 6-7).

Produces:
  - outputs/shap_summary_global.png  : global feature importance (bar)
  - outputs/shap_beeswarm_global.png : global SHAP distribution
  - outputs/shap_local_example.png   : local waterfall for one test case
  - outputs/shap_explanations.json   : per-feature contributions for every
                                        test-set example (strong/negative
                                        factors per candidate, slide 7)
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

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
    model = joblib.load(OUT_DIR / "model.pkl")
    X_test = pd.read_csv(OUT_DIR / "X_test.csv")

    explainer = shap.LinearExplainer(model, shap.maskers.Independent(X_test, max_samples=len(X_test)))
    shap_values = explainer(X_test)

    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_global.png", dpi=150)
    plt.close()

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_beeswarm_global.png", dpi=150)
    plt.close()

    example_idx = 0
    plt.figure()
    shap.plots.waterfall(shap_values[example_idx], show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_local_example.png", dpi=150)
    plt.close()

    explanations = []
    proba = model.predict_proba(X_test)[:, 1]
    for i in range(len(X_test)):
        contribs = {
            FEATURES[j]: round(float(shap_values.values[i, j]), 4)
            for j in range(len(FEATURES))
        }
        sorted_contribs = sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)
        explanations.append({
            "row_index": int(i),
            "match_probability": round(float(proba[i]), 4),
            "positive_factors": [k for k, v in sorted_contribs if v > 0],
            "negative_factors": [k for k, v in sorted_contribs if v < 0],
            "feature_contributions": contribs,
        })

    with open(OUT_DIR / "shap_explanations.json", "w") as f:
        json.dump(explanations, f, indent=2)

    print(f"Saved SHAP plots and explanations to {OUT_DIR}")
    print(f"\nExample explanation (row 0):")
    print(json.dumps(explanations[0], indent=2))


if __name__ == "__main__":
    main()
