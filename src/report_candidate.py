"""
Pretty-print a single candidate's prediction + SHAP explanation in the
"Example Prediction" format from slide 7: match probability, strong
positive factors, negative factors, and a plain-language local explanation.

Usage:
    python src/report_candidate.py --list
    python src/report_candidate.py --resume-id R0005
    python src/report_candidate.py --random
"""
import argparse
import json
import random
from pathlib import Path

import joblib
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

FEATURE_LABELS = {
    "required_skill_match": "Required skill coverage",
    "preferred_skill_match": "Preferred skill coverage",
    "experience_match": "Experience vs. requirement",
    "education_match": "Education compatibility",
    "technology_match": "Technology stack overlap",
    "project_relevance": "Relevant project experience",
    "semantic_similarity": "Resume-JD semantic similarity",
}

POSITIVE_PHRASES = {
    "required_skill_match": "strong coverage of the required skills",
    "preferred_skill_match": "good coverage of the preferred skills",
    "experience_match": "experience that meets or exceeds the requirement",
    "education_match": "an education background that matches the requirement",
    "technology_match": "solid overlap with the job's technology stack",
    "project_relevance": "past projects that are highly relevant to this role",
    "semantic_similarity": "a resume that closely aligns with the job description",
}

NEGATIVE_PHRASES = {
    "required_skill_match": "missing several required skills",
    "preferred_skill_match": "limited coverage of the preferred skills",
    "experience_match": "experience below the stated requirement",
    "education_match": "an education background that doesn't fully match the requirement",
    "technology_match": "limited overlap with the job's technology stack",
    "project_relevance": "projects with limited relevance to this role",
    "semantic_similarity": "a resume that doesn't closely align with the job description text",
}


def load_context():
    df = pd.read_csv(DATA_DIR / "dataset.csv")
    with open(DATA_DIR / "jobs.json", encoding="utf-8") as f:
        jobs = {j["id"]: j for j in json.load(f)}
    with open(DATA_DIR / "resumes.json", encoding="utf-8") as f:
        resumes = {r["id"]: r for r in json.load(f)}
    model = joblib.load(OUT_DIR / "model.pkl")
    return df, jobs, resumes, model


def narrative(name, title, prob, pos_factors, neg_factors):
    """Plain-language explanation. Drivers are the factors pushing in the direction of the prediction."""
    strong = prob >= 0.5
    label = "a strong match" if strong else "a weak match"
    sentence = f"{name} is predicted to be {label} for the {title} role ({prob*100:.0f}% match probability)"
    drivers, drivers_phrases = (pos_factors, POSITIVE_PHRASES) if strong else (neg_factors, NEGATIVE_PHRASES)
    offsets, offsets_phrases = (neg_factors, NEGATIVE_PHRASES) if strong else (pos_factors, POSITIVE_PHRASES)
    if drivers:
        sentence += ", mainly because of " + " and ".join(drivers_phrases[f] for f in drivers[:2])
    sentence += "."
    if offsets:
        verb = "held back somewhat by" if strong else "partly offset by"
        sentence += f" This is {verb} {offsets_phrases[offsets[0]]}."
    return sentence


def print_report(resume_id, df, jobs, resumes, model):
    if resume_id not in set(df["resume_id"]):
        raise SystemExit(f"Unknown resume_id '{resume_id}'. Use --list to see available ids.")

    row = df[df["resume_id"] == resume_id].iloc[0]
    job = jobs[row["job_id"]]
    resume = resumes[resume_id]

    X_full = df[FEATURES]
    explainer = shap.LinearExplainer(model, shap.maskers.Independent(X_full, max_samples=len(X_full)))
    x_row = X_full[df["resume_id"] == resume_id]
    sv = explainer(x_row)

    prob = model.predict_proba(x_row)[0, 1]
    predicted = "Good Match" if prob >= 0.5 else "Poor Match"
    actual = "Good Match" if row["good_match"] == 1 else "Poor Match"

    contribs = sorted(
        zip(FEATURES, sv.values[0]), key=lambda kv: kv[1], reverse=True
    )
    pos = [(f, v) for f, v in contribs if v > 0]
    neg = sorted([(f, v) for f, v in contribs if v < 0], key=lambda kv: kv[1])

    w = 72
    print("=" * w)
    print(f" CANDIDATE: {resume['name']:<30} ROLE: {job['title']} ({job['id']})")
    print("=" * w)
    print(f" MATCH PROBABILITY: {prob*100:5.1f}%")
    print(f" PREDICTED: {predicted:<12}  ACTUAL LABEL: {actual:<12}  TIER: {resume['tier']}")
    print("-" * w)
    print(" STRONG POSITIVE FACTORS")
    if pos:
        for f, v in pos:
            print(f"   + {FEATURE_LABELS[f]:<32} contribution: +{v:.3f}")
    else:
        print("   (none)")
    print()
    print(" NEGATIVE FACTORS")
    if neg:
        for f, v in neg:
            print(f"   - {FEATURE_LABELS[f]:<32} contribution: {v:.3f}")
    else:
        print("   (none)")
    print("-" * w)
    print(" LOCAL EXPLANATION")
    print(" " + narrative(
        resume["name"], job["title"], prob,
        [f for f, _ in pos], [f for f, _ in neg],
    ))
    print("=" * w)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume-id", type=str, help="e.g. R0005")
    group.add_argument("--random", action="store_true", help="pick a random candidate")
    group.add_argument("--list", action="store_true", help="list available resume ids")
    args = parser.parse_args()

    df, jobs, resumes, model = load_context()

    if args.list:
        for _, row in df.iterrows():
            print(f"{row['resume_id']}  job={row['job_id']}  tier={row['tier']:<6}  label={'Good' if row['good_match'] else 'Poor'}")
        return

    resume_id = args.resume_id
    if args.random or not resume_id:
        resume_id = random.choice(df["resume_id"].tolist())

    print_report(resume_id, df, jobs, resumes, model)


if __name__ == "__main__":
    main()
