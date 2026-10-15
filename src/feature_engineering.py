"""
Feature engineering for the resume-job matching dataset (slide 5 features).

Loads data/jobs.json + data/resumes.json (produced by generate_dataset.py)
and produces data/dataset.csv with one row per resume-job pair, holding the
7 model features plus a derived binary label (good_match).
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from skills import mentions, normalize_skills

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EDU_RANK = {"associate": 2, "bachelor": 3, "master": 4, "phd": 5}
EDU_LEVELS = [
    (r"phd|doctor", 5),
    (r"master", 4),
    (r"bachelor", 3),
    (r"associate", 2),
    (r"high school", 1),
]


def education_level(text: str) -> int:
    text = text.lower()
    for pattern, level in EDU_LEVELS:
        if re.search(pattern, text):
            return level
    return 3  # default to bachelor's-equivalent when unclear


def skill_overlap_ratio(candidate_skills: list, job_skills: list) -> float:
    """Share of the job's skills the candidate has, after skill normalization (slide 3)."""
    if not job_skills:
        return 0.0
    cand = normalize_skills(candidate_skills)
    job = normalize_skills(job_skills)
    return len(cand & job) / len(job)


def _field_in(field: str, options: list) -> bool:
    """Case-insensitive; also tolerates 'Computer Science & Engineering' vs 'Computer Science'."""
    f = field.strip().lower()
    return any(f == o.lower() or o.lower() in f for o in options)


def education_compatibility(resume: dict, job: dict) -> float:
    """Degree level vs. requirement, discounted when the field of study is unrelated."""
    cand_edu = EDU_RANK.get(resume.get("education_level")) or education_level(resume["education"])
    req_edu = EDU_RANK.get(job.get("education_level")) or education_level(job["education"])
    level_score = 1.0 if cand_edu >= req_edu else cand_edu / req_edu

    related, adjacent = job.get("related_fields"), job.get("adjacent_fields")
    field = resume.get("education_field")
    if related is None or field is None:
        return level_score
    if _field_in(field, related):
        factor = 1.0
    elif _field_in(field, adjacent or []):
        factor = 0.8
    else:
        factor = 0.55
    return level_score * factor


def project_relevance(projects: list, keywords: list) -> float:
    if not projects:
        return 0.0
    kw = {k.lower() for k in keywords}
    hits = 0
    for p in projects:
        text = p.get("title", "") + " " + p.get("description", "")
        if any(mentions(text, k) for k in kw):  # whole-token match: "java" != "javascript"
            hits += 1
    return hits / len(projects)


def compute_features(resume: dict, job: dict, semantic_similarity: float) -> dict:
    """The 7 slide-5 features for one resume against one job. Shared by training and the app."""
    return {
        "required_skill_match": skill_overlap_ratio(resume["skills"], job["required_skills"]),
        "preferred_skill_match": skill_overlap_ratio(resume["skills"], job["preferred_skills"]),
        "experience_match": (
            min(resume["experience_years"] / job["min_experience"], 1.5) if job["min_experience"] else 1.0
        ),
        "education_match": education_compatibility(resume, job),
        "technology_match": skill_overlap_ratio(resume["technologies"], job["technologies"]),
        "project_relevance": project_relevance(
            resume["projects"], job["required_skills"] + job["technologies"]
        ),
        "semantic_similarity": semantic_similarity,
    }


def build_dataset() -> pd.DataFrame:
    with open(DATA_DIR / "jobs.json", encoding="utf-8") as f:
        jobs = {j["id"]: j for j in json.load(f)}
    with open(DATA_DIR / "resumes.json", encoding="utf-8") as f:
        resumes = json.load(f)

    # Semantic similarity via TF-IDF over resume_text + job description_text
    job_ids = list(jobs.keys())
    job_texts = [jobs[jid]["description_text"] for jid in job_ids]
    resume_texts = [r["resume_text"] for r in resumes]
    vectorizer = TfidfVectorizer(stop_words="english")
    build_dataset.vectorizer = vectorizer  # saved by main() so the app scores similarity on the training scale
    all_vecs = vectorizer.fit_transform(job_texts + resume_texts)
    job_vecs = all_vecs[: len(job_texts)]
    resume_vecs = all_vecs[len(job_texts):]
    job_vec_by_id = {jid: job_vecs[i] for i, jid in enumerate(job_ids)}

    rows = []
    for i, r in enumerate(resumes):
        job = jobs[r["job_id"]]

        feats = compute_features(
            r, job, float(cosine_similarity(resume_vecs[i], job_vec_by_id[r["job_id"]])[0, 0])
        )
        required_skill_match = feats["required_skill_match"]
        preferred_skill_match = feats["preferred_skill_match"]
        experience_match = feats["experience_match"]
        education_match = feats["education_match"]
        technology_match = feats["technology_match"]
        proj_rel = feats["project_relevance"]
        semantic_similarity = feats["semantic_similarity"]

        score = (
            0.30 * required_skill_match
            + 0.10 * preferred_skill_match
            + 0.15 * min(experience_match, 1.0)
            + 0.10 * education_match
            + 0.15 * technology_match
            + 0.10 * proj_rel
            + 0.10 * semantic_similarity
        )
        if "label" in r:
            # Ground truth from the dataset author: a simulated recruiter decision made from the
            # candidate's true attributes (independent of the engineered features above).
            good_match = int(r["label"])
            score = r.get("recruiter_score", score)
        else:
            # Legacy path (API-generated resumes carry no label): noisy threshold on a weighted score.
            noise = np.random.normal(0, 0.12)
            good_match = int((score + noise) >= 0.55)
            if np.random.rand() < 0.06:
                good_match = 1 - good_match

        rows.append({
            "resume_id": r["id"],
            "job_id": r["job_id"],
            "tier": r["tier"],
            "required_skill_match": round(required_skill_match, 4),
            "preferred_skill_match": round(preferred_skill_match, 4),
            "experience_match": round(experience_match, 4),
            "education_match": round(education_match, 4),
            "technology_match": round(technology_match, 4),
            "project_relevance": round(proj_rel, 4),
            "semantic_similarity": round(semantic_similarity, 4),
            "match_score": round(score, 4),
            "good_match": good_match,
        })

    return pd.DataFrame(rows)


def main():
    np.random.seed(42)
    df = build_dataset()
    out_path = DATA_DIR / "dataset.csv"
    df.to_csv(out_path, index=False)
    (DATA_DIR.parent / "outputs").mkdir(exist_ok=True)
    joblib.dump(build_dataset.vectorizer, DATA_DIR.parent / "outputs" / "tfidf.pkl")
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df["good_match"].value_counts(normalize=True).rename("proportion"))


if __name__ == "__main__":
    main()
