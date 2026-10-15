# Explainable Resume-Job Matching Using SHAP

Implementation of the project described in `Explainable_Resume_Job_Matching_SHAP.pptx`:
a synthetic dataset (generated via the Claude API), feature engineering,
a Logistic Regression matcher, and SHAP-based explanations of each prediction.

## Quick start: the app

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Enter a role's requirements (or start from a template), choose applicants (the bundled
600-resume pool, 5 sample resumes in `sample_resumes/`, and/or your own PDF/DOCX/TXT uploads), and press
**Find best applicants**. You get a ranked list with a match score per candidate; click a row for matched and
missing skills, a plain-English explanation and the SHAP breakdown. The model files in `outputs/` must exist
first (see Run below).

## Pipeline

```
src/author_dataset.py      -> data/jobs.json, data/resumes.json   (offline dataset, no API key; default)
src/generate_dataset.py    -> data/jobs.json, data/resumes.json   (optional: Claude-API dataset creation)
src/skills.py              -> skill normalization helpers (aliases, whole-token matching)
src/resume_parser.py       -> PDF/DOCX/TXT resume parsing into the model's record format
src/matcher.py             -> ranking engine used by the app (features -> model -> SHAP)
app.py                     -> Streamlit app
src/feature_engineering.py -> data/dataset.csv                    (slide 5 features + label)
src/train_model.py         -> outputs/model.pkl, outputs/metrics.json
src/explain_shap.py        -> outputs/shap_*.png, outputs/shap_explanations.json
src/report_candidate.py    -> prints a slide-7-style "Example Prediction" report
```

## Setup

```powershell
pip install -r requirements.txt
# Only needed for src/generate_dataset.py (the offline author_dataset.py needs no key):
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

## Run

```powershell
python src/author_dataset.py --num-resumes 600   # offline, deterministic (--seed 42)
python src/feature_engineering.py
python src/train_model.py
python src/explain_shap.py
python src/report_candidate.py --list            # see all resume ids
python src/report_candidate.py --resume-id R0001 # slide-7-style report for one candidate
python src/report_candidate.py --random           # a random candidate
```

## What each stage does

1. **Dataset generation** (`author_dataset.py`, v2) — 10 job templates and 600
   resume-job pairs. Each resume has a *continuous* latent fit (not three fixed
   tiers), so signals overlap realistically across the quality range. Resumes
   include irrelevant extra skills, varied skill spellings ("ML", "k8s",
   "Amazon Web Services") that exercise skill normalization (slide 3), and a
   field of study. ~30% of resumes are paired with a job outside their own
   field (hard negatives). The `good_match` label is a simulated recruiter
   decision computed from the candidate's *true* attributes with a formula that
   differs from the model's features (absolute year-gaps, degree-gap penalties,
   a hard penalty for missing most required skills, subjectivity noise, 4%
   random flips), so the model approximates the decision rather than inverting
   a known formula. The old 100-row version is kept in `data/v1_backup/`.
   `generate_dataset.py` remains available for Claude-written resumes; those
   carry no label, so feature engineering falls back to a noisy-threshold label.

2. **Feature engineering** — computes the 7 features from slide 5
   (required/preferred skill match %, experience match, education match,
   technology match, project relevance, semantic similarity via TF-IDF +
   cosine similarity). Skills are normalized before overlap is computed,
   education combines degree level with field-of-study relevance, and project
   relevance uses whole-token matching. The label comes from the recruiter
   decision in `resumes.json`.

3. **Model training** — a Logistic Regression classifier (slide 5), with an
   80/20-ish stratified split, evaluated with accuracy/precision/recall/F1 and
   a confusion matrix (slide 8).

4. **SHAP explainability** — `shap.LinearExplainer` produces global feature
   importance (bar + beeswarm plots) and a local waterfall explanation for an
   individual prediction, plus a JSON file listing each test example's
   positive/negative contributing factors (slide 7).

5. **Resume parsing + ranking app** — `resume_parser.py` is rule-based (regex and vocabularies, no LLM):
   it extracts text, splits sections, detects skills (aliases normalized, plus any skill typed into the
   requirements), sums date ranges for years of experience, and detects degree level, field of study and
   projects. Anything it cannot find is returned as a warning and shown in the app. Scanned/image-only
   PDFs are not supported (no OCR). `matcher.py` computes the same 7 features for every applicant
   against the requirements entered and scores them with the trained model.

## Notes / limitations (carried over from slide 8)

- The model is linear, so experience is a weighted factor rather than a hard cut-off; the app offers an
  optional hard minimum-experience filter.

- The label is a *simulated* recruiter, so metrics measure how well the model
  recovers that simulated judgement, not real hiring outcomes.
- The dataset is fully synthetic; it approximates realistic resumes but
  isn't a substitute for real hiring data.
- Feature extraction here works off Claude's structured JSON output rather
  than parsing raw resume documents (PDF/DOCX parsing is future scope).
- SHAP explains the logistic regression's behavior, not whether the
  underlying prediction is substantively "correct."
