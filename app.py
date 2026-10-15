"""
Explainable Resume-Job Matching: Streamlit app.

Describe the role on the left, add applicants (bundled pool, sample resumes,
or your own PDF / DOCX / TXT uploads), and get a ranked list with a match
score for each candidate and a SHAP explanation of why.

Run:  streamlit run app.py
"""
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matcher as m  # noqa: E402
from resume_parser import KNOWN_FIELDS, extract_text  # noqa: E402

GREEN, RED = "#2e9e5b", "#d1495b"
EDU_OPTIONS = ["associate", "bachelor", "master", "phd"]

st.set_page_config(page_title="Resume Matcher", page_icon=":mag:", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return m.Artifacts()


def read_sample_resumes():
    files = sorted((ROOT / "sample_resumes").glob("*"))
    return [{"filename": f.name, "text": extract_text(f.read_bytes(), f.name)} for f in files if f.is_file()]


def chips(items, color):
    return " ".join(f":{color}-badge[{s}]" for s in items) if items else "none"


def shap_chart(contributions: dict):
    items = sorted(contributions.items(), key=lambda kv: kv[1])
    labels = [m.FEATURE_LABELS[k] for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.barh(labels, values, color=[GREEN if v > 0 else RED for v in values])
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_xlabel("Contribution to match score (SHAP, log-odds)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


try:
    art = get_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.title("Explainable Resume-Job Matching")
st.caption("Describe the role, add applicants, and see who fits best, and exactly why.")

# --------------------------------------------------------------------------
# Sidebar: requirements + applicants
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Job requirements")
    template_names = ["Custom (blank)"] + [f"{j['title']} (template)" for j in art.jobs]
    choice = st.selectbox("Start from a template", template_names)
    tpl = None if choice == template_names[0] else art.jobs[template_names.index(choice) - 1]
    k = choice  # widget keys include the template so switching templates refreshes the defaults

    with st.form("job_form"):
        title = st.text_input("Job title", tpl["title"] if tpl else "", key=f"title_{k}")
        required = st.text_area(
            "Required skills (comma-separated)", ", ".join(tpl["required_skills"]) if tpl else "",
            key=f"req_{k}", height=80,
        )
        preferred = st.text_area(
            "Preferred skills (nice to have)", ", ".join(tpl["preferred_skills"]) if tpl else "",
            key=f"pref_{k}", height=68,
        )
        technologies = st.text_input(
            "Technologies / tools (optional)", ", ".join(tpl["technologies"]) if tpl else "", key=f"tech_{k}",
            help="Leave blank to use the required skills.",
        )
        min_exp = st.number_input(
            "Minimum experience (years)", 0.0, 30.0, float(tpl["min_experience"]) if tpl else 2.0, 0.5, key=f"exp_{k}",
        )
        edu_default = EDU_OPTIONS.index(tpl["education_level"]) if tpl else 1
        edu_level = st.selectbox(
            "Minimum education", EDU_OPTIONS, index=edu_default, format_func=lambda x: m.EDU_LABEL[x], key=f"edu_{k}",
        )
        field_default = [f for f in (tpl or {}).get("related_fields") or [] if f in KNOWN_FIELDS]
        fields = st.multiselect(
            "Relevant fields of study (optional)", KNOWN_FIELDS, default=field_default, key=f"fields_{k}",
            help="Leave empty to compare degree level only.",
        )
        jd_text = st.text_area(
            "Job description text (optional)", "", key=f"jd_{k}", height=90,
            help="Paste a full JD for better semantic matching. Otherwise one is written from the fields above.",
        )

        st.header("2. Applicants")
        use_pool = st.checkbox("Bundled synthetic pool (600 resumes)", value=True)
        use_samples = st.checkbox("Sample resumes (5 PDF/DOCX/TXT files)", value=True)
        uploaded = st.file_uploader(
            "Upload resumes (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Find best applicants", type="primary", width="stretch")

# --------------------------------------------------------------------------
# Scoring (on submit)
# --------------------------------------------------------------------------
if submitted:
    req_terms = m.split_terms(required)
    if not req_terms:
        st.sidebar.error("Add at least one required skill.")
    else:
        uploads, upload_errors = [], []
        if use_samples:
            uploads += read_sample_resumes()
        for f in uploaded or []:
            try:
                uploads.append({"filename": f.name, "text": extract_text(f.getvalue(), f.name)})
            except Exception as e:  # corrupt/encrypted file
                upload_errors.append(f"{f.name}: {e}")
        job = m.make_job(
            title, req_terms, m.split_terms(preferred), m.split_terms(technologies), min_exp, edu_level, fields, jd_text,
        )
        candidates = m.prepare_candidates(job, art.pool if use_pool else [], uploads)
        if not candidates:
            st.sidebar.error("No applicants selected. Enable the pool or add resumes.")
        else:
            st.session_state["result"] = {"job": job, "df": m.score_candidates(art, job, candidates), "errors": upload_errors}

tab_rank, tab_model, tab_how = st.tabs(["Ranking", "Model insights", "How it works"])

# --------------------------------------------------------------------------
# Ranking tab
# --------------------------------------------------------------------------
with tab_rank:
    result = st.session_state.get("result")
    if not result:
        st.info("Fill in the requirements on the left (or pick a template) and press **Find best applicants**.")
    else:
        job, df = result["job"], result["df"]
        for err in result["errors"]:
            st.warning(f"Could not read {err}")

        st.subheader(f"Ranking for: {job['title']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Applicants scored", len(df))
        c2.metric("Predicted good matches", int((df["score"] >= 0.5).sum()))
        c3.metric("Top match score", f"{df['score'].iloc[0] * 100:.1f}%")
        c4.metric("Uploaded resumes", int((df["source"] == "Uploaded").sum()))

        f1, f2, f3 = st.columns([1, 1, 2])
        top_n = f1.slider("Show top", 5, 100, 15, step=5)
        sources = f2.multiselect("Source", ["Pool", "Uploaded"], default=["Pool", "Uploaded"])
        hard_exp = f3.checkbox(
            f"Hard requirement: at least {job['min_experience']:g} years of experience",
            help="The model treats experience as one weighted factor. Tick this to drop anyone below the minimum outright.",
        )

        view = df[df["source"].isin(sources)]
        if hard_exp:
            view = view[view["experience_years"] >= job["min_experience"]]
        view = view.head(top_n).reset_index(drop=True)

        if view.empty:
            st.warning("No applicants match the current filters.")
        else:
            table = pd.DataFrame({
                "Rank": view["rank"],
                "Name": view["name"],
                "Source": view["source"],
                "Match score": view["score"] * 100,
                "Verdict": view["verdict"],
                "Required skills": view["required_skill_match"] * 100,
                "Experience (yrs)": view["experience_years"],
                "Education": view["education"],
            })
            event = st.dataframe(
                table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                column_config={
                    "Match score": st.column_config.ProgressColumn("Match score", format="%.1f%%", min_value=0, max_value=100),
                    "Required skills": st.column_config.ProgressColumn("Required skills", format="%.0f%%", min_value=0, max_value=100),
                    "Experience (yrs)": st.column_config.NumberColumn(format="%.1f"),
                },
            )
            st.caption("Click a row to see the explanation. Scores are the model's match probability; ranks are across all scored applicants.")

            export = df.assign(
                missing_required=df["required_missing"].map(", ".join),
                matched_required=df["required_matched"].map(", ".join),
            )[["rank", "name", "source", "score", "verdict", "experience_years", "education",
               "matched_required", "missing_required"] + m.FEATURES]
            st.download_button(
                "Download full ranking (CSV)", export.to_csv(index=False).encode("utf-8"),
                file_name="ranking.csv", mime="text/csv",
            )

            selected = event.selection.rows[0] if event.selection.rows else 0
            cand = view.iloc[selected]

            st.divider()
            st.subheader(f"{cand['name']}  ·  rank #{cand['rank']}")
            left, right = st.columns([3, 2])
            with left:
                a, b = st.columns(2)
                a.metric("Match score", f"{cand['score'] * 100:.1f}%")
                b.metric("Verdict", cand["verdict"])
                st.write(cand["narrative"])
                st.markdown("**Required skills matched:** " + chips(cand["required_matched"], "green"))
                st.markdown("**Required skills missing:** " + chips(cand["required_missing"], "red"))
                st.markdown("**Preferred skills matched:** " + chips(cand["preferred_matched"], "green"))
                st.markdown("**Preferred skills missing:** " + chips(cand["preferred_missing"], "orange"))
                st.markdown(
                    f"**Experience:** {cand['experience_years']:g} yrs (needs {job['min_experience']:g})  \n"
                    f"**Education:** {cand['education']}"
                    + (f"  \n**Email:** {cand['email']}" if cand["email"] else "")
                )
                for w in cand["warnings"]:
                    st.warning(w)
                if cand["projects"]:
                    with st.expander(f"Projects ({len(cand['projects'])})"):
                        for p in cand["projects"]:
                            st.markdown(f"**{p['title']}**: {p['description']}")
            with right:
                st.markdown("**Why this score**")
                st.pyplot(shap_chart(cand["contributions"]), clear_figure=True)
                st.caption("Green bars push the score up, red bars pull it down, relative to an average applicant.")

# --------------------------------------------------------------------------
# Model insights tab
# --------------------------------------------------------------------------
with tab_model:
    st.subheader("Model performance")
    mt = art.metrics
    if mt:
        cols = st.columns(5)
        for col, (label, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                                            ("F1", "f1_score"), ("ROC-AUC", "roc_auc")]):
            if key in mt:
                col.metric(label, f"{mt[key]:.3f}")
        if "cv_5fold" in mt:
            st.caption(
                f"Hold-out test set: {mt.get('n_test', '?')} pairs. 5-fold cross-validated accuracy: "
                f"{mt['cv_5fold']['accuracy']['mean']:.3f} ± {mt['cv_5fold']['accuracy']['std']:.3f}."
            )
        cm = mt.get("confusion_matrix")
        if cm:
            st.markdown("**Confusion matrix** (rows: actual, columns: predicted)")
            st.dataframe(pd.DataFrame(cm, index=["Actual poor", "Actual good"], columns=["Pred poor", "Pred good"]))
    st.subheader("Global feature importance (SHAP)")
    g1, g2 = st.columns(2)
    for col, name, cap in [(g1, "shap_summary_global.png", "Mean absolute SHAP value per feature"),
                           (g2, "shap_beeswarm_global.png", "Distribution of each feature's effect")]:
        path = ROOT / "outputs" / name
        if path.exists():
            col.image(str(path), caption=cap)
    st.info(
        "The labels used to train this model come from a **simulated recruiter** on synthetic resumes, so these "
        "numbers show how well the model recovers that judgement, not real hiring outcomes."
    )

# --------------------------------------------------------------------------
# How it works tab
# --------------------------------------------------------------------------
with tab_how:
    st.markdown(
        """
**Pipeline** (slide 4): resume + job description → text extraction → information extraction → skill
normalization → feature engineering → Logistic Regression → match probability → SHAP explanation.

**The 7 features** (slide 5), each computed for every applicant against *your* requirements:
required-skill match, preferred-skill match, experience vs. requirement, education (degree level and field),
technology overlap, project relevance, and TF-IDF semantic similarity between the resume and the job text.

**Resume parsing** is rule-based: it reads PDF/DOCX/TXT, splits sections (Skills, Experience, Education,
Projects), finds skills from a vocabulary plus anything you typed as a requirement (aliases like *k8s*, *ML*
are normalized), sums date ranges for years of experience, and detects degree level and field. Anything it could
not find is flagged under the candidate. Scanned/image-only PDFs are not supported (no OCR).

**Limitations**
- Trained on synthetic data with simulated recruiter labels; treat scores as decision support, not decisions.
- The model is linear, so experience is a weighted factor, not a hard cut-off. Use the filter above if it is one.
- SHAP explains the *model's* behaviour; it does not prove the prediction is correct or fair.
        """
    )
