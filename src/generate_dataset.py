"""
Generate a synthetic resume-job dataset using the Anthropic Claude API.

Addresses the "synthetic training data" limitation noted in the project
(slide 8) by producing realistic, structured resumes against a fixed set of
job postings, with controlled match quality so the resulting dataset has a
meaningful spread of good/poor matches for the downstream Logistic
Regression + SHAP pipeline.

Usage:
    set ANTHROPIC_API_KEY=sk-ant-...
    python src/generate_dataset.py --num-pairs 200
"""
import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from anthropic import Anthropic
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

JOB_TEMPLATES = [
    {
        "id": "J01", "title": "Data Scientist",
        "required_skills": ["python", "machine learning", "sql", "statistics", "pandas"],
        "preferred_skills": ["deep learning", "nlp", "tableau"],
        "min_experience": 3, "education": "Bachelor's in Computer Science, Statistics, or related field",
        "technologies": ["python", "scikit-learn", "sql", "spark"],
    },
    {
        "id": "J02", "title": "Backend Engineer",
        "required_skills": ["java", "spring boot", "rest apis", "sql", "microservices"],
        "preferred_skills": ["kafka", "docker", "kubernetes"],
        "min_experience": 2, "education": "Bachelor's in Computer Science or related field",
        "technologies": ["java", "spring", "postgresql", "docker"],
    },
    {
        "id": "J03", "title": "Frontend Developer",
        "required_skills": ["javascript", "react", "html", "css", "typescript"],
        "preferred_skills": ["next.js", "redux", "testing"],
        "min_experience": 2, "education": "Bachelor's in Computer Science or related field",
        "technologies": ["react", "typescript", "webpack"],
    },
    {
        "id": "J04", "title": "DevOps Engineer",
        "required_skills": ["linux", "ci/cd", "docker", "kubernetes", "terraform"],
        "preferred_skills": ["aws", "monitoring", "ansible"],
        "min_experience": 3, "education": "Bachelor's in Computer Science or related field",
        "technologies": ["docker", "kubernetes", "terraform", "aws"],
    },
    {
        "id": "J05", "title": "Machine Learning Engineer",
        "required_skills": ["python", "deep learning", "pytorch", "mlops", "machine learning"],
        "preferred_skills": ["kubernetes", "nlp", "computer vision"],
        "min_experience": 3, "education": "Master's in Computer Science, AI, or related field",
        "technologies": ["python", "pytorch", "docker", "kubernetes"],
    },
    {
        "id": "J06", "title": "Product Manager",
        "required_skills": ["product strategy", "roadmapping", "stakeholder management", "agile", "analytics"],
        "preferred_skills": ["sql", "a/b testing", "ux research"],
        "min_experience": 4, "education": "Bachelor's degree in any field; MBA preferred",
        "technologies": ["jira", "sql", "figma"],
    },
    {
        "id": "J07", "title": "Cloud Solutions Architect",
        "required_skills": ["aws", "cloud architecture", "networking", "security", "terraform"],
        "preferred_skills": ["azure", "gcp", "cost optimization"],
        "min_experience": 5, "education": "Bachelor's in Computer Science or related field",
        "technologies": ["aws", "terraform", "kubernetes"],
    },
    {
        "id": "J08", "title": "QA Automation Engineer",
        "required_skills": ["selenium", "test automation", "python", "api testing", "ci/cd"],
        "preferred_skills": ["performance testing", "cypress", "java"],
        "min_experience": 2, "education": "Bachelor's in Computer Science or related field",
        "technologies": ["selenium", "pytest", "jenkins"],
    },
    {
        "id": "J09", "title": "Data Engineer",
        "required_skills": ["python", "sql", "etl", "spark", "data pipelines"],
        "preferred_skills": ["airflow", "kafka", "aws"],
        "min_experience": 3, "education": "Bachelor's in Computer Science or related field",
        "technologies": ["spark", "airflow", "sql", "aws"],
    },
    {
        "id": "J10", "title": "UX Designer",
        "required_skills": ["figma", "user research", "wireframing", "prototyping", "ux design"],
        "preferred_skills": ["design systems", "usability testing", "html/css"],
        "min_experience": 2, "education": "Bachelor's in Design, HCI, or related field",
        "technologies": ["figma", "adobe xd"],
    },
]

TIERS = [
    ("strong", 0.4),
    ("medium", 0.35),
    ("weak", 0.25),
]

CLIENT = Anthropic()


def _extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(match.group(0))


def generate_job_description(job: dict) -> str:
    prompt = f"""Write a realistic, concise job description (120-180 words) for the role
"{job['title']}". It must mention these required skills: {', '.join(job['required_skills'])}.
It should also mention these preferred/nice-to-have skills: {', '.join(job['preferred_skills'])}.
State the minimum experience as {job['min_experience']}+ years and the education requirement:
"{job['education']}". Return plain text only, no markdown, no headers."""
    resp = CLIENT.messages.create(
        model=MODEL, max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


TIER_INSTRUCTIONS = {
    "strong": (
        "This candidate should be a STRONG match: they should possess 80-100% of the "
        "required skills, meet or exceed the minimum experience, hold the required "
        "education level, list most of the job's technologies, and have 2-3 projects "
        "clearly relevant to this role."
    ),
    "medium": (
        "This candidate should be a MEDIUM match: they should possess roughly 40-65% of "
        "the required skills, have experience close to but somewhat below the minimum "
        "(e.g. 1 year short), have a related but not exact education background, overlap "
        "with only some of the job's technologies, and have 1-2 projects of partial relevance."
    ),
    "weak": (
        "This candidate should be a WEAK match: they should possess fewer than 30% of the "
        "required skills, have noticeably less experience than the minimum requirement, "
        "an unrelated education background, little to no overlap with the job's "
        "technologies, and projects unrelated to this role."
    ),
}


def generate_resume(job: dict, tier: str, idx: int) -> dict:
    prompt = f"""You are generating a SYNTHETIC resume for a machine learning training dataset.
Target job: "{job['title']}".
Job required skills: {', '.join(job['required_skills'])}.
Job preferred skills: {', '.join(job['preferred_skills'])}.
Job technologies: {', '.join(job['technologies'])}.
Job minimum experience: {job['min_experience']} years.
Job education requirement: {job['education']}.

{TIER_INSTRUCTIONS[tier]}

Return ONLY a JSON object with this exact shape (no markdown fences, no commentary):
{{
  "name": "<realistic fake full name>",
  "skills": ["<skill1>", "<skill2>", ...],
  "experience_years": <number>,
  "education": "<degree and field, e.g. Bachelor's in Computer Science>",
  "technologies": ["<tech1>", "<tech2>", ...],
  "projects": [{{"title": "<project title>", "description": "<1 sentence description>"}}, ...],
  "resume_text": "<a realistic 150-220 word resume summary paragraph combining the above>"
}}
Lowercase all entries in "skills" and "technologies" for consistency."""
    resp = CLIENT.messages.create(
        model=MODEL, max_tokens=900,
        messages=[{"role": "user", "content": prompt}],
    )
    data = _extract_json(resp.content[0].text)
    data["id"] = f"R{idx:04d}"
    data["job_id"] = job["id"]
    data["tier"] = tier
    return data


def pick_tier() -> str:
    r = random.random()
    cum = 0.0
    for tier, weight in TIERS:
        cum += weight
        if r <= cum:
            return tier
    return TIERS[-1][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Set it in your environment before running:\n"
            "  setx ANTHROPIC_API_KEY \"sk-ant-...\"   (Windows, new shells)\n"
            "  $env:ANTHROPIC_API_KEY=\"sk-ant-...\"   (current PowerShell session)"
        )

    random.seed(args.seed)
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Generating {len(JOB_TEMPLATES)} job descriptions...")
    jobs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(generate_job_description, job): job for job in JOB_TEMPLATES}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            job = futures[fut]
            job_out = dict(job)
            job_out["description_text"] = fut.result()
            jobs.append(job_out)
    jobs.sort(key=lambda j: j["id"])
    with open(DATA_DIR / "jobs.json", "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)

    jobs_by_id = {j["id"]: j for j in jobs}
    assignments = [
        (random.choice(JOB_TEMPLATES), pick_tier(), i)
        for i in range(args.num_pairs)
    ]

    print(f"Generating {args.num_pairs} synthetic resumes...")
    resumes = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_resume, jobs_by_id[job["id"]], tier, idx): idx
            for job, tier, idx in assignments
        }
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                resumes.append(fut.result())
            except Exception as e:
                print(f"  skipped one resume due to error: {e}")
    resumes.sort(key=lambda r: r["id"])
    with open(DATA_DIR / "resumes.json", "w", encoding="utf-8") as f:
        json.dump(resumes, f, indent=2)

    print(f"Done. Wrote {len(jobs)} jobs and {len(resumes)} resumes to {DATA_DIR}")


if __name__ == "__main__":
    main()
