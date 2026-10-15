"""
Author the synthetic resume-job dataset offline (no Anthropic API calls).

v2 design (replaces the 100-row, tier-templated v1):
  * Each resume has a continuous latent "fit" q (Beta-distributed), not a
    discrete strong/medium/weak tier, so skill/experience/education/project
    signals overlap across the quality range instead of being constant per tier.
  * Candidates list extra, irrelevant skills and write skills in varied forms
    ("ML", "k8s", "Amazon Web Services") so skill normalization (slide 3) is
    actually exercised.
  * ~30% of resumes are paired with a job outside their own field, giving
    realistic hard negatives and partial-overlap cases.
  * Education has a field-of-study dimension, not just degree level.
  * The ground-truth label is a simulated recruiter decision computed from the
    candidate's true attributes with a formula that differs from the model's
    engineered features, plus subjectivity noise and random flips.

Usage:
    python src/author_dataset.py [--num-resumes 600] [--cross-role-frac 0.3] [--seed 42]
"""
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from skills import ALIASES, mentions, normalize_skills

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

JOBS = [
    {
        "id": "J01", "title": "Data Scientist",
        "required_skills": ["python", "machine learning", "sql", "statistics", "pandas"],
        "preferred_skills": ["deep learning", "nlp", "tableau"],
        "min_experience": 3, "education_level": "bachelor", "education_field": "Computer Science",
        "education": "Bachelor's in Computer Science, Statistics, or related field",
        "technologies": ["python", "scikit-learn", "sql", "spark"],
        "description_text": (
            "We are seeking a Data Scientist to join our analytics team. The ideal candidate "
            "will have strong experience in python, machine learning, sql, statistics, and "
            "pandas to build predictive models and extract actionable insights from large "
            "datasets. Experience with deep learning, nlp, and tableau is a plus. "
            "Responsibilities include designing experiments, building and validating models, "
            "and communicating results to stakeholders. Requires a minimum of 3 years of "
            "relevant experience and a Bachelor's degree in Computer Science, Statistics, or "
            "a related field. The role involves close collaboration with engineering and "
            "product teams to deploy models into production pipelines built on spark and "
            "scikit-learn."
        ),
    },
    {
        "id": "J02", "title": "Backend Engineer",
        "required_skills": ["java", "spring boot", "rest apis", "sql", "microservices"],
        "preferred_skills": ["kafka", "docker", "kubernetes"],
        "min_experience": 2, "education_level": "bachelor", "education_field": "Computer Science",
        "education": "Bachelor's in Computer Science or related field",
        "technologies": ["java", "spring", "postgresql", "docker"],
        "description_text": (
            "We're hiring a Backend Engineer to design and maintain scalable services. You'll "
            "work daily with java, spring boot, rest apis, sql, and microservices architecture "
            "to power our core platform. Familiarity with kafka, docker, and kubernetes is a "
            "strong plus. You will own service reliability, participate in on-call rotations, "
            "and collaborate with frontend and product teams on API design. Requires at least "
            "2 years of backend development experience and a Bachelor's degree in Computer "
            "Science or a related field. Our stack runs on java/spring with postgresql and "
            "docker-based deployments."
        ),
    },
    {
        "id": "J03", "title": "Frontend Developer",
        "required_skills": ["javascript", "react", "html", "css", "typescript"],
        "preferred_skills": ["next.js", "redux", "testing"],
        "min_experience": 2, "education_level": "bachelor", "education_field": "Computer Science",
        "education": "Bachelor's in Computer Science or related field",
        "technologies": ["react", "typescript", "webpack"],
        "description_text": (
            "Join our product team as a Frontend Developer building delightful, performant "
            "user interfaces. Core requirements include javascript, react, html, css, and "
            "typescript. Experience with next.js, redux, and automated testing is preferred. "
            "You'll work closely with designers and backend engineers to ship features across "
            "web and mobile breakpoints, maintain our component library, and uphold "
            "accessibility standards. Requires at least 2 years of frontend experience and a "
            "Bachelor's degree in Computer Science or a related field. Our stack is built on "
            "react, typescript, and webpack."
        ),
    },
    {
        "id": "J04", "title": "DevOps Engineer",
        "required_skills": ["linux", "ci/cd", "docker", "kubernetes", "terraform"],
        "preferred_skills": ["aws", "monitoring", "ansible"],
        "min_experience": 3, "education_level": "bachelor", "education_field": "Computer Science",
        "education": "Bachelor's in Computer Science or related field",
        "technologies": ["docker", "kubernetes", "terraform", "aws"],
        "description_text": (
            "We are looking for a DevOps Engineer to own our deployment and infrastructure "
            "automation. Required skills include linux administration, ci/cd pipeline design, "
            "docker, kubernetes, and terraform. Experience with aws, monitoring tooling, and "
            "ansible is preferred. You'll be responsible for infrastructure-as-code, release "
            "automation, and incident response. Requires at least 3 years of relevant "
            "experience and a Bachelor's degree in Computer Science or a related field. Our "
            "platform runs on docker and kubernetes, provisioned via terraform on aws."
        ),
    },
    {
        "id": "J05", "title": "Machine Learning Engineer",
        "required_skills": ["python", "deep learning", "pytorch", "mlops", "machine learning"],
        "preferred_skills": ["kubernetes", "nlp", "computer vision"],
        "min_experience": 3, "education_level": "master", "education_field": "Computer Science / AI",
        "education": "Master's in Computer Science, AI, or related field",
        "technologies": ["python", "pytorch", "docker", "kubernetes"],
        "description_text": (
            "We are hiring a Machine Learning Engineer to take models from research to "
            "production. Required: python, deep learning, pytorch, mlops practices, and "
            "general machine learning fundamentals. Experience with kubernetes, nlp, or "
            "computer vision is a plus. You will build training and serving pipelines, "
            "monitor model performance in production, and collaborate with research "
            "scientists. Requires at least 3 years of relevant experience and a Master's "
            "degree in Computer Science, AI, or a related field. Our serving stack is built "
            "on pytorch, docker, and kubernetes."
        ),
    },
    {
        "id": "J06", "title": "Product Manager",
        "required_skills": ["product strategy", "roadmapping", "stakeholder management", "agile", "analytics"],
        "preferred_skills": ["sql", "a/b testing", "ux research"],
        "min_experience": 4, "education_level": "bachelor", "education_field": "any field",
        "education": "Bachelor's degree in any field; MBA preferred",
        "technologies": ["jira", "sql", "figma"],
        "description_text": (
            "We're looking for a Product Manager to drive product strategy and execution. "
            "You should be skilled in roadmapping, stakeholder management, agile delivery, "
            "and using analytics to guide decisions. Experience with sql, a/b testing, and ux "
            "research is a plus. You'll partner with engineering, design, and sales to define "
            "and ship high-impact features. Requires at least 4 years of product management "
            "experience and a Bachelor's degree in any field; an MBA is preferred. Our team "
            "runs on jira, sql-based reporting, and figma for design collaboration."
        ),
    },
    {
        "id": "J07", "title": "Cloud Solutions Architect",
        "required_skills": ["aws", "cloud architecture", "networking", "security", "terraform"],
        "preferred_skills": ["azure", "gcp", "cost optimization"],
        "min_experience": 5, "education_level": "bachelor", "education_field": "Computer Science",
        "education": "Bachelor's in Computer Science or related field",
        "technologies": ["aws", "terraform", "kubernetes"],
        "description_text": (
            "We need a Cloud Solutions Architect to design secure, scalable cloud "
            "infrastructure. Required: deep aws expertise, cloud architecture design, "
            "networking, security best practices, and terraform. Experience with azure, gcp, "
            "or cost optimization initiatives is a plus. You'll lead architecture reviews, "
            "define infrastructure standards, and mentor engineering teams. Requires at least "
            "5 years of relevant experience and a Bachelor's degree in Computer Science or a "
            "related field. Our infrastructure is provisioned with terraform across aws and "
            "kubernetes."
        ),
    },
    {
        "id": "J08", "title": "QA Automation Engineer",
        "required_skills": ["selenium", "test automation", "python", "api testing", "ci/cd"],
        "preferred_skills": ["performance testing", "cypress", "java"],
        "min_experience": 2, "education_level": "bachelor", "education_field": "Computer Science",
        "education": "Bachelor's in Computer Science or related field",
        "technologies": ["selenium", "pytest", "jenkins"],
        "description_text": (
            "We are hiring a QA Automation Engineer to build and maintain our automated test "
            "suites. Required skills: selenium, test automation, python, api testing, and "
            "ci/cd integration. Experience with performance testing, cypress, or java is a "
            "plus. You'll design regression suites, integrate tests into our pipelines, and "
            "triage failures with engineering. Requires at least 2 years of relevant "
            "experience and a Bachelor's degree in Computer Science or a related field. Our "
            "testing stack uses selenium, pytest, and jenkins."
        ),
    },
    {
        "id": "J09", "title": "Data Engineer",
        "required_skills": ["python", "sql", "etl", "spark", "data pipelines"],
        "preferred_skills": ["airflow", "kafka", "aws"],
        "min_experience": 3, "education_level": "bachelor", "education_field": "Computer Science",
        "education": "Bachelor's in Computer Science or related field",
        "technologies": ["spark", "airflow", "sql", "aws"],
        "description_text": (
            "We're looking for a Data Engineer to build reliable data pipelines at scale. "
            "Required: python, sql, etl development, spark, and general data pipeline design. "
            "Experience with airflow, kafka, or aws is a plus. You'll own data ingestion, "
            "transformation, and quality monitoring for our analytics and ML platforms. "
            "Requires at least 3 years of relevant experience and a Bachelor's degree in "
            "Computer Science or a related field. Our pipelines run on spark and airflow, "
            "deployed on aws."
        ),
    },
    {
        "id": "J10", "title": "UX Designer",
        "required_skills": ["figma", "user research", "wireframing", "prototyping", "ux design"],
        "preferred_skills": ["design systems", "usability testing", "html/css"],
        "min_experience": 2, "education_level": "bachelor", "education_field": "Design / HCI",
        "education": "Bachelor's in Design, HCI, or related field",
        "technologies": ["figma", "adobe xd"],
        "description_text": (
            "We are hiring a UX Designer to shape intuitive, delightful product experiences. "
            "Required skills: figma, user research, wireframing, prototyping, and core ux "
            "design practice. Experience with design systems, usability testing, or html/css "
            "is a plus. You'll run research studies, produce wireframes and prototypes, and "
            "partner closely with product and engineering. Requires at least 2 years of "
            "relevant experience and a Bachelor's degree in Design, HCI, or a related field. "
            "We design primarily in figma and adobe xd."
        ),
    },
]

RELEVANT_PROJECTS = {
    "J01": [
        ("Customer Churn Prediction", "Built a machine learning model in python using scikit-learn and pandas to predict customer churn, improving retention targeting by 18%."),
        ("Sales Forecasting Pipeline", "Developed a statistics-based sales forecasting pipeline using sql and python, feeding a spark-based batch job."),
        ("A/B Test Analysis Framework", "Created a reusable statistics framework in python for analyzing A/B test results across product teams."),
    ],
    "J02": [
        ("Order Management Microservice", "Designed and implemented a java/spring boot microservice exposing rest apis for order management, backed by sql."),
        ("Payment Service Migration", "Migrated a monolithic payment module into independent microservices, containerized with docker."),
        ("Internal API Gateway", "Built an internal rest apis gateway in java to route requests across backend microservices."),
    ],
    "J03": [
        ("E-commerce Storefront Revamp", "Rebuilt a storefront using react and typescript, improving page load speed and accessibility."),
        ("Design System Component Library", "Created a reusable html/css component library adopted across multiple react applications."),
        ("Marketing Site Rebuild", "Rebuilt the company marketing site using next.js and typescript with a focus on seo."),
    ],
    "J04": [
        ("CI/CD Pipeline Overhaul", "Redesigned the ci/cd pipeline using docker and kubernetes, cutting deployment time by 60%."),
        ("Infrastructure as Code Migration", "Migrated manual infrastructure provisioning to terraform across multiple aws accounts."),
        ("Kubernetes Cluster Hardening", "Hardened production kubernetes clusters running on linux, improving security posture."),
    ],
    "J05": [
        ("Image Classification Service", "Trained and deployed a deep learning model in pytorch for image classification, served via a kubernetes-based mlops pipeline."),
        ("MLOps Pipeline Automation", "Built an mlops pipeline automating retraining and deployment for several machine learning models."),
        ("NLP Sentiment Pipeline", "Developed an nlp sentiment analysis pipeline in python integrated into a production pytorch serving stack."),
    ],
    "J06": [
        ("Mobile App Roadmap Relaunch", "Led roadmapping and stakeholder management to relaunch a mobile app, applying agile ceremonies across three squads."),
        ("Pricing Strategy Analysis", "Used sql-driven analytics to inform a new product strategy for a subscription pricing tier."),
        ("Cross-team OKR Rollout", "Drove stakeholder management and analytics reporting for a company-wide OKR rollout."),
    ],
    "J07": [
        ("Multi-Region AWS Migration", "Architected a multi-region aws cloud architecture migration with terraform, improving networking resilience."),
        ("Zero Trust Security Rollout", "Designed a zero-trust security model across cloud networking infrastructure."),
        ("Kubernetes Platform Blueprint", "Built a reusable kubernetes platform blueprint for internal teams adopting aws cloud architecture."),
    ],
    "J08": [
        ("Regression Suite Automation", "Built a selenium-based test automation suite in python integrated into the ci/cd pipeline."),
        ("API Test Framework", "Developed an api testing framework covering over 200 endpoints with automated regression checks."),
        ("Cross-browser Test Grid", "Set up a cross-browser selenium grid to run test automation suites in parallel."),
    ],
    "J09": [
        ("Data Warehouse ETL Pipeline", "Built an etl pipeline in python and spark feeding a sql data warehouse, orchestrated with airflow."),
        ("Real-time Data Pipeline", "Designed real-time data pipelines using spark streaming deployed on aws."),
        ("Airflow Orchestration Revamp", "Migrated legacy cron-based etl jobs to airflow-orchestrated data pipelines."),
    ],
    "J10": [
        ("Checkout Flow Redesign", "Led user research and prototyping in figma to redesign a checkout flow, improving conversion."),
        ("Mobile App Wireframe Set", "Produced wireframing and ux design deliverables for a new mobile app from concept to prototype."),
        ("Usability Testing Program", "Ran a usability testing and user research program informing key ux design decisions."),
    ],
}

UNRELATED_PROJECTS = [
    ("Community Garden Fundraiser", "Organized a fundraising event for a local community garden initiative."),
    ("Travel Blog", "Wrote and maintained a travel blog chronicling backpacking trips across Southeast Asia."),
    ("Youth Soccer Coaching", "Coached a youth soccer team, focusing on teamwork and fundamentals."),
    ("Home Renovation Project", "Planned and executed a full kitchen renovation on a fixed budget."),
    ("Book Club Organizer", "Founded and ran a monthly book club with over 20 regular members."),
    ("Charity Bake Sale", "Coordinated a bake sale that raised funds for a local animal shelter."),
    ("Photography Portfolio", "Built a personal portfolio showcasing landscape and street photography."),
    ("Local Trivia Night Host", "Hosted a weekly trivia night at a neighborhood bar."),
]


# ---------------------------------------------------------------------------
# Education fields: which fields count as relevant / adjacent for each job.
# ---------------------------------------------------------------------------
_TECH_FIELDS = ["Computer Science", "Software Engineering", "Information Technology", "Computer Engineering"]
_ADJACENT_STEM = ["Electrical Engineering", "Mechanical Engineering", "Physics", "Mathematics", "Business Analytics"]
UNRELATED_FIELDS = ["Fine Arts", "History", "Business Administration", "Hospitality Management", "Marketing"]

JOB_FIELDS = {
    "J01": (["Computer Science", "Statistics", "Data Science", "Mathematics"], ["Electrical Engineering", "Physics", "Business Analytics"]),
    "J02": (_TECH_FIELDS, _ADJACENT_STEM),
    "J03": (_TECH_FIELDS, _ADJACENT_STEM + ["Graphic Design"]),
    "J04": (_TECH_FIELDS, _ADJACENT_STEM),
    "J05": (["Computer Science", "Artificial Intelligence", "Data Science", "Computer Engineering"], ["Statistics", "Mathematics", "Electrical Engineering", "Physics"]),
    "J06": (None, None),  # "any field"
    "J07": (_TECH_FIELDS, _ADJACENT_STEM),
    "J08": (_TECH_FIELDS, _ADJACENT_STEM),
    "J09": (_TECH_FIELDS + ["Data Science"], _ADJACENT_STEM + ["Statistics"]),
    "J10": (["Design", "Human-Computer Interaction", "Graphic Design", "Cognitive Science"], ["Psychology", "Marketing", "Computer Science"]),
}
for _job in JOBS:
    _rel, _adj = JOB_FIELDS[_job["id"]]
    _job["related_fields"] = _rel
    _job["adjacent_fields"] = _adj

EDU_LABEL = {"associate": "Associate's", "bachelor": "Bachelor's", "master": "Master's", "phd": "PhD"}
EDU_RANK = {"associate": 2, "bachelor": 3, "master": 4, "phd": 5}

SOFT_SKILLS = ["communication", "teamwork", "problem solving", "time management", "leadership"]
FILLER = [
    "Known for clear communication and reliable delivery in cross-functional teams.",
    "Comfortable working in fast-paced, collaborative environments.",
    "Consistently focused on measurable outcomes and continuous improvement.",
    "Enjoys mentoring teammates and documenting best practices.",
    "Proactive self-starter who takes ownership of ambiguous problems.",
    "Passionate about learning new tools and sharing knowledge with the team.",
]

FIRST_NAMES = [
    "Ava", "Liam", "Maya", "Noah", "Zara", "Ethan", "Priya", "Lucas", "Sofia", "Daniel",
    "Aisha", "Mateo", "Hana", "Omar", "Elena", "Kenji", "Leila", "Samuel", "Nina", "Arjun",
    "Chloe", "Diego", "Fatima", "Gabriel", "Isla", "Jonas", "Kavya", "Tomas", "Yara", "Felix",
]
LAST_NAMES = [
    "Patel", "Kim", "Garcia", "Chen", "Okafor", "Nguyen", "Rossi", "Johnson", "Muller", "Silva",
    "Khan", "Tanaka", "Ivanov", "Haddad", "Santos", "Larsen", "Mehta", "Dubois", "Cohen", "Reyes",
]
NAMES = [f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES]  # 600 unique names

RESUME_TEMPLATES = [
    "{name} is a {prev_title} professional with {experience_years} years of experience, holding a {education}. "
    "Core skills include {skills_str}. Hands-on experience with {tech_str}. Recent work: {project_summary} {filler}",

    "{name} brings {experience_years} years of experience, most recently in {prev_title} roles, with a {education}. "
    "Technical strengths span {skills_str}, alongside practical exposure to {tech_str}. {project_summary} {filler}",

    "With {experience_years} years in the field and a {education}, {name} has developed expertise in "
    "{skills_str}. Tooling experience includes {tech_str}. {project_summary} {filler}",

    "{name} ({education}) has spent {experience_years} years building skills in {skills_str}, "
    "complemented by familiarity with {tech_str}. {project_summary} {filler}",

    "{prev_title} with {experience_years} years of experience. {name} holds a {education} and works with "
    "{tech_str}. Skills: {skills_str}. {project_summary} {filler}",

    "Experienced {prev_title} ({experience_years} years). Education: {education}. {name} has worked across "
    "{skills_str} and tools such as {tech_str}. {project_summary} {filler}",
]

# Canonical skill -> alternative spellings a real candidate might use.
ALIAS_DISPLAY = {}
for _alias, _canon in ALIASES.items():
    ALIAS_DISPLAY.setdefault(_canon, []).append(_alias)

ALL_SKILLS = sorted({s for j in JOBS for s in j["required_skills"] + j["preferred_skills"] + j["technologies"]})


def clip01(x):
    return max(0.0, min(1.0, x))


def write_skill(canon):
    """Render a canonical skill the way a candidate might actually write it."""
    options = ALIAS_DISPLAY.get(canon)
    if options and random.random() < 0.30:
        return random.choice(options)
    if random.random() < 0.15:
        return canon.title()
    return canon


def bernoulli_subset(items, p_mean, spread=0.15):
    """Include each item independently; the per-resume probability is itself noisy."""
    p = clip01(random.gauss(p_mean, spread))
    return [x for x in items if random.random() < p]


def sample_latent_fit(bucket):
    """Latent 'how good a candidate is for their own field' in [0, 1]."""
    if bucket == "strong":
        return random.betavariate(6, 1.8)
    if bucket == "medium":
        return random.betavariate(4, 4)
    return random.betavariate(1.8, 6)


def quality_buckets(n):
    """Exactly 35% strong / 35% medium / 30% weak, shuffled (stratified per job)."""
    n_strong, n_medium = round(0.35 * n), round(0.35 * n)
    buckets = ["strong"] * n_strong + ["medium"] * n_medium + ["weak"] * (n - n_strong - n_medium)
    random.shuffle(buckets)
    return buckets


def pick_education(persona, q):
    w_assoc = 0.10 * (1 - q) + 0.02
    w_master = 0.12 + 0.22 * q
    w_phd = 0.02 + 0.06 * q
    w_bach = 1 - w_assoc - w_master - w_phd
    level = random.choices(["associate", "bachelor", "master", "phd"], weights=[w_assoc, w_bach, w_master, w_phd])[0]
    rel, adj = JOB_FIELDS[persona["id"]]
    if rel is None:
        field = random.choice(["Business Administration", "Economics", "Computer Science", "Marketing", "Psychology", "Engineering"])
    else:
        r = random.random()
        p_rel = 0.30 + 0.65 * q
        if r < p_rel:
            field = random.choice(rel)
        elif r < p_rel + (1 - p_rel) * 0.5:
            field = random.choice(adj)
        else:
            field = random.choice(UNRELATED_FIELDS)
    return level, field


def pick_projects(persona_id, q):
    n_total = random.choice([2, 3, 3])
    n_rel = sum(random.random() < (0.10 + 0.85 * q) for _ in range(n_total))
    chosen = random.sample(RELEVANT_PROJECTS[persona_id], k=min(n_rel, len(RELEVANT_PROJECTS[persona_id])))
    chosen += random.sample(UNRELATED_PROJECTS, k=n_total - len(chosen))
    random.shuffle(chosen)
    return [{"title": t, "description": d} for t, d in chosen]


def field_factor(job, field):
    rel, adj = job.get("related_fields"), job.get("adjacent_fields")
    if rel is None:
        return 1.0
    if field in rel:
        return 1.0
    if field in adj:
        return 0.8
    return 0.55


def count_relevant_projects(projects, job):
    terms = job["required_skills"] + job["technologies"]
    return sum(
        any(mentions(p["title"] + " " + p["description"], t) for t in terms) for p in projects
    )


def recruiter_decision(cand, job):
    """
    Simulated recruiter ground truth, computed from the candidate's *true*
    attributes. Deliberately NOT the same formula as the model's 7 features:
    it uses absolute year-gaps, ordinal degree gaps, a hard 'missing too many
    required skills' penalty, per-recruiter subjectivity, and random flips.
    The classifier only sees noisy engineered features, so it has to learn an
    approximation rather than invert a known formula.
    """
    canon = cand["_canon_skills"]
    canon_tech = cand["_canon_tech"]
    req = len(canon & normalize_skills(job["required_skills"])) / len(job["required_skills"])
    pref = len(canon & normalize_skills(job["preferred_skills"])) / len(job["preferred_skills"])
    tech = len(canon_tech & normalize_skills(job["technologies"])) / len(job["technologies"])
    exp_term = 1 / (1 + math.exp(-(cand["experience_years"] - job["min_experience"] + 0.5) / 0.8))

    gap = EDU_RANK[job["education_level"]] - EDU_RANK[cand["_edu_level"]]
    level_term = 1.0 if gap <= 0 else (0.6 if gap == 1 else 0.3)
    edu_term = level_term * field_factor(job, cand["_edu_field"])

    proj_term = min(count_relevant_projects(cand["projects"], job) / 2, 1.0)

    s = 0.36 * req + 0.07 * pref + 0.13 * tech + 0.17 * exp_term + 0.12 * edu_term + 0.15 * proj_term
    if req < 0.4:
        s -= 0.12
    if req >= 0.99 and exp_term > 0.6:
        s += 0.05
    s += random.gauss(0, 0.07)
    label = int(s >= LABEL_THRESHOLD)
    if random.random() < 0.04:
        label = 1 - label
    return label, s


def build_resume(idx, persona, job, q, name):
    # Skills / technologies / experience are generated relative to the persona's own field
    skills = bernoulli_subset(persona["required_skills"], q)
    skills += bernoulli_subset(persona["preferred_skills"], q * 0.85)
    technologies = bernoulli_subset(persona["technologies"], q, spread=0.2)
    # Distractors: real candidates list extra skills that are irrelevant (or partly relevant) to this job
    distractors = random.sample([s for s in ALL_SKILLS if s not in skills], k=random.randint(1, 4))
    skills += distractors
    skills += random.sample(SOFT_SKILLS, k=random.randint(0, 2))
    random.shuffle(skills)

    years = round(persona["min_experience"] * (0.25 + 1.5 * q) + random.gauss(0, 1.1))
    if random.random() < 0.05:
        years += 3  # long-tenured outliers
    years = max(0, years)

    level, field = pick_education(persona, q)
    education = f"{EDU_LABEL[level]} in {field}"
    projects = pick_projects(persona["id"], q)

    written_skills = [write_skill(s) for s in skills]
    written_tech = [write_skill(t) for t in technologies]
    project_summary = " ".join(f"Led \"{p['title']}\": {p['description']}" for p in projects)
    resume_text = random.choice(RESUME_TEMPLATES).format(
        name=name,
        prev_title=persona["title"],
        experience_years=years,
        education=education,
        skills_str=", ".join(written_skills) if written_skills else "general professional skills",
        tech_str=", ".join(written_tech) if written_tech else "standard office tooling",
        project_summary=project_summary,
        filler=random.choice(FILLER),
    )

    cand = {
        "id": f"R{idx:04d}",
        "job_id": job["id"],
        "persona_job_id": persona["id"],
        "name": name,
        "skills": written_skills,
        "experience_years": years,
        "education": education,
        "education_level": level,
        "education_field": field,
        "technologies": written_tech,
        "projects": projects,
        "resume_text": resume_text,
        "_canon_skills": normalize_skills(skills),
        "_canon_tech": normalize_skills(technologies),
        "_edu_level": level,
        "_edu_field": field,
    }
    cand["label"], cand["recruiter_score"] = recruiter_decision(cand, job)
    if persona["id"] != job["id"]:
        cand["tier"] = "cross-role"
    else:
        cand["tier"] = "strong" if q >= 0.7 else ("medium" if q >= 0.4 else "weak")
    cand["recruiter_score"] = round(cand["recruiter_score"], 4)
    return cand


LABEL_THRESHOLD = 0.55


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-resumes", type=int, default=600, help="total resume-job pairs (default 600)")
    parser.add_argument("--cross-role-frac", type=float, default=0.30,
                        help="fraction of resumes evaluated against a job other than their own field")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    if args.num_resumes > len(NAMES):
        raise SystemExit(f"--num-resumes is capped at {len(NAMES)} (unique names).")

    DATA_DIR.mkdir(exist_ok=True)
    names = random.sample(NAMES, args.num_resumes)

    per_job = [args.num_resumes // len(JOBS) + (i < args.num_resumes % len(JOBS)) for i in range(len(JOBS))]
    resumes = []
    idx = 0
    for persona, n in zip(JOBS, per_job):
        for bucket in quality_buckets(n):
            job = persona
            if random.random() < args.cross_role_frac:
                job = random.choice([j for j in JOBS if j["id"] != persona["id"]])
            resumes.append(build_resume(idx, persona, job, sample_latent_fit(bucket), names[idx]))
            idx += 1
    random.shuffle(resumes)
    for new_idx, r in enumerate(resumes):  # re-number after shuffle so ids are in file order
        r["id"] = f"R{new_idx:04d}"
    for r in resumes:
        for k in [k for k in r if k.startswith("_")]:
            del r[k]

    with open(DATA_DIR / "jobs.json", "w", encoding="utf-8") as f:
        json.dump(JOBS, f, indent=2)
    with open(DATA_DIR / "resumes.json", "w", encoding="utf-8") as f:
        json.dump(resumes, f, indent=2)

    print(f"Wrote {len(JOBS)} jobs and {len(resumes)} resumes to {DATA_DIR}")
    print("Tier distribution:", dict(Counter(r["tier"] for r in resumes)))
    print(f"Recruiter label: {sum(r['label'] for r in resumes)}/{len(resumes)} good matches")


if __name__ == "__main__":
    main()
