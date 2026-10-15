"""
Author the synthetic resume-job dataset directly (no billed Anthropic API
calls) using hand-crafted job templates, skill/project pools, and resume
templates, combined programmatically into 100 resume-job pairs with
controlled match tiers (strong/medium/weak).

This replaces generate_dataset.py for users who don't want to spend API
credits: the content (job descriptions, skill taxonomies, project pools,
resume phrasing) is authored directly; this script only handles the
combinatorics of assembling it into 100 varied, labeled rows.

Usage:
    python src/author_dataset.py
"""
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
random.seed(42)

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

FIRST_NAMES = ["Ava", "Liam", "Maya", "Noah", "Zara", "Ethan", "Priya", "Lucas", "Sofia", "Daniel"]
LAST_NAMES = ["Patel", "Kim", "Garcia", "Chen", "Okafor", "Nguyen", "Rossi", "Johnson", "Muller", "Silva"]
NAMES = [f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES]  # 100 unique names

RESUME_TEMPLATES = [
    "{name} is a {title} professional with {experience_years} years of experience, holding a {education}. "
    "Core skills include {skills_str}. Hands-on experience with {tech_str}. Recent work: {project_summary}. "
    "Known for clear communication and reliable delivery in cross-functional teams.",

    "{name} brings {experience_years} years of experience relevant to {title} roles, with a {education}. "
    "Technical strengths span {skills_str}, alongside practical exposure to {tech_str}. {project_summary} "
    "Comfortable working in fast-paced, collaborative environments.",

    "With {experience_years} years in the field and a {education}, {name} has developed expertise in "
    "{skills_str}. Tooling experience includes {tech_str}. {project_summary} Consistently focused on "
    "measurable outcomes and continuous improvement.",

    "{name} ({education}) has spent {experience_years} years building skills in {skills_str}, "
    "complemented by familiarity with {tech_str}. {project_summary} Enjoys mentoring teammates and "
    "documenting best practices along the way.",
]

TIER_COUNTS = [("strong", 4), ("medium", 3), ("weak", 3)]


def pick_subset(items, frac_low, frac_high):
    n = len(items)
    k = max(0, min(n, round(n * random.uniform(frac_low, frac_high))))
    return random.sample(items, k) if k > 0 else []


def education_for_tier(job, tier):
    if tier == "strong":
        level, field = job["education_level"], job["education_field"]
    elif tier == "medium":
        level, field = ("bachelor" if job["education_level"] == "master" else "associate"), job["education_field"]
    else:
        level, field = random.choice(["associate", "bachelor"]), random.choice(
            ["Fine Arts", "History", "Business Administration", "Hospitality Management"]
        )
    label = {"associate": "Associate's", "bachelor": "Bachelor's", "master": "Master's"}[level]
    return f"{label} in {field}"


def experience_for_tier(job, tier):
    base = job["min_experience"]
    if tier == "strong":
        return base + random.randint(0, 3)
    if tier == "medium":
        return max(1, base - 1)
    return max(0, base - random.randint(2, 3))


def projects_for_tier(job_id, tier):
    relevant = RELEVANT_PROJECTS[job_id]
    if tier == "strong":
        chosen = random.sample(relevant, k=min(len(relevant), random.choice([2, 3])))
    elif tier == "medium":
        chosen = random.sample(relevant, k=1) + random.sample(UNRELATED_PROJECTS, k=1)
    else:
        chosen = random.sample(UNRELATED_PROJECTS, k=random.choice([1, 2]))
    return [{"title": t, "description": d} for t, d in chosen]


def build_resume(idx, job, tier, name):
    skills_frac = {"strong": (0.8, 1.0), "medium": (0.4, 0.65), "weak": (0.0, 0.25)}[tier]
    tech_frac = {"strong": (0.75, 1.0), "medium": (0.35, 0.6), "weak": (0.0, 0.25)}[tier]
    pref_frac = {"strong": (0.33, 1.0), "medium": (0.0, 0.5), "weak": (0.0, 0.0)}[tier]

    skills = pick_subset(job["required_skills"], *skills_frac)
    skills += pick_subset(job["preferred_skills"], *pref_frac)
    technologies = pick_subset(job["technologies"], *tech_frac)
    experience_years = experience_for_tier(job, tier)
    education = education_for_tier(job, tier)
    projects = projects_for_tier(job["id"], tier)

    template = RESUME_TEMPLATES[idx % len(RESUME_TEMPLATES)]
    project_summary = " ".join(
        f"Led \"{p['title']}\": {p['description']}" for p in projects
    )
    resume_text = template.format(
        name=name,
        title=job["title"],
        experience_years=experience_years,
        education=education,
        skills_str=", ".join(skills) if skills else "general professional and organizational skills",
        tech_str=", ".join(technologies) if technologies else "standard office tooling",
        project_summary=project_summary,
    )

    return {
        "id": f"R{idx:04d}",
        "job_id": job["id"],
        "tier": tier,
        "name": name,
        "skills": skills,
        "experience_years": experience_years,
        "education": education,
        "technologies": technologies,
        "projects": projects,
        "resume_text": resume_text,
    }


def main():
    DATA_DIR.mkdir(exist_ok=True)

    with open(DATA_DIR / "jobs.json", "w", encoding="utf-8") as f:
        json.dump(JOBS, f, indent=2)

    resumes = []
    idx = 0
    for job in JOBS:
        tiers = [t for t, count in TIER_COUNTS for _ in range(count)]
        random.shuffle(tiers)
        for tier in tiers:
            resumes.append(build_resume(idx, job, tier, NAMES[idx]))
            idx += 1

    with open(DATA_DIR / "resumes.json", "w", encoding="utf-8") as f:
        json.dump(resumes, f, indent=2)

    print(f"Wrote {len(JOBS)} jobs and {len(resumes)} resumes to {DATA_DIR}")
    from collections import Counter
    print("Tier distribution:", Counter(r["tier"] for r in resumes))


if __name__ == "__main__":
    main()
