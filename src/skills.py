"""
Skill normalization (slide 3, objective 2): map the many ways a candidate can
write a skill ("ML", "k8s", "Amazon Web Services") onto one canonical name.
"""
import re

# alias (lowercase) -> canonical skill name (lowercase)
ALIASES = {
    "ml": "machine learning",
    "dl": "deep learning",
    "natural language processing": "nlp",
    "js": "javascript",
    "ecmascript": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "amazon web services": "aws",
    "postgres": "postgresql",
    "restful apis": "rest apis",
    "rest api": "rest apis",
    "restful services": "rest apis",
    "cicd": "ci/cd",
    "ci-cd": "ci/cd",
    "continuous integration": "ci/cd",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "pyspark": "spark",
    "apache spark": "spark",
    "apache kafka": "kafka",
    "apache airflow": "airflow",
    "torch": "pytorch",
    "ab testing": "a/b testing",
    "a-b testing": "a/b testing",
    "ux": "ux design",
    "user experience design": "ux design",
    "stats": "statistics",
    "statistical analysis": "statistics",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
    "spring": "spring boot",
    "test automation framework": "test automation",
    "automated testing": "testing",
    "data pipeline": "data pipelines",
    "html5": "html",
    "css3": "css",
    "reactjs": "react",
    "react.js": "react",
    "nextjs": "next.js",
}


def normalize_skill(skill: str) -> str:
    s = re.sub(r"\s+", " ", skill.strip().lower())
    return ALIASES.get(s, s)


def normalize_skills(skills) -> set:
    return {normalize_skill(s) for s in skills if s and s.strip()}


def mentions(text: str, term: str) -> bool:
    """Whole-token match, so 'java' does not fire on 'javascript'."""
    pattern = r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None
