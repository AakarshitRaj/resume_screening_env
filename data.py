"""
data.py — Static resume and job description data for all 3 tasks.
"""

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 · binary_screen (Easy)
# Scenario: Resume clearly under-qualifies on years of experience → REJECT
# ─────────────────────────────────────────────────────────────────────────────

TASK1_JD = """
Position: Senior Python Developer
Company: TechCorp Inc.

REQUIREMENTS (all mandatory):
- 5+ years of professional Python development experience
- Bachelor's degree in Computer Science or related field
- Strong experience with Django or Flask web frameworks
- Proficiency in SQL databases (PostgreSQL preferred)
- Experience with RESTful API design and implementation
- Knowledge of Docker and Kubernetes for deployment
- Ability to mentor junior developers

RESPONSIBILITIES:
- Design and implement scalable backend services
- Lead technical code reviews
- Collaborate with product and infrastructure teams
""".strip()

TASK1_RESUME = """
John Smith   |   john.smith@email.com   |   github.com/jsmith

SUMMARY
Motivated Python developer with 2 years of hands-on experience building
web applications. Passionate about clean code and agile practices.

EXPERIENCE
Junior Python Developer — StartupXYZ  (Jan 2022 – Present · 2 years)
  • Developed REST APIs using Flask for internal tooling
  • Wrote unit tests achieving 80% code coverage
  • Assisted in PostgreSQL schema design
  • Participated in weekly code reviews

Software Engineer Intern — LocalAgency  (Jun 2021 – Dec 2021 · 6 months)
  • Built Python automation scripts for data processing
  • Fixed front-end bugs in JavaScript/React

EDUCATION
B.S. Computer Science — State University (2021)

SKILLS
Python, Flask, PostgreSQL, REST APIs, Git, Linux, Docker (basic), JavaScript

CERTIFICATIONS
AWS Cloud Practitioner (2023)
""".strip()

TASK1_GROUND_TRUTH = {
    "decision": "REJECT",
    "key_reason": "insufficient_experience",
    "explanation": (
        "Candidate has ~2.5 years total experience (2 yr + 6-month internship) "
        "vs the mandatory 5+ year requirement. The experience gap is disqualifying "
        "regardless of other partially met qualifications."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 · skill_match (Medium)
# Scenario: Data Scientist role — 7 of 10 required skills present → score ≈ 0.70
# ─────────────────────────────────────────────────────────────────────────────

TASK2_JD = """
Position: Data Scientist
Company: Analytics Corp

REQUIRED SKILLS (evaluate each individually):
  1. Python (advanced)
  2. SQL
  3. Machine Learning (scikit-learn / production experience)
  4. Statistical Analysis
  5. TensorFlow or PyTorch (deep learning framework)
  6. Data Visualization (Matplotlib, Tableau, or Power BI)
  7. AWS or GCP (cloud platform)
  8. Feature Engineering
  9. A/B Testing
  10. Communication & Presentation

RESPONSIBILITIES:
- Build and deploy ML models to production
- Analyse large datasets to extract business insights
- Create dashboards for stakeholders
- Collaborate with engineering on data pipelines
""".strip()

TASK2_RESUME = """
Sarah Johnson   |   sarah.j@email.com   |   linkedin.com/in/sarahjohnson

SUMMARY
Data Scientist with 4 years of experience in analytics and ML model development.
Strong foundation in statistical analysis and Python-based ML pipelines.

EXPERIENCE
Data Scientist — FinTech Solutions  (2021 – Present)
  • Built customer churn prediction models using scikit-learn (87% accuracy)
  • Conducted A/B tests for new product feature launches
  • Created Tableau dashboards presented to C-suite
  • Partnered with engineering on data pipeline architecture

Junior Data Analyst — RetailCo  (2019 – 2021)
  • Wrote complex SQL queries for customer behaviour analysis
  • Automated Excel reports using Python (Pandas, openpyxl)
  • Delivered weekly insight presentations to product team

EDUCATION
M.S. Statistics — Data University (2019)

SKILLS
Python (advanced), SQL, scikit-learn, Statistical Analysis, A/B Testing,
Tableau, Pandas, NumPy, Matplotlib, R (basic)

NOTABLE PROJECTS
  • Customer Segmentation — K-Means clustering pipeline
  • Time Series Forecasting — ARIMA models for demand planning
""".strip()

TASK2_REQUIRED_SKILLS = [
    "Python",
    "SQL",
    "Machine Learning",
    "Statistical Analysis",
    "TensorFlow or PyTorch",
    "Data Visualization",
    "AWS or GCP",
    "Feature Engineering",
    "A/B Testing",
    "Communication",
]

TASK2_GROUND_TRUTH = {
    # 7 skills clearly present in resume
    "matched_skills": [
        "Python",
        "SQL",
        "Machine Learning",
        "Statistical Analysis",
        "Data Visualization",
        "A/B Testing",
        "Communication",
    ],
    # 3 skills absent
    "missing_skills": [
        "TensorFlow or PyTorch",
        "AWS or GCP",
        "Feature Engineering",
    ],
    "score": 0.70,   # 7/10
    "explanation": (
        "Candidate is strong on core data science skills but has no cloud experience, "
        "no deep learning framework (TF/PyTorch), and no explicit feature engineering work."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 · rank_candidates (Hard)
# Scenario: 5 PM candidates; ground-truth ranking based on JD fit
# ─────────────────────────────────────────────────────────────────────────────

TASK3_JD = """
Position: Senior Product Manager
Company: GrowthCo

REQUIREMENTS:
- 6+ years of product management experience
- Proven track record with B2B SaaS products
- Data-driven decision making (SQL proficiency, analytics tools)
- Successful product launches with measurable impact
- Cross-functional team leadership
- MBA preferred
- Agile / Scrum experience
- Executive-level stakeholder communication
""".strip()

TASK3_RESUMES = [
    # index 0 — rank 3 (good but all B2C, no B2B SaaS)
    """
CANDIDATE 0 — Alex Chen
Experience: 7 years in Product Management
  • Senior PM at ConsumerApp (4 yr) — iOS/Android B2C
  • PM at E-Commerce Startup (3 yr) — B2C marketplace
Education: MBA, Top Business School (2016)
Skills: Product roadmapping, A/B testing, SQL (intermediate), Agile, User research
Notable: Led 3 major consumer app launches; 2 M+ users acquired
Gap: ALL experience is B2C consumer — zero B2B SaaS exposure
""".strip(),

    # index 1 — rank 1 (best match overall)
    """
CANDIDATE 1 — Maria Rodriguez
Experience: 8 years in Product Management
  • Director of Product at B2B SaaS Co (3 yr) — enterprise HR platform
  • Senior PM at Enterprise Software Inc (5 yr) — CRM/ERP tools
Education: MBA Stanford (2015), B.S. Computer Science
Skills: B2B SaaS, SQL (advanced), Salesforce, Agile/Scrum, Data analytics,
        Stakeholder management, OKR framework
Notable: 5 product launches growing ARR by $12 M; managed team of 4 PMs
Strength: Strongest B2B SaaS depth + technical background + leadership
""".strip(),

    # index 2 — rank 5 (weakest; very junior)
    """
CANDIDATE 2 — Tom Wilson
Experience: 2 years in Product Management
  • Associate PM at TechStartup (2 yr) — mobile app features
Education: B.S. Marketing (2021)
Skills: Jira, Confluence, User stories, Google Analytics (basic)
Notable: Contributed to 2 minor feature releases
Gap: Only 2 years (need 6+); no B2B, no SQL, no launch ownership
""".strip(),

    # index 3 — rank 2 (strong; minor gaps vs Maria)
    """
CANDIDATE 3 — Priya Patel
Experience: 6 years in Product Management
  • Senior PM at SaaS Platform (3 yr) — B2B project management tool
  • PM at B2B Analytics Tool (3 yr) — enterprise reporting SaaS
Education: M.S. Information Systems (2017), B.S. Engineering
Skills: B2B SaaS, SQL (advanced), Python (basic), Scrum, OKR, Cross-functional leadership
Notable: 4 B2B product launches; NPS improved 28 pts; $8 M ARR product roadmap
Gap: No MBA (preferred); slightly less executive stakeholder exposure than Maria
""".strip(),

    # index 4 — rank 4 (average; borderline experience)
    """
CANDIDATE 4 — David Kim
Experience: 5 years in Product Management  (1 year short of requirement)
  • PM at Mid-size SaaS (3 yr) — B2B project tools
  • PM at Consulting Firm (2 yr) — internal tools
Education: MBA (2018)
Skills: Product strategy, Roadmapping, SQL (basic), Agile, Stakeholder communication
Notable: 2 product launches; improved user retention 15%
Gap: 5 yr vs 6 yr required; limited data-analytics depth; modest impact metrics
""".strip(),
]

# Ground-truth ranking: list of candidate indices ordered BEST → WORST
# 1 (Maria) → 3 (Priya) → 0 (Alex) → 4 (David) → 2 (Tom)
TASK3_GROUND_TRUTH = {
    "ranking": [1, 3, 0, 4, 2],
    "scores": {0: 0.65, 1: 0.95, 2: 0.10, 3: 0.85, 4: 0.50},
    "explanation": (
        "Maria(1) best fits all criteria. Priya(3) is strong on B2B SaaS. "
        "Alex(0) experienced but B2C only. David(4) borderline on experience/depth. "
        "Tom(2) far too junior."
    ),
}
