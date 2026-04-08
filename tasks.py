"""
tasks.py — Three graded resume screening tasks.

Task 1 · binary_screen   (easy)    2 steps   max reward 1.00
Task 2 · skill_match     (medium)  3 steps   max reward 1.00
Task 3 · rank_candidates (hard)    4 steps   max reward 1.00

Each task exposes:
  reset()          → {"observation": Observation, "internal": dict}
  step(state, act) → {"observation", "state", "reward", "done", "info"}
  grade(...)       → float in [0, 1]  (pure deterministic grader)
"""

from __future__ import annotations

from typing import Any, Dict, List

from env import Action, Observation
from data import (
    TASK1_GROUND_TRUTH,
    TASK1_JD,
    TASK1_RESUME,
    TASK2_GROUND_TRUTH,
    TASK2_JD,
    TASK2_REQUIRED_SKILLS,
    TASK2_RESUME,
    TASK3_GROUND_TRUTH,
    TASK3_JD,
    TASK3_RESUMES,
)


def _strict(value: float) -> float:
    """Clamp reward/score to strictly (0.01, 0.99) — safe for :.2f formatting."""
    return round(max(0.01, min(0.99, float(value))), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 · binary_screen  (Easy)
# ─────────────────────────────────────────────────────────────────────────────

class BinaryScreenTask:
    """
    Binary ACCEPT / REJECT screening.

    Episode flow
    ─────────────
    Step 1  Analyze  — partial reward for spotting the experience gap.
    Step 2  Decide   — main reward for correct REJECT + good reasoning.

    Max reward breakdown
    ────────────────────
    Step 1 analysis quality  →  up to 0.25
    Step 2 correct decision  →  0.65
    Step 2 correct reasoning →  0.10
    ────────────────────────────── total 1.00
    """

    MAX_STEPS = 2
    _CORRECT = TASK1_GROUND_TRUTH["decision"]  # "REJECT"

    def reset(self) -> Dict[str, Any]:
        obs = Observation(
            task_name="binary_screen",
            step=0,
            job_description=TASK1_JD,
            resume=TASK1_RESUME,
            feedback=(
                "You are screening a resume for the job description above.\n"
                "Step 1 of 2: Provide your initial analysis. "
                "Identify whether the candidate meets key requirements."
            ),
        )
        return {"observation": obs, "internal": {"step": 0}}

    def step(self, state: Dict, action: Action) -> Dict[str, Any]:
        internal = state["internal"]
        step = internal["step"] + 1
        reward = 0.0
        done = False
        feedback = ""
        new_internal = {**internal, "step": step}

        if step == 1:
            # ── Analysis step ──────────────────────────────────────────────
            text = (action.content or "").lower()

            # Did the agent identify the critical experience gap?
            mentions_yrs = any(
                kw in text
                for kw in ["2 year", "2-year", "two year", "years of exp",
                            "experience gap", "insufficient exp", "junior",
                            "not enough exp"]
            )
            mentions_req = any(
                kw in text
                for kw in ["5 year", "5+", "five year", "requires", "minimum",
                            "mandatory", "does not meet", "lacks"]
            )
            mentions_qualify = any(
                kw in text
                for kw in ["qualif", "meets", "fails", "missing", "short"]
            )

            reward += 0.10 if mentions_yrs else 0.0
            reward += 0.10 if mentions_req else 0.0
            reward += 0.05 if mentions_qualify else 0.0

            feedback = (
                "Analysis noted.\n"
                "Step 2 of 2: Make your final screening decision. "
                "Set 'decision' to ACCEPT or REJECT and explain your reasoning in 'content'."
            )

        elif step == 2:
            # ── Decision step ──────────────────────────────────────────────
            decision = (action.decision or "").upper().strip()
            text = (action.content or "").lower()

            if decision == self._CORRECT:
                reward += 0.65  # correct decision
                # Bonus: did reasoning cite the experience gap?
                if any(kw in text for kw in
                       ["experience", "years", "2 year", "insufficient",
                        "junior", "requirement", "5 year", "5+"]):
                    reward += 0.10
            # Wrong decision → 0.0 (no partial credit — this is the binary signal)

            done = True
            feedback = (
                f"Decision recorded: {decision or '(none)'}. "
                f"Ground truth: {self._CORRECT}. Episode complete."
            )

        new_obs = Observation(
            task_name="binary_screen",
            step=step,
            job_description=TASK1_JD,
            resume=TASK1_RESUME,
            feedback=feedback,
            done=done,
        )
        # Clamp step reward strictly within (0, 1)
        reward = max(0.01, min(0.99, reward))
        return {
            "observation": new_obs,
            "state": {"observation": new_obs, "internal": new_internal},
            "reward": _strict(reward),
            "done": done,
            "info": {"step": step, "ground_truth_decision": self._CORRECT},
        }

    @staticmethod
    def grade(decision: str) -> float:
        """Deterministic terminal grader (ignores reasoning)."""
        return 0.99 if decision.upper().strip() == TASK1_GROUND_TRUTH["decision"] else 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 · skill_match  (Medium)
# ─────────────────────────────────────────────────────────────────────────────

class SkillMatchTask:
    """
    Skill matching and scoring against a Data Scientist JD.

    Episode flow
    ─────────────
    Step 1  List MATCHED skills    → partial reward (F1 vs ground truth)
    Step 2  List MISSING skills    → partial reward (F1 vs ground truth)
    Step 3  Provide match SCORE    → reward based on proximity to 0.70

    Max reward breakdown
    ────────────────────
    Step 1  matched_skills F1   →  up to 0.30
    Step 2  missing_skills F1   →  up to 0.30
    Step 3  score proximity     →  up to 0.40
    ────────────────────────────── total 1.00
    """

    MAX_STEPS = 3
    _GT = TASK2_GROUND_TRUTH

    # Canonical lowercase names and their acceptable aliases
    _ALIASES: Dict[str, List[str]] = {
        "python": ["python", "py"],
        "sql": ["sql", "structured query", "postgresql", "postgres", "mysql"],
        "machine learning": ["machine learning", "ml", "scikit", "sklearn", "ml models"],
        "statistical analysis": ["statistical", "statistics", "stats"],
        "tensorflow or pytorch": ["tensorflow", "pytorch", "torch", "tf", "keras", "deep learning"],
        "data visualization": ["visualization", "visualisation", "tableau", "matplotlib",
                               "plotly", "power bi", "dashboar"],
        "aws or gcp": ["aws", "amazon web", "gcp", "google cloud", "azure", "cloud platform"],
        "feature engineering": ["feature engineer", "feature extract", "feature select"],
        "a/b testing": ["a/b test", "ab test", "split test", "experiment"],
        "communication": ["communication", "presentation", "stakeholder", "present"],
    }

    def _normalize(self, skill: str) -> str:
        s = skill.lower().strip()
        for canonical, aliases in self._ALIASES.items():
            if any(alias in s for alias in aliases):
                return canonical
        return s

    def _f1(self, agent_list: List[str], ground_truth: List[str]) -> float:
        if not agent_list:
            return 0.01
        gt = {self._normalize(s) for s in ground_truth}
        ag = {self._normalize(s) for s in agent_list}
        tp = len(gt & ag)
        prec = tp / len(ag) if ag else 0.0
        rec = tp / len(gt) if gt else 0.0
        if prec + rec == 0:
            return 0.01
        raw = 2 * prec * rec / (prec + rec)
        return round(min(0.99, max(0.01, raw)), 4)  # strictly within (0, 1)

    def reset(self) -> Dict[str, Any]:
        skills_block = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(TASK2_REQUIRED_SKILLS))
        obs = Observation(
            task_name="skill_match",
            step=0,
            job_description=TASK2_JD,
            resume=TASK2_RESUME,
            required_skills=TASK2_REQUIRED_SKILLS,
            feedback=(
                f"Required skills to evaluate:\n{skills_block}\n\n"
                "Step 1 of 3: List which of the required skills are PRESENT "
                "in the resume. Use the 'matched_skills' field."
            ),
        )
        return {"observation": obs, "internal": {"step": 0}}

    def step(self, state: Dict, action: Action) -> Dict[str, Any]:
        internal = state["internal"]
        step = internal["step"] + 1
        reward = 0.0
        done = False
        new_internal = {**internal, "step": step}

        if step == 1:
            matched = action.matched_skills or []
            f1 = self._f1(matched, self._GT["matched_skills"])
            reward = round(f1 * 0.30, 4)
            feedback = (
                f"Matched skills recorded ({len(matched)} provided, "
                f"F1={f1:.2f} vs ground truth).\n"
                "Step 2 of 3: List which required skills are MISSING from the resume. "
                "Use the 'missing_skills' field."
            )

        elif step == 2:
            missing = action.missing_skills or []
            f1 = self._f1(missing, self._GT["missing_skills"])
            reward = round(f1 * 0.30, 4)
            feedback = (
                f"Missing skills recorded ({len(missing)} provided, "
                f"F1={f1:.2f} vs ground truth).\n"
                "Step 3 of 3: Provide a final match score between 0.0 and 1.0 "
                "in the 'score' field."
            )

        elif step == 3:
            agent_score = action.score
            if agent_score is not None:
                gt = self._GT["score"]   # 0.70
                diff = abs(float(agent_score) - gt)
                if diff <= 0.05:
                    reward = 0.40
                elif diff <= 0.10:
                    reward = 0.35
                elif diff <= 0.15:
                    reward = 0.25
                elif diff <= 0.20:
                    reward = 0.15
                else:
                    reward = max(0.01, 0.40 * (1.0 - diff / 0.50))
            done = True
            feedback = (
                f"Score recorded: {action.score}. "
                f"Ground truth: {self._GT['score']}. Episode complete."
            )

        new_obs = Observation(
            task_name="skill_match",
            step=step,
            job_description=TASK2_JD,
            resume=TASK2_RESUME,
            required_skills=TASK2_REQUIRED_SKILLS,
            feedback=feedback,
            done=done,
        )
        # Clamp step reward strictly within (0, 1)
        reward = max(0.01, min(0.99, reward))
        return {
            "observation": new_obs,
            "state": {"observation": new_obs, "internal": new_internal},
            "reward": _strict(reward),
            "done": done,
            "info": {"step": step},
        }

    def grade(self, matched: List[str], missing: List[str], score: float) -> float:
        """Deterministic terminal grader."""
        f1_m = self._f1(matched, self._GT["matched_skills"])
        f1_x = self._f1(missing, self._GT["missing_skills"])
        diff = abs(score - self._GT["score"])
        score_acc = max(0.01, min(0.99, 1.0 - diff * 5))  # strictly (0,1)
        return _strict(f1_m * 0.35 + f1_x * 0.35 + score_acc * 0.30)


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 · rank_candidates  (Hard)
# ─────────────────────────────────────────────────────────────────────────────

class RankCandidatesTask:
    """
    Rank 5 Product Manager candidates (best → worst).

    Episode flow
    ─────────────
    Step 1  Assess candidates 0 & 1  → partial reward for identifying the best
    Step 2  Assess candidates 2–4    → partial reward for identifying the worst
    Step 3  Comparative analysis     → partial reward for citing B2B / SaaS signals
    Step 4  Final RANKING            → main reward = Kendall-τ correlation

    Max reward breakdown
    ────────────────────
    Step 1  correct top-2 assessment   →  up to 0.15
    Step 2  correct bottom assessment  →  up to 0.15
    Step 3  analysis quality signals   →  up to 0.20
    Step 4  Kendall-τ × 0.50           →  up to 0.50
    ────────────────────────────────────────── total 1.00
    """

    MAX_STEPS = 4
    _GT = TASK3_GROUND_TRUTH

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _kendall_tau_normalized(predicted: List[int], ground_truth: List[int]) -> float:
        """
        Normalised Kendall-τ distance in [0, 1].
        1.0 = perfect agreement, 0.0 = perfect reversal, 0.5 = random.
        Handles missing indices gracefully.
        """
        n = len(ground_truth)
        if n <= 1:
            return 0.99

        gt_pos = {idx: rank for rank, idx in enumerate(ground_truth)}
        pred_pos = {idx: rank for rank, idx in enumerate(predicted)}

        concordant = discordant = 0
        items = list(gt_pos.keys())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                gt_order = gt_pos[a] < gt_pos[b]
                try:
                    pred_order = pred_pos[a] < pred_pos[b]
                    if gt_order == pred_order:
                        concordant += 1
                    else:
                        discordant += 1
                except KeyError:
                    discordant += 1

        total = concordant + discordant
        if total == 0:
            return 0.5
        tau = (concordant - discordant) / total
        raw = (tau + 1.0) / 2.0               # map [-1,1] → [0,1]
        return round(max(0.01, min(0.99, raw)), 4)

    # ── OpenEnv methods ──────────────────────────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        obs = Observation(
            task_name="rank_candidates",
            step=0,
            job_description=TASK3_JD,
            resumes=TASK3_RESUMES,
            feedback=(
                "You have 5 candidates for the Senior Product Manager role above "
                "(index 0 to 4).\n\n"
                "Step 1 of 4: Assess CANDIDATES 0 and 1. "
                "Describe each candidate's strengths and weaknesses relative to the JD. "
                "Use the 'content' field."
            ),
        )
        return {"observation": obs, "internal": {"step": 0}}

    def step(self, state: Dict, action: Action) -> Dict[str, Any]:
        internal = state["internal"]
        step = internal["step"] + 1
        reward = 0.0
        done = False
        new_internal = {**internal, "step": step}
        text = (action.content or "").lower()

        if step == 1:
            # Does agent correctly prefer candidate 1 (Maria) over candidate 0 (Alex)?
            prefers_1 = any(s in text for s in
                            ["candidate 1", "maria", "1 is better", "1 is stronger",
                             "index 1", "candidate 2 is", "b2b saas"])
            reward = 0.15 if prefers_1 else 0.05
            feedback = (
                "Candidates 0 & 1 assessed.\n"
                "Step 2 of 4: Assess CANDIDATES 2, 3, and 4. "
                "Focus on differentiators vs the JD requirements."
            )

        elif step == 2:
            # Does agent identify candidate 2 (Tom) as weakest?
            spots_worst = any(s in text for s in
                              ["candidate 2", "tom", "most junior", "weakest", "least qualified",
                               "only 2 year", "index 2", "too junior"])
            reward = 0.15 if spots_worst else 0.05
            feedback = (
                "Candidates 2–4 assessed.\n"
                "Step 3 of 4: Provide a COMPARATIVE ANALYSIS of all 5 candidates. "
                "Explain their relative rankings in 'content'."
            )

        elif step == 3:
            # Quality signals: B2B, SaaS, experience depth
            signals = ["b2b", "saas", "experience", "senior", "junior", "years",
                       "data-driven", "launch", "mba", "cross-functional", "stakeholder"]
            count = sum(1 for s in signals if s in text)
            reward = min(0.20, round(count * 0.033, 4))
            feedback = (
                "Comparative analysis noted.\n"
                "Step 4 of 4 (FINAL): Provide your ranking of ALL 5 candidates. "
                "Use the 'rankings' field — a list of candidate indices (0–4) "
                "ordered from BEST to WORST. Example: [1, 3, 0, 4, 2]"
            )

        elif step == 4:
            rankings = action.rankings or []
            if len(rankings) >= 5:
                tau = self._kendall_tau_normalized(rankings[:5], self._GT["ranking"])
                reward = round(tau * 0.50, 4)
            else:
                reward = 0.01
            done = True
            feedback = (
                f"Ranking recorded: {rankings[:5] if rankings else 'none'}. "
                f"Ground truth: {self._GT['ranking']}. Episode complete."
            )

        new_obs = Observation(
            task_name="rank_candidates",
            step=step,
            job_description=TASK3_JD,
            resumes=TASK3_RESUMES,
            feedback=feedback,
            done=done,
        )
        # Clamp step reward strictly within (0, 1)
        reward = max(0.01, min(0.99, reward))
        return {
            "observation": new_obs,
            "state": {"observation": new_obs, "internal": new_internal},
            "reward": _strict(reward),
            "done": done,
            "info": {"step": step},
        }

    def grade(self, rankings: List[int]) -> float:
        """Deterministic terminal grader (pure Kendall-τ)."""
        return _strict(self._kendall_tau_normalized(rankings, self._GT["ranking"]))


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, Any] = {
    "binary_screen": BinaryScreenTask,
    "skill_match": SkillMatchTask,
    "rank_candidates": RankCandidatesTask,
}


def get_task(task_name: str):
    """Instantiate and return a task by name."""
    if task_name not in _REGISTRY:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[task_name]()


def list_tasks() -> List[str]:
    return list(_REGISTRY.keys())