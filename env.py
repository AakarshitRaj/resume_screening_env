"""
env.py — Core OpenEnv environment: typed Pydantic models + ResumeScreeningEnv class.

OpenEnv Interface implemented:
  reset()  → Observation
  step()   → StepResult(observation, reward, done, info)
  state()  → EnvState
  close()  → None
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Typed Models (OpenEnv spec)
# ─────────────────────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """What the agent observes at each step."""
    task_name: str
    step: int
    job_description: str
    resume: Optional[str] = None           # single resume (tasks 1 & 2)
    resumes: Optional[List[str]] = None    # multiple resumes (task 3)
    required_skills: Optional[List[str]] = None  # task 2
    feedback: str = ""                     # env instruction / result of last action
    done: bool = False


class Action(BaseModel):
    """What the agent sends at each step."""
    action_type: str = Field(
        default="analyze",
        description="One of: analyze | screen | score_skills | rank",
    )
    content: str = Field(default="", description="Free-text reasoning / analysis")
    decision: Optional[str] = Field(
        default=None,
        description="ACCEPT or REJECT (task 1 final step)",
    )
    score: Optional[float] = Field(
        default=None,
        description="Match score 0.0–1.0 (task 2 final step)",
    )
    rankings: Optional[List[int]] = Field(
        default=None,
        description="Candidate indices ordered best→worst (task 3 final step)",
    )
    matched_skills: Optional[List[str]] = Field(
        default=None,
        description="Skills present in resume (task 2 step 1)",
    )
    missing_skills: Optional[List[str]] = Field(
        default=None,
        description="Skills absent from resume (task 2 step 2)",
    )


class Reward(BaseModel):
    """Reward model (OpenEnv typed reward)."""
    value: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float] = {}
    reason: str = ""


class StepResult(BaseModel):
    """Return value of env.step()."""
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class EnvState(BaseModel):
    """Return value of env.state()."""
    task_name: str
    step: int
    done: bool
    total_reward: float
    rewards: List[float]
    available_tasks: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# ResumeScreeningEnv
# ─────────────────────────────────────────────────────────────────────────────

class ResumeScreeningEnv:
    """
    Resume Screening Environment.

    Simulates the real-world HR task of evaluating job applications.

    Tasks
    -----
    binary_screen   (easy)    Accept/Reject a single resume.
    skill_match     (medium)  Identify matched/missing skills and score a resume.
    rank_candidates (hard)    Rank 5 candidates from best to worst.

    Usage
    -----
    env = ResumeScreeningEnv(task_name="binary_screen")
    obs = env.reset()
    result = env.step(Action(action_type="analyze", content="..."))
    s = env.state()
    env.close()
    """

    AVAILABLE_TASKS: List[str] = ["binary_screen", "skill_match", "rank_candidates"]

    def __init__(self, task_name: str = "binary_screen") -> None:
        if task_name not in self.AVAILABLE_TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Available: {self.AVAILABLE_TASKS}"
            )
        self.task_name = task_name
        self._task = None
        self._internal_state: Dict[str, Any] = {}
        self._step_count: int = 0
        self._rewards: List[float] = []
        self._done: bool = False

    # ── private ──────────────────────────────────────────────────────────────

    def _load_task(self) -> None:
        from tasks import get_task  # deferred to avoid circular imports
        self._task = get_task(self.task_name)

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self._load_task()
        raw = self._task.reset()
        self._internal_state = raw["internal"]
        self._step_count = 0
        self._rewards = []
        self._done = False
        return raw["observation"]

    def step(self, action: Action) -> StepResult:
        """
        Apply action and advance the environment by one step.

        Parameters
        ----------
        action : Action

        Returns
        -------
        StepResult(observation, reward, done, info)
        """
        if self._task is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode finished. Call reset() to start a new episode.")

        state_bundle = {
            "observation": None,   # not needed by task; internal carries truth
            "internal": self._internal_state,
        }
        result = self._task.step(state_bundle, action)

        # Update env state
        self._internal_state = result["state"]["internal"]
        reward = float(result["reward"])
        self._rewards.append(reward)
        self._step_count += 1
        self._done = bool(result["done"])

        return StepResult(
            observation=result["observation"],
            reward=reward,
            done=self._done,
            info=result.get("info", {}),
        )

    def state(self) -> EnvState:
        """Return the current environment state (not the observation)."""
        return EnvState(
            task_name=self.task_name,
            step=self._step_count,
            done=self._done,
            total_reward=round(sum(self._rewards), 4),
            rewards=list(self._rewards),
            available_tasks=self.AVAILABLE_TASKS,
        )

    def close(self) -> None:
        """Release any resources (no-op for direct Python mode)."""
        self._task = None
