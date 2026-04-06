"""
inference.py — Baseline inference script for Resume Screening OpenEnv.

Runs all 3 tasks sequentially using an LLM (via OpenAI-compatible API).

Required environment variables
───────────────────────────────
  HF_TOKEN        Your HuggingFace / API key
  API_BASE_URL    LLM endpoint  (default: HF Inference Router)
  MODEL_NAME      Model identifier (default: Qwen/Qwen2.5-72B-Instruct)

STDOUT format (mandatory — do not change)
─────────────────────────────────────────
  [START] task=<name> env=resume_screening model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

Usage
─────
  export HF_TOKEN=hf_...
  python inference.py
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from env import Action, Observation, ResumeScreeningEnv

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_KEY: str = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
    or "MISSING_KEY"
)
API_BASE_URL: str = os.getenv(
    "API_BASE_URL", "https://router.huggingface.co/v1"
)
MODEL_NAME: str = os.getenv(
    "MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"
)
BENCHMARK: str = "resume_screening"
SUCCESS_SCORE_THRESHOLD: float = 0.50
TEMPERATURE: float = 0.1
MAX_TOKENS: int = 700

# Tasks ordered easy → medium → hard
TASK_CONFIGS: List[Dict[str, Any]] = [
    {"name": "binary_screen",   "max_steps": 2},
    {"name": "skill_match",     "max_steps": 3},
    {"name": "rank_candidates", "max_steps": 4},
]

# ─────────────────────────────────────────────────────────────────────────────
# System prompt (shared across all tasks)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = textwrap.dedent("""
You are an expert HR recruiter and resume screening specialist.
Your job is to evaluate resumes against job descriptions.

You MUST respond with a valid JSON object — no markdown, no prose, just JSON.
Use this schema (include only relevant fields):

{
  "action_type": "analyze" | "screen" | "score_skills" | "rank",
  "content": "<your detailed reasoning here>",
  "decision": "ACCEPT" | "REJECT" | null,
  "score": <float 0.0–1.0> | null,
  "rankings": [<int>, ...] | null,
  "matched_skills": ["<skill>", ...] | null,
  "missing_skills": ["<skill>", ...] | null
}

Field rules
───────────
• "content"       Always include your detailed reasoning.
• "decision"      Set to ACCEPT or REJECT ONLY on a final screening decision step.
• "score"         Float 0.0–1.0 ONLY on a final skill-match scoring step.
• "rankings"      List of 0-based candidate indices (best→worst) ONLY for final ranking.
• "matched_skills" List of required skill names found in the resume.
• "missing_skills" List of required skill names NOT found in the resume.
• Unused fields   Set to null.

Respond ONLY with the JSON object. No preamble, no explanation outside JSON.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers  (mandatory format — do not modify)
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Keep action single-line and capped to avoid terminal clutter
    action_clean = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(
    obs: Observation,
    step: int,
    history: List[str],
) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"

    resume_block = f"\nRESUME:\n{obs.resume}" if obs.resume else ""

    candidates_block = ""
    if obs.resumes:
        parts = [
            f"CANDIDATE {i} (index {i}):\n{r}"
            for i, r in enumerate(obs.resumes)
        ]
        candidates_block = "\nCANDIDATES:\n" + "\n\n---\n".join(parts)

    skills_block = ""
    if obs.required_skills:
        skills_block = "\nREQUIRED SKILLS: " + ", ".join(obs.required_skills)

    return textwrap.dedent(f"""
    TASK: {obs.task_name}   STEP: {step}

    JOB DESCRIPTION:
    {obs.job_description}
    {resume_block}
    {candidates_block}
    {skills_block}

    ENVIRONMENT INSTRUCTION:
    {obs.feedback}

    STEP HISTORY (recent):
    {history_block}

    Respond now with your JSON action:
    """).strip()


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

def get_model_action(
    client: OpenAI,
    obs: Observation,
    step: int,
    history: List[str],
) -> Tuple[Action, str]:
    """
    Call the LLM and parse the response into an Action.

    Returns (action, raw_text).
    Falls back to a safe default Action on any error.
    """
    prompt = build_user_prompt(obs, step, history)
    raw_text = ""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw_text = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if present
        if "```" in raw_text:
            parts = raw_text.split("```")
            # Take the second segment (inside the fences)
            inner = parts[1] if len(parts) > 1 else parts[0]
            if inner.startswith("json"):
                inner = inner[4:]
            raw_text = inner.strip()

        data: Dict[str, Any] = json.loads(raw_text)

        # Only pass recognised Action fields
        valid_fields = set(Action.model_fields.keys())
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        action = Action(**filtered)
        return action, raw_text

    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e} | raw={raw_text[:300]}", flush=True)
        fallback = Action(action_type="analyze", content="(parse error fallback)")
        return fallback, raw_text or "parse_error"

    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        fallback = Action(action_type="analyze", content="(api error fallback)")
        return fallback, f"error:{e}"


# ─────────────────────────────────────────────────────────────────────────────
# Single task runner
# ─────────────────────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_cfg: Dict[str, Any]) -> float:
    """
    Run one task episode and return the episode score in [0, 1].
    Emits [START], [STEP]×n, [END] to stdout.
    """
    task_name: str = task_cfg["name"]
    max_steps: int = task_cfg["max_steps"]

    env = ResumeScreeningEnv(task_name=task_name)
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken: int = 0
    history: List[str] = []
    score: float = 0.0
    success: bool = False

    try:
        obs: Observation = env.reset()

        for step_n in range(1, max_steps + 1):
            if obs.done:
                break

            action, raw_text = get_model_action(client, obs, step_n, history)
            result = env.step(action)

            reward = result.reward
            done = result.done
            rewards.append(reward)
            steps_taken = step_n

            log_step(
                step=step_n,
                action=raw_text,
                reward=reward,
                done=done,
                error=None,
            )

            history.append(
                f"Step {step_n} [{action.action_type}] → reward={reward:.2f}"
            )
            obs = result.observation

            if done:
                break

        # Score = sum of step rewards clamped to [0, 1]
        # (each task is designed so max total reward = 1.0)
        score = min(max(sum(rewards), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task '{task_name}' raised exception: {e}", flush=True)
        score = 0.0
        success = False

    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Main — run all tasks
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(
        f"[INFO] API_BASE_URL={API_BASE_URL}  MODEL={MODEL_NAME}  "
        f"BENCHMARK={BENCHMARK}",
        flush=True,
    )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores: List[float] = []
    for task_cfg in TASK_CONFIGS:
        score = run_task(client, task_cfg)
        all_scores.append(score)
        print()  # blank line between tasks

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    scores_str = ",".join(f"{s:.2f}" for s in all_scores)
    print(
        f"[SUMMARY] tasks={len(all_scores)} avg_score={avg:.3f} "
        f"scores={scores_str}",
        flush=True,
    )


if __name__ == "__main__":
    main()
