"""
inference.py — Resume Screening OpenEnv · Baseline Inference Script
=====================================================================
MANDATORY CHECKLIST COMPLIANCE
────────────────────────────────
✅ API_BASE_URL  has a default: os.getenv("API_BASE_URL", "<your-active-endpoint>")
✅ MODEL_NAME    has a default: os.getenv("MODEL_NAME",   "<your-active-model>")
✅ HF_TOKEN      has NO default: os.getenv("HF_TOKEN")   ← must be set at runtime
✅ LOCAL_IMAGE_NAME present (optional — env runs in-process, no Docker call needed)
✅ All LLM calls use OpenAI client configured via the variables above
✅ stdout logs follow START / STEP / END format exactly

STDOUT FORMAT
─────────────
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>

Rules
  • One [START] per episode, one [STEP] per env.step(), one [END] always (even on exception)
  • reward and rewards → 2 decimal places
  • score             → 2 decimal places
  • done / success    → lowercase true / false
  • error             → raw error string or null
  • All fields on a single line — no embedded newlines

Usage
─────
  export HF_TOKEN="hf_..."                                    # required
  export API_BASE_URL="https://router.huggingface.co/v1"      # optional
  export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"              # optional
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
# Environment variables  (checklist-compliant)
# ─────────────────────────────────────────────────────────────────────────────

# ✅ HF_TOKEN — NO default value (must be supplied at runtime)
HF_TOKEN = os.getenv("HF_TOKEN")

# ✅ API_BASE_URL — has a default
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")

# ✅ MODEL_NAME — has a default
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# ✅ LOCAL_IMAGE_NAME — present (optional; env runs in-process so Docker not needed)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# api_key for the OpenAI client = HF_TOKEN
API_KEY = HF_TOKEN

# ─────────────────────────────────────────────────────────────────────────────
# Internal config
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK = "resume_screening"
SUCCESS_SCORE_THRESHOLD = 0.50
TEMPERATURE = 0.1
MAX_TOKENS = 700

# Tasks ordered easy → medium → hard
TASK_CONFIGS: List[Dict[str, Any]] = [
    {"name": "binary_screen",   "max_steps": 2},
    {"name": "skill_match",     "max_steps": 3},
    {"name": "rank_candidates", "max_steps": 4},
]

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = textwrap.dedent("""
You are an expert HR recruiter and resume screening specialist.
Your job is to evaluate resumes against job descriptions.

You MUST respond with a valid JSON object — no markdown, no prose, just raw JSON.
Use this schema (include only relevant fields):

{
  "action_type": "analyze" | "screen" | "score_skills" | "rank",
  "content": "<your detailed reasoning>",
  "decision": "ACCEPT" | "REJECT" | null,
  "score": <float 0.0-1.0> | null,
  "rankings": [<int>, ...] | null,
  "matched_skills": ["<skill>", ...] | null,
  "missing_skills": ["<skill>", ...] | null
}

Field rules:
• content        — always include your detailed reasoning
• decision       — ACCEPT or REJECT ONLY on a final binary screening step
• score          — float 0.0-1.0 ONLY on a final skill-match scoring step
• rankings       — candidate indices best-to-worst ONLY on the final ranking step
• matched_skills — required skill names FOUND in the resume (skill_match step 1)
• missing_skills — required skill names ABSENT from the resume (skill_match step 2)
• Unused fields  — set to null

Respond ONLY with the JSON object. No preamble, no text outside the JSON.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Mandatory stdout log functions — format must not be changed
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    """One [START] line at episode begin."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    """One [STEP] line immediately after env.step() returns."""
    error_val = error if error else "null"
    done_val = str(done).lower()                          # lowercase true/false
    action_clean = action.replace("\n", " ").replace("\r", "").strip()[:200]
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
    """One [END] line after env.close() — always emitted, even on exception."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)  # 2 decimal places
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(obs: Observation, step: int, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    resume_block = f"\nRESUME:\n{obs.resume}" if obs.resume else ""

    candidates_block = ""
    if obs.resumes:
        parts = [f"CANDIDATE {i} (index {i}):\n{r}" for i, r in enumerate(obs.resumes)]
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
# LLM call — all calls go through OpenAI client (API_BASE_URL + HF_TOKEN)
# ─────────────────────────────────────────────────────────────────────────────

def get_model_action(
    client: OpenAI,
    obs: Observation,
    step: int,
    history: List[str],
) -> Tuple[Action, str]:
    """
    Call LLM via OpenAI client and parse JSON response into an Action.
    Returns (action, raw_text_for_logging).
    Falls back to a safe default Action on any error.
    """
    prompt = build_user_prompt(obs, step, history)
    raw_text = ""

    try:
        # ✅ All LLM calls use OpenAI client configured via API_BASE_URL / HF_TOKEN
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

        # Strip markdown code fences if model wraps response in ```json ... ```
        if "```" in raw_text:
            parts = raw_text.split("```")
            inner = parts[1] if len(parts) > 1 else parts[0]
            if inner.startswith("json"):
                inner = inner[4:]
            raw_text = inner.strip()

        data: Dict[str, Any] = json.loads(raw_text)

        # Forward only recognised Action fields
        valid_fields = set(Action.model_fields.keys())
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        action = Action(**filtered)
        return action, raw_text

    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e} | raw={raw_text[:300]}", flush=True)
        return (
            Action(action_type="analyze", content="(parse error fallback)"),
            "parse_error",
        )

    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return (
            Action(action_type="analyze", content="(api error fallback)"),
            f"error:{e}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Single task runner
# ─────────────────────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_cfg: Dict[str, Any]) -> float:
    """
    Run one task episode end-to-end.
    Emits [START] → [STEP] × n → [END] to stdout.
    Returns episode score in [0, 1].
    """
    task_name: str = task_cfg["name"]
    max_steps: int = task_cfg["max_steps"]

    env = ResumeScreeningEnv(task_name=task_name)

    rewards: List[float] = []
    steps_taken: int = 0
    history: List[str] = []
    score: float = 0.001
    success: bool = False

    # ── [START] — one per episode ─────────────────────────────────────────────
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs: Observation = env.reset()

        for step_n in range(1, max_steps + 1):
            if obs.done:
                break

            action, raw_text = get_model_action(client, obs, step_n, history)
            result = env.step(action)

            reward = result.reward
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step_n

            # ── [STEP] — one per env.step() ───────────────────────────────────
            log_step(
                step=step_n,
                action=raw_text,
                reward=reward,
                done=done,
                error=error,
            )

            history.append(f"Step {step_n} [{action.action_type}] reward={reward:.2f}")
            obs = result.observation

            if done:
                break

        # Score must be STRICTLY between 0 and 1 (exclusive) — not 0.0, not 1.0
        raw = sum(rewards)
        score = max(0.001, min(0.999, raw))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task '{task_name}' exception: {e}", flush=True)
        score = 0.001  # strictly > 0.0
        success = False

    finally:
        env.close()
        # ── [END] — always emitted, even on exception ─────────────────────────
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Main — runs all 3 tasks
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(
        f"[INFO] inference.py starting | "
        f"API_BASE_URL={API_BASE_URL} | MODEL={MODEL_NAME} | "
        f"HF_TOKEN={'SET' if HF_TOKEN else 'NOT SET'}",
        flush=True,
    )

    if not HF_TOKEN:
        print(
            "[WARNING] HF_TOKEN is not set. "
            "Export it before running:  export HF_TOKEN=hf_...",
            flush=True,
        )

    # ✅ OpenAI client — configured with API_BASE_URL and HF_TOKEN (api_key)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores: List[float] = []

    for task_cfg in TASK_CONFIGS:
        task_score = run_task(client, task_cfg)
        all_scores.append(task_score)
        print()  # blank line between tasks

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    scores_str = ",".join(f"{s:.2f}" for s in all_scores)
    print(
        f"[SUMMARY] tasks={len(all_scores)} avg_score={avg:.2f} scores={scores_str}",
        flush=True,
    )


if __name__ == "__main__":
    main()