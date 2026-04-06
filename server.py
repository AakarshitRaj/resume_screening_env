"""
server.py — FastAPI HTTP server exposing the OpenEnv interface.

Endpoints
─────────
POST /reset          Start a new episode (accepts optional task_name)
POST /step           Apply an action
GET  /state          Return current environment state
GET  /tasks          List available tasks
GET  /health         Health check
GET  /               Root info

Start locally:
  python server.py
  # or
  uvicorn server:app --host 0.0.0.0 --port 7860

HF Space: set SDK=docker, exposes port 7860.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import Action, EnvState, Observation, ResumeScreeningEnv, StepResult

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Resume Screening OpenEnv",
    description=(
        "An OpenEnv-compliant environment that simulates the real-world HR task "
        "of evaluating job applications through resume screening."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Global session (single-user HF Space)
# ─────────────────────────────────────────────────────────────────────────────

_env: Optional[ResumeScreeningEnv] = None


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    model_config = {"extra": "allow"}   # Pydantic v2 style — accepts empty {} from validator
    task_name: str = "binary_screen"


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}


class StepRequest(BaseModel):
    action: Action


class StateResponse(BaseModel):
    task_name: str
    step: int
    done: bool
    total_reward: float
    rewards: list
    available_tasks: list


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["info"])
def root() -> Dict[str, Any]:
    return {
        "name": "resume-screening-env",
        "version": "1.0.0",
        "description": "OpenEnv environment for resume screening",
        "tasks": ResumeScreeningEnv.AVAILABLE_TASKS,
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "tasks": "GET /tasks",
            "health": "GET /health",
        },
    }


@app.get("/health", tags=["info"])
def health() -> Dict[str, str]:
    return {"status": "ok", "env": "resume_screening"}


@app.get("/tasks", tags=["info"])
def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "name": "binary_screen",
                "difficulty": "easy",
                "max_steps": 2,
                "description": "Accept/Reject a single resume against a job description.",
            },
            {
                "name": "skill_match",
                "difficulty": "medium",
                "max_steps": 3,
                "description": "Identify matched/missing skills and provide a match score.",
            },
            {
                "name": "rank_candidates",
                "difficulty": "hard",
                "max_steps": 4,
                "description": "Rank 5 candidates from best to worst for a PM role.",
            },
        ]
    }


@app.post("/reset", tags=["openenv"], response_model=ResetResponse)
def reset(request: ResetRequest = None) -> ResetResponse:
    """
    Reset the environment and start a new episode.
    Accepts optional JSON body: {"task_name": "binary_screen"}.
    Empty body {} defaults to task binary_screen.
    """
    global _env

    # Handle None (no body) or empty body
    task_name = "binary_screen"
    if request is not None and hasattr(request, "task_name"):
        task_name = request.task_name or "binary_screen"

    try:
        _env = ResumeScreeningEnv(task_name=task_name)
        obs: Observation = _env.reset()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ResetResponse(
        observation=obs.model_dump(),
        reward=0.0,
        done=False,
        info={"task": task_name, "available_tasks": ResumeScreeningEnv.AVAILABLE_TASKS},
    )


@app.post("/step", tags=["openenv"])
def step(request: StepRequest) -> Dict[str, Any]:
    """Apply an action and advance the environment."""
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

    try:
        result: StepResult = _env.step(request.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.get("/state", tags=["openenv"])
def state() -> Dict[str, Any]:
    """Return the current environment state (not the observation)."""
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return _env.state().model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
    )
