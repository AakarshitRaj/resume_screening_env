---

title: Resume Screening Env
emoji: 🏢
colorFrom: red
colorTo: red
sdk: docker
pinned: false
---
# Resume Screening OpenEnv

An **OpenEnv-compliant** environment that simulates the real-world HR task of evaluating job applications through resume screening.

> **Real-world task**: Human recruiters spend hours reviewing resumes against job descriptions. This environment trains agents to do the same — binary screening, skill gap analysis, and multi-candidate ranking.

---

## 🚀 Environment Overview

Resume screening is one of the most time-consuming tasks in hiring. This environment provides realistic job descriptions and resumes, requiring an agent to make structured hiring decisions across multiple stages.

Each episode includes:

* A job description with strict requirements
* One or more candidate resumes
* Step-by-step evaluation workflow

The reward system is **dense and step-based**, encouraging reasoning quality at each stage.

---

## 📁 Project Structure

```
resume_screening_env/
├── env.py           # Core environment logic
├── tasks.py         # Task definitions
├── data.py          # Resume & job description dataset
├── server.py        # FastAPI server (OpenEnv endpoints)
├── inference.py     # LLM inference script
├── openenv.yaml     # OpenEnv metadata
├── Dockerfile       # Container config (HF Spaces)
├── requirements.txt
└── README.md
```

---

## 🧠 Tasks

### 1. binary_screen (Easy)

* Evaluate if candidate meets requirements
* Final decision: **ACCEPT / REJECT**

### 2. skill_match (Medium)

* Identify:

  * Matched skills
  * Missing skills
* Provide score (0.0 – 1.0)

### 3. rank_candidates (Hard)

* Rank multiple candidates
* Output ordered list (best → worst)

---

## 📊 Reward System

| Step Type   | Description                      |
| ----------- | -------------------------------- |
| Early Steps | Reward reasoning quality         |
| Final Step  | Reward correctness               |
| Penalties   | Incorrect decisions → low reward |

---

## ⚙️ Running Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run server

```bash
python server.py
```

Server runs at:

```
http://localhost:7860
```

---

## 🐳 Run with Docker

### Build image

```bash
docker build -t resume-screening-env .
```

### Run container

```bash
docker run -p 7860:7860 resume-screening-env
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description       |
| ------ | -------- | ----------------- |
| GET    | /health  | Health check      |
| GET    | /tasks   | List tasks        |
| POST   | /reset   | Start new episode |
| POST   | /step    | Execute step      |
| GET    | /state   | Get current state |

---

## 🧪 Example API Usage

### Reset environment

```bash
curl -X POST http://localhost:7860/reset \
-H "Content-Type: application/json"
```

### Run step

```bash
curl -X POST http://localhost:7860/step \
-H "Content-Type: application/json" \
-d "{\"task\":\"binary_screen\",\"input\":\"Candidate lacks required experience\"}"
```

---

## ☁️ Deploy on Hugging Face Spaces

1. Create a Space with **SDK = Docker**
2. Push this repository
3. Wait for automatic build

Your app will be live at:

```
https://<username>-resume-screening-env.hf.space
```

---

## 🔐 Environment Variables

| Variable     | Description                 |
| ------------ | --------------------------- |
| HF_TOKEN     | Hugging Face API token      |
| API_BASE_URL | LLM API endpoint            |
| MODEL_NAME   | Model identifier            |
| PORT         | Server port (default: 7860) |

---

## 🧩 Use Programmatically

```python
from env import ResumeScreeningEnv

env = ResumeScreeningEnv(task_name="binary_screen")
obs = env.reset()

result = env.step({
    "task": "binary_screen",
    "input": "Candidate has insufficient experience"
})

print(result)
```

---

## 📌 Key Features

* Multi-step evaluation pipeline
* Real-world HR simulation
* Dense reward feedback
* OpenEnv compatible
* Docker-ready deployment

---

## 📜 License

MIT License
