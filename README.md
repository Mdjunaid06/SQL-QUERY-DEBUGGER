# SQL Query Debugger — OpenEnv Environment

> **META × PyTorch × SST OpenEnv Hackathon** | Round 1 | March 28 – April 5, 2025

An OpenEnv-compliant reinforcement learning environment where AI agents learn to debug SQL queries across three difficulty levels: syntax errors, logic bugs, and performance issues.

---

## Motivation

SQL is the most widely used data language in the world. Every software engineer, data scientist, and analyst writes SQL daily. Yet debugging SQL queries remains a frustrating, time-consuming task — a developer staring at a wrong JOIN or a missing index can lose hours of productive work.

Despite this, no OpenEnv environment exists for SQL debugging. Existing RL benchmarks focus on code generation, not debugging. This environment fills that gap by training agents to diagnose and fix real SQL problems that real engineers face every day — from simple syntax errors to complex N+1 performance anti-patterns that silently destroy application performance at scale.

## Why This Domain?

SQL debugging is uniquely well-suited for RL evaluation:

1. **Deterministic grading** — a fixed query either matches expected output or it doesn't. No ambiguity, no LLM-based scoring.
2. **Natural difficulty curve** — syntax errors (easy) → logic bugs (medium) → performance anti-patterns (hard) map perfectly to agent skill levels.
3. **Real business value** — companies lose millions in engineering hours and infrastructure costs to slow or incorrect SQL. An agent that debugs SQL has immediate commercial value.
4. **Gap in ecosystem** — no OpenEnv environment for SQL debugging exists. This is genuinely novel.

---

## Environment Overview

| Property | Value |
|---|---|
| Domain | SQL Query Debugging |
| Tasks | 15 (5 easy, 5 medium, 5 hard) |
| Max Steps | 20 per episode |
| Reward Type | Dense (-1.0 to 1.0) |
| Grader Type | Deterministic (programmatic) |
| API Port | 7860 |

---

## Action Space

Agents can take 6 action types:

| Action | Description | Reward Signal |
|---|---|---|
| `identify_error` | Identify error location and type | +0.15 step reward + partial grader |
| `propose_fix` | Propose a fix without committing | +0.25 step reward + 40% grader score |
| `submit_answer` | Submit final fixed query | Full grader score |
| `request_hint` | Request a progressive hint | -0.05 penalty |
| `explain_issue` | Explain the issue in detail | +0.10 step reward |
| `optimize_query` | Submit optimized query (hard tasks) | +0.20 step reward + full grader score |

---

## Observation Space

Every observation contains:
```json
{
  "task_id": "easy_001",
  "task_description": "Fix the SQL syntax error: missing comma in SELECT clause",
  "current_context": {
    "buggy_query": "SELECT id name email FROM users WHERE active = 1",
    "error_message": "ERROR: syntax error at or near 'name'",
    "database_schema": {"users": ["id INT", "name VARCHAR", "email VARCHAR"]},
    "error_type_hint": "syntax",
    "steps_remaining": 20
  },
  "step_count": 0,
  "difficulty": "easy",
  "max_steps": 20,
  "hints_used": 0,
  "previous_actions": []
}
```

**Critical:** Ground truth (fixed query) is never included in the observation.

---

## Task Descriptions

### Easy — Syntax Errors
| ID | Description |
|---|---|
| easy_001 | Missing commas in SELECT clause |
| easy_002 | Missing WHERE keyword |
| easy_003 | Unclosed string literal |
| easy_004 | ORDER instead of ORDER BY |
| easy_005 | GROUP instead of GROUP BY |

### Medium — Logic Bugs
| ID | Description |
|---|---|
| medium_001 | INNER JOIN excludes users with zero orders — should be LEFT JOIN |
| medium_002 | Wrong JOIN condition causing incorrect product associations |
| medium_003 | Aggregate function in WHERE instead of HAVING |
| medium_004 | Correlated subquery correlating on wrong column |
| medium_005 | COUNT(DISTINCT *) — invalid DISTINCT usage |

### Hard — Performance Issues
| ID | Description |
|---|---|
| hard_001 | N+1 correlated subqueries in SELECT — O(n) DB hits |
| hard_002 | Function on indexed column prevents index usage |
| hard_003 | Implicit cartesian product — missing JOIN condition |
| hard_004 | SELECT * across 3-table JOIN causing over-fetching |
| hard_005 | Window function in WHERE clause + missing PARTITION BY |

---

## Reward Design

Reward is **dense** — the agent receives signal at every step, not just at the end.
```
Step 1: identify_error correctly    → +0.15 (step) + 0.03 (partial grader)
Step 2: propose_fix with good query → +0.25 (step) + 0.36 (40% grader)
Step 3: submit_answer perfectly     → +0.90 (full grader) + 0.10 (efficiency bonus)

Hint requested                      → -0.05 (penalty)
Same action 3x in a row             → -0.05 per repeat (loop penalty)
Null / invalid action               → -0.10 (penalty)
Max steps reached                   → -0.10 (penalty)
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check — always 200 |
| `/reset` | POST | Start new episode → Observation |
| `/step` | POST | Submit action → (obs, reward, done, info) |
| `/state` | GET | Current episode state |
| `/tasks` | GET | All 15 tasks + action schema |
| `/grader` | POST | Grade an episode → float score |
| `/baseline` | POST | Run baseline agent → scores JSON |

---

## Setup & Installation

### Requirements
- Python 3.11+
- Docker Desktop

### Local Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/sql-query-debugger
cd sql-query-debugger

# Install dependencies
pip install -r requirements.txt

# Set environment variable
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run the server
uvicorn api.server:app --host 0.0.0.0 --port 7860 --reload
```

### Docker Setup
```bash
# Build
docker build -t sql-query-debugger .

# Run
docker run -p 7860:7860 -e OPENAI_API_KEY=your-key sql-query-debugger
```

### Verify
```bash
curl http://localhost:7860/health
# {"status":"ok","version":"1.0.0"}

curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'
# Returns initial Observation

curl http://localhost:7860/tasks
# Returns all 15 tasks with action schema
```

---

## Baseline Scores

The rule-based baseline agent scores:

| Difficulty | Task | Score | Steps |
|---|---|---|---|
| Easy | easy_001 | 0.80 | 2 |
| Medium | medium_001 | 0.98 | 2 |
| Hard | hard_001 | 0.80 | 2 |
| **Average** | | **0.86** | **2** |

Baseline uses heuristic rules — no LLM calls. A trained RL agent is expected to significantly outperform this baseline on hard tasks.

---

## Project Structure
```
sql-query-debugger/
├── openenv.yaml          # OpenEnv metadata
├── Dockerfile            # Container definition
├── requirements.txt      # Pinned dependencies
├── README.md             # This file
├── baseline.py           # Baseline inference script
├── .env.example          # Environment variable template
├── env/
│   ├── environment.py    # Core: step() reset() state()
│   ├── models.py         # Pydantic models
│   ├── tasks.py          # Task definitions + manager
│   ├── graders.py        # Deterministic graders
│   └── reward.py         # Dense reward shaping
├── api/
│   └── server.py         # FastAPI — all 7 endpoints
├── dataset/
│   ├── easy_cases.json   # 5 syntax error tasks
│   ├── medium_cases.json # 5 logic bug tasks
│   └── hard_cases.json   # 5 performance tasks
└── tests/
    ├── test_environment.py
    └── test_graders.py
```

---

## Built For

**META × PyTorch × SST OpenEnv Hackathon**
Round 1: March 28 – April 5, 2025 | $30,000 Prize Pool

*Build something you would be proud to show to a senior engineer at Meta.*