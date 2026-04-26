---
title: SQL Database Engineer Agent
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - sql
  - database
  - engineering
  - long-horizon
  - self-improvement
  - wildcard
license: mit
---

# SQL Database Engineer Agent — OpenEnv Environment

> **META × PyTorch × SST OpenEnv Hackathon** | Finals April 25–26, 2026 | Bangalore
> Evolved from SQL Query Debugger (Round 1 — all 4 checks passed ✅)

An OpenEnv-compliant reinforcement learning environment where AI agents learn to act like **senior database engineers**. The agent manages a simulated production database over 50+ steps — inspecting slow queries, creating indexes, rewriting queries, and partitioning tables.

---

## 🔗 Quick Links

| Resource | Link |
|---|---|
|  **Live Demo** | https://huggingface.co/spaces/junaid0600/sql-db-agent-demo-ui |
|  **Training Notebook** | https://huggingface.co/spaces/junaid0600/sql-db-engineer-agent/blob/main/SDEA_Training_Notebook.ipynb |
|  **Blog Post** | https://huggingface.co/spaces/junaid0600/sql-db-engineer-agent/blob/main/blog_post.md |
|  **Source Code** | [HF Space]  | https://huggingface.co/spaces/junaid0600/sql-db-engineer-agent/tree/main |
|                  | [Git Repo] | https://github.com/Mdjunaid06/sql-db-engineer-agent |
---

## From Round 1 → Round 2

| | Round 1 — SQL Query Debugger | Round 2 — SQL Database Engineer Agent |
|---|---|---|
| **Task** | Fix one broken SQL query | Optimize entire production database |
| **Steps** | 20 per episode | 50 per episode |
| **Actions** | 6 (identify, fix, submit...) | 15 (inspect, index, rewrite, partition...) |
| **Reward** | Dense per step | Dense + milestone bonuses |
| **Scenarios** | 15 single-query tasks | 30 total (15 new + 15 original) |
| **Training** | Rule-based baseline | Unsloth + GRPO on Qwen2.5-7B |
| **Theme** | Real-world SQL | Long-Horizon + World Modeling + Wildcard |

---

## Motivation

Every production database degrades over time.

Your app launches. Queries run in 50ms. Six months later, users are complaining. P95 query time: **8,500ms**. A senior DBA sits down — runs EXPLAIN queries, finds missing indexes, rewrites bad JOINs, partitions 50-million-row tables. **This takes 10 years to learn.**

We asked: **can we train an LLM to do it?**

SQL database engineering is uniquely well-suited for RL:
1. **100% measurable** — query time in milliseconds, index hit rates, performance scores
2. **Long-horizon** — real fixes require 10-50 careful, ordered steps
3. **World modeling** — agent must maintain internal model of DB state, indexes, query plans
4. **Self-improving** — curriculum generates harder scenarios as agent improves
5. **Novel** — no OpenEnv environment for DB engineering exists anywhere

---

## 📊 Training Results

Trained **Qwen2.5-7B-Instruct** with **GRPO** using **Unsloth** (only 0.53% of parameters via LoRA):

### GRPO Training Curves — 200 Steps

![Demo](assets/loss_curve_demo.png)

| Metric | Value |
|---|---|
| Training steps | 200 |
| Loss | `4.92e-07 → 1.23e-05` |
| Reward | `0.235 → 0.456` |
| Improvement | **+94%** |
| Model | Qwen2.5-7B (0.53% trainable via LoRA) |
| Epochs | 29 |
| Batch size | 8 (4 × 2 grad accum × 1 GPU) |

> ⚠️ Note: GRPO policy loss rises as the model becomes more confident — this is expected behaviour, not divergence. The reward curve confirms consistent improvement.

### Evaluation — Trained vs Random Agent (15 Scenarios)

![Demo](assets/reward_curve_demo.png)

| Agent | Avg Improvement | Best Scenario | Worst Scenario |
|---|---|---|---|
| Random (wrong index) | +0.0 pts | 0 pts | 0 pts |
| Trained (GRPO) | **+31.4 pts** | **+59 pts** (Scenario 8 ) | +10 pts |

- Trained agent outperformed random baseline on **every single scenario**
- Scenario 8 flagged as outlier (±1.5σ) — agent found especially impactful index combination
- Relative gain: **∞** (baseline scored exactly 0 on all scenarios)

### Training Progression

| Stage | Avg Reward | Agent Behavior |
|---|---|---|
| Before training | 0.05 | Random actions, no strategy |
| 50 steps | 0.25 | Learns to inspect before acting |
| 200 steps | **0.456** | Multi-step planning emerges |

---

## Environment Overview

| Property | Value |
|---|---|
| Domain | Database Engineering |
| Tasks | 30 (15 Round 2 scenarios + 15 Round 1 cases) |
| Max Steps | 50 per episode |
| Reward Type | Dense + milestone bonuses |
| Performance Score | 0–100 (real DB metric) |
| API Port | 7860 |
| Themes | Long-Horizon (2) + World Modeling (3.1) + Self-Improvement (4) + Wildcard (5) |

---

## Action Space (15 Actions)

### Round 2 — DB Engineering Actions
| Action | What It Does | Reward |
|---|---|---|
| `inspect_query` | EXPLAIN a slow query — scan type, rows examined, cost | +0.05 |
| `analyze_indexes` | Show all indexes + missing index hints | +0.05 |
| `create_index` | Add composite index on specified columns | +0.10 + delta |
| `rewrite_query` | Submit rewritten SQL — measures improvement | +0.15 + delta |
| `add_column` | Add denormalization column to reduce JOINs | +0.08 + delta |
| `drop_index` | Remove unused index (reduce write overhead) | +0.05 + delta |
| `partition_table` | Partition large table by date/ID range | +0.15 + delta |
| `analyze_statistics` | Update table statistics for query planner | +0.05 + delta |
| `request_hint` | Get progressive hint | −0.10 penalty |
| `submit_report` | **TERMINAL**: Final optimization report + full score | 0.0–1.0 |

### Round 1 — SQL Debugging Actions (backward compatible)
`identify_error` · `propose_fix` · `submit_answer` · `explain_issue` · `optimize_query` · `request_hint`

---

## Observation Space

Every observation contains the full DB state:
```json
{
  "task_id": "medium_s001",
  "task_description": "E-commerce DB: 50K orders. P95 query time > 8s. Target: < 500ms.",
  "current_context": {
    "performance_score": 12.5,
    "target_score": 75.0,
    "tables": [
      {"name": "orders", "rows": 50000, "indexes": ["PRIMARY"], "size_mb": 280},
      {"name": "users",  "rows": 8000,  "indexes": ["PRIMARY", "email_idx"]}
    ],
    "slow_queries": [
      {"id": "q1", "sql": "SELECT * FROM orders WHERE user_id=? AND status=?", "avg_ms": 8500},
      {"id": "q2", "sql": "SELECT COUNT(*) FROM orders o JOIN users u ON o.user_id=u.id", "avg_ms": 3200}
    ],
    "improvement_history": [12.5],
    "milestones_earned": [],
    "steps_remaining": 50
  },
  "step_count": 0,
  "difficulty": "medium",
  "max_steps": 50
}
```

---

## Reward Design

Dense reward at every step + milestone bonuses:

```
inspect_query / analyze_indexes  → +0.05 (investigation rewarded)
create_index with improvement    → +0.10 + delta_reward
Milestone: 25% improvement       → +0.15 ONE-TIME bonus
Milestone: 50% improvement       → +0.25 ONE-TIME bonus
Milestone: 75% improvement       → +0.40 ONE-TIME bonus
submit_report (terminal)         → 0.0–1.0 full score
Efficiency bonus (< 70% budget)  → +0.10
Loop penalty (same action x2+)   → −0.08
Hint penalty                     → −0.10
Backtrack penalty                → −0.05
Budget exhaustion                → −0.15
```

### GRPO Reward Breakdown (Expected per action)
```
inspect_query / analyze_indexes       →  ~0.10
create_index (no table/col match)     →  ~0.10
create_index (partial hint match)     →  ~0.20–0.45
create_index (perfect hint match)     →  ~0.55–0.80
create_index (simulator confirms)     →  ~0.75–0.99
Milestones: 25%=+0.15  50%=+0.25  75%=+0.40  (cumulative)
```

### Terminal Score Formula
```python
perf_improvement = (final_score - baseline) / (100 - baseline)
step_efficiency  = 1.0 - (steps_used / max_steps)
terminal_score   = (perf_improvement * 0.60) + (step_efficiency * 0.20) + 0.10
```

---

## Scenarios — 30 Tasks

### Round 2: DB Engineering (15 new tasks)

#### Easy (15 steps, target 80+)
| ID | Description |
|---|---|
| easy_s001 | User lookup — missing email index on 10K users |
| easy_s002 | Order status — composite index on 50K orders |
| easy_s003 | Product search — LIKE query on 20K products |
| easy_s004 | Session lookup — 15K sessions, no index |
| easy_s005 | Log filter — compound index on 30K logs |

#### Medium (25–30 steps, target 72–78)
| ID | Description |
|---|---|
| medium_s001 | E-commerce: 50K orders + 8K users, 2 slow queries |
| medium_s002 | Blog: 100K posts + 20K authors, search slow |
| medium_s003 | Inventory: 200K stock movements, rewrite + index |
| medium_s004 | Ticketing: 60K tickets, status queue degraded |
| medium_s005 | Analytics: 150K events, funnel query slow |

#### Hard (50 steps, target 65–70)
| ID | Description |
|---|---|
| hard_s001 | Financial: 500K transactions, 4 tables, 3 slow queries |
| hard_s002 | SaaS: 8-table schema, 2M activity log, dashboard 20s+ |
| hard_s003 | Healthcare: 1M patient records, compliance queries |
| hard_s004 | Gaming: 2M players, 5M matches, leaderboard degraded |
| hard_s005 | Logistics: 6 tables, 3M shipments + 10M tracking rows |

### Round 1: SQL Debugging (15 original tasks — backward compatible)
Easy: syntax errors · Medium: logic bugs · Hard: performance anti-patterns

---

## Self-Improving Curriculum

```
Agent avg score > 0.75  →  Advance to harder tier
Agent avg score < 0.30  →  Drop back a tier
Ultra tier (tier 3)     →  Auto-generated 5-8 table scenarios, no hints
```

The environment gets harder as the agent gets smarter. **Genuine adaptive curriculum.**

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check — always 200 |
| `/reset` | POST | Start new episode → Observation |
| `/step` | POST | Submit action → (obs, reward, done, info) |
| `/state` | GET | Current episode state |
| `/tasks` | GET | All 30 tasks + action schema |
| `/grader` | POST | Grade an episode → float score |
| `/baseline` | POST | Run baseline agent → scores |
| `/progress` | GET | DB performance history + milestones |
| `/trained-agent` | POST | Run trained GRPO agent → steps + score |
| `/trained-agent/status` | GET | Check if trained model is loaded |
| `/demo` | GET | Live HTML demo UI |

---

## Live Demo

```bash
# Reset with e-commerce scenario
curl -X POST https://junaid0600-sql-db-engineer-agent.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy", "task_id": "easy_s001"}'

# Agent inspects slow query → sees FULL TABLE SCAN
curl -X POST https://junaid0600-sql-db-engineer-agent.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "inspect_query", "payload": {"query_id": "q1"}}'

# Agent creates index → performance score 8.0 → 82.0
curl -X POST https://junaid0600-sql-db-engineer-agent.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "create_index", "payload": {"table": "users", "columns": ["email"]}}'

# Agent submits report → terminal score 0.82
curl -X POST https://junaid0600-sql-db-engineer-agent.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "submit_report", "payload": {"summary": "Added email index. Performance 8 to 82."}}'
```

---

## Project Structure

```
sql-query-debugger/
├── .env                          # Environment variables
├── .env.example                  # Environment variables template
├── .gitignore
├── Dockerfile                    # Container definition
├── README.md                     # This file
├── blog_post.md                  # HF blog post (separate from README)
├── loss_curve.png                # GRPO training curves ✅ evidence
├── reward_curve.png              # Evaluation results ✅ evidence
├── openenv.yaml                  # OpenEnv metadata (v2.0.0)
├── pyproject.toml
├── requirements.txt              # Pinned dependencies
├── uv.lock
├── baseline.py                   # Rule-based baseline agent
├── demo_app.py                   # Gradio demo app
├── inference.py                  # LLM inference agent
│
├── api/
│   ├── __init__.py
│   └── server.py                 # FastAPI — 11 endpoints
│
├── dataset/
│   ├── easy_cases.json           # Round 1: easy SQL tasks
│   ├── easy_scenarios.json       # Round 2: easy DB scenarios
│   ├── hard_cases.json           # Round 1: hard SQL tasks
│   ├── hard_scenarios.json       # Round 2: hard DB scenarios
│   ├── medium_cases.json         # Round 1: medium SQL tasks
│   └── medium_scenarios.json     # Round 2: medium DB scenarios
│
├── env/
│   ├── __init__.py
│   ├── scenarios/                # Scenario definitions
│   ├── curriculum.py             # Self-improving curriculum
│   ├── db_simulator.py           # DB performance simulator
│   ├── environment.py            # Core: reset() step() state()
│   ├── graders.py                # Deterministic graders
│   ├── models.py                 # Pydantic models (15 action types)
│   ├── reward.py                 # Dense reward + milestones
│   ├── scenario_generator.py     # Dynamic scenario generation
│   └── tasks.py                  # Task manager (30 tasks)
│
├── sdea-trained/
│   └── eval_results.json         # Evaluation results JSON
│
├── training/
│   ├── colab_notebook.py         # Colab training notebook
│   ├── evaluate_agent.py         # Evaluation + reward curve generator
│   ├── generate_plots.py         # Fixed plot generator
│   ├── generate_training_data.py # Expert trajectory collector
│   └── train_agent.py            # Unsloth + GRPO training script
│
└── tests/
    ├── __init__.py
    ├── test_environment.py       # Environment tests
    ├── test_graders.py           # Grader tests
    ├── test_reward.py            # Reward tests
    └── test_tasks.py             # Task tests
```

---

## Setup & Installation

```bash
# Clone
git clone https://github.com/Mdjunaid06/sql-db-engineer-agent
cd sql-db-engineer-agent

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add HF_TOKEN to .env

# Run
uvicorn api.server:app --host 0.0.0.0 --port 7860 --reload

# Verify
curl http://localhost:7860/health
# {"status":"ok","version":"2.0.0"}

# Open demo
# http://localhost:7860/demo
```

---

## Validation

```bash
pytest tests/ -v          # 24/24 passed
openenv validate .         # [OK] Ready for multi-mode deployment
```

---

## Built For

**META × PyTorch × SST OpenEnv Hackathon**
Finals: April 25–26, 2026 | Bangalore

*"We didn't build an environment. We built a DBA training simulator."*