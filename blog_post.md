# Training a SQL Database Engineer Agent with GRPO on Qwen2.5

*Fine-tuning a language model to autonomously diagnose and fix slow database queries using Reinforcement Learning*

---

##  Overview

Modern applications live and die by their database performance. Slow queries cause timeouts, poor user experience, and infrastructure costs — yet diagnosing and fixing them requires deep expertise. What if a language model could learn to do this autonomously?

In this project, we trained **Qwen2.5-7B-Instruct** to act as a senior database engineer — inspecting slow queries, identifying missing indexes, and applying targeted fixes — using **Group Relative Policy Optimization (GRPO)**, a reinforcement learning algorithm that teaches the model through reward signals rather than labeled examples.

After **200 training steps**, the agent achieved a **+94% reward improvement** (0.235 → 0.456) and outperformed a random baseline by an average of **+31.4 database performance points** across 15 scenarios.

---

##  The Problem

Given a database with slow-running SQL queries, the agent must:
1. **Investigate** — understand why queries are slow
2. **Diagnose** — identify missing indexes or inefficient query patterns
3. **Fix** — apply the correct indexes and optimizations
4. **Verify** — confirm the performance score improved

A random agent that creates indexes on arbitrary columns scores **0 pts** on every scenario. Our trained agent had to learn — purely from feedback — which tables and columns actually matter.

---

##  Architecture

### Environment — DatabaseSimulator
We built a custom `DatabaseSimulator` that:
- Loads SQL scenarios (tables, slow queries, missing index hints)
- Tracks a **performance score (0–100)** that improves when correct indexes are applied
- Returns delta rewards based on how much the score improved
- Runs **locally** — no HTTP calls, no shared state, fully deterministic

### Scenarios
We created **15 scenarios** across 3 difficulty levels:

| Level | Count | Description |
|-------|-------|-------------|
| Easy | 5 | Single table, one missing index |
| Medium | 5 | E-commerce DB, composite indexes |
| Hard | 5 | 4-table financial schema, complex joins |

### Action Space
The agent can take 10 actions:

```json
{"action_type": "inspect_query",     "payload": {"query_id": "q1"}}
{"action_type": "analyze_indexes",   "payload": {}}
{"action_type": "create_index",      "payload": {"table": "orders", "columns": ["user_id", "status"]}}
{"action_type": "rewrite_query",     "payload": {"query_id": "q1", "new_sql": "..."}}
{"action_type": "analyze_statistics","payload": {"table": "orders"}}
{"action_type": "submit_report",     "payload": {"summary": "..."}}
```

---

##  Training Setup

### Model
- **Base model:** `unsloth/Qwen2.5-7B-Instruct` (7.66B parameters)
- **Trainable parameters:** 40,370,176 of 7,655,986,688 **(only 0.53% via LoRA)**
- **Fine-tuning:** LoRA (r=16, alpha=16) via Unsloth — 2x faster free finetuning
- **Training algorithm:** GRPO (Group Relative Policy Optimization)
- **Framework:** TRL + Unsloth + PyTorch
- **GPU:** Single GPU (1x)

### Training Data
- **Examples:** 15 scenarios
- **Epochs:** 29 (cycling through all 15 scenarios)
- **Total steps:** 200
- **Effective batch size:** 8 (batch size 4 × gradient accumulation 2 × 1 GPU)

### GRPO Reward Function
The reward function combines three signals:

```python
total_reward = step_reward + delta_reward + milestone_bonus
```

| Component | Description | Range |
|-----------|-------------|-------|
| `step_reward` | Base reward per valid action type | 0.05–0.20 |
| `delta_reward` | Proportional to DB performance improvement | 0.0–0.65 |
| `milestone_bonus` | Bonus at 25%, 50%, 75% improvement thresholds | 0.15–0.40 |
| `wrong_index_penalty` | Penalty for indexing useless columns | -0.05 |

**Expected rewards per action:**
```
inspect_query / analyze_indexes       →  ~0.10
create_index (no table/col match)     →  ~0.10
create_index (partial hint match)     →  ~0.20–0.45
create_index (perfect hint match)     →  ~0.55–0.80
create_index (simulator confirms)     →  ~0.75–0.99
Milestones: 25%=+0.15  50%=+0.25  75%=+0.40  (cumulative)
```

**Key design decision:** We used a **hint-match fallback** to give GRPO a gradient signal early in training — before the model has learned exact column names, partial column matches still receive proportional rewards. This prevented the cold-start problem where the model gets 0 reward for everything and never improves.

### Training Config
```python
GRPOConfig(
    max_steps                   = 200,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 2,
    learning_rate               = 2e-5,
    max_completion_length       = 150,
    num_generations             = 4,
    temperature                 = 1.0,
    warmup_steps                = 10,
)
```

---

##  Results

### Training Curves

After 200 steps of GRPO training:

- **Loss:** `4.92e-07 → 1.23e-05`
  *(GRPO policy loss rises as the model becomes more confident in its policy — this is expected behaviour in GRPO, not divergence. The 10-step rolling average confirms stable learning without collapse.)*
- **Reward:** `0.235 → 0.456 (+94% improvement)`
  The reward shows a strong and consistent upward trend from ~0.20 to ~0.45, with the 10-step rolling average clearly confirming the model improved throughout training.

### Evaluation — Trained vs Random Agent

We evaluated both agents across all 15 scenarios:

| Agent | Avg Improvement | Best Scenario | Worst Scenario |
|-------|----------------|---------------|----------------|
| Random (wrong index) | +0.0 pts | 0 pts | 0 pts |
| Trained (GRPO) | +31.4 pts | +59 pts (Scenario 8 ) | +10 pts |

The trained agent outperformed the random baseline on **every single scenario**, with an average improvement of **+31.4 database performance points**. Scenario 8 was flagged as a statistical outlier (±1.5σ above mean) — the agent found an especially impactful index combination. The relative gain is **∞** since the untrained baseline scored exactly 0 on all scenarios.

---

##  Key Learnings

### 1. Reward shaping is everything in GRPO
The model started producing low-reward outputs for the first ~10 steps until the hint-match fallback kicked in. Without partial credit for close-but-not-perfect column names, training would have stalled completely.

### 2. LoRA makes 7B models trainable on a single GPU
With only **0.53% of parameters trainable** via LoRA, we fine-tuned a full 7B model on a single GPU in under 2 hours. Without LoRA this would require multiple A100s.

### 3. Local simulation beats API calls for training
Using `DatabaseSimulator` directly (instead of calling a REST API) made rewards deterministic, removed shared state bugs, and made training 10x faster with no network latency.

### 4. GRPO loss behaviour differs from supervised loss
Unlike cross-entropy loss in supervised fine-tuning, GRPO policy loss can increase as the model becomes more confident in its policy. This is normal and does not indicate a problem — what matters is whether the reward is trending upward.

### 5. Composite indexes are hard to learn
The model consistently struggled with scenarios requiring composite indexes on 3+ columns. Single-column indexes were learned quickly (by step ~20), while multi-column patterns took much longer to emerge.

---

##  Live Demo

Try the agent yourself — pick a scenario difficulty, choose between the trained GRPO agent and the rule-based baseline, and watch it diagnose and fix the database in real time:

 **[SQL Database Engineer Agent — Live Demo](https://huggingface.co/spaces/YOUR_USERNAME/sql-db-engineer-demo)**

---

##  Resources

| Resource | Link |
|----------|------|
| Demo Space  |https://huggingface.co/spaces/junaid0600/sql-db-agent-demo-ui |
|             |
| Source code | GitHub Repo - https://github.com/Mdjunaid06/sql-db-engineer-agent |
|             | HF Repo - https://huggingface.co/spaces/junaid0600/sql-db-engineer-agent/tree/main |
|             |
|Training Run Notebook URL| https://huggingface.co/spaces/junaid0600/sql-db-engineer-agent/blob/main/SDEA_Training_Notebook.ipynb 

---

##  What's Next

- **More steps:** 200 steps showed strong learning — 500+ steps would likely push the average score above 50 pts
- **Harder scenarios:** 8-table schemas with nested subqueries and CTEs
- **Query rewriting:** The agent currently focuses on indexing — teaching it to rewrite SQL itself is the next frontier
- **Multi-step episodes:** Chain multiple actions per episode so the agent can inspect → diagnose → fix → verify in sequence

---

##  Acknowledgements

Built for the **META × PyTorch × SST Hackathon** using:
- [Unsloth](https://github.com/unslothai/unsloth) — 2x faster LoRA fine-tuning
- [TRL](https://github.com/huggingface/trl) — GRPO implementation
- [Hugging Face](https://huggingface.co) — model hosting and Spaces
- [Qwen2.5](https://huggingface.co/Qwen) — base language model

