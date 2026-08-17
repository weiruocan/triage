# TRIAGE

**Three-level Routing and Intelligent Agent Guidance for Efficient Execution**

Reduce LLM agent token consumption by reusing historical experience trajectories. Core idea: **let agents accumulate experience like humans, instead of reasoning from scratch every time**.

## Results

| Metric | Baseline ReAct | TRIAGE | Savings |
|--------|---------------|--------|---------|
| Total Tokens (237 queries) | 127,095 | 57,007 | **55.1%** |
| Direct Reuse (L1, 0 tokens) | — | 38.8% of queries | 92/237 |
| Edit Reuse (L2, partial) | — | 49.8% of queries | 118/237 |
| Cross-Domain Avg | — | 59.5% | 3 domains (ToolBench) |

## Requirements

- **Python**: 3.10+
- **LLM API**: Any OpenAI-compatible API (deepseek, OpenAI, vLLM, etc.)
- **Optional** (for semantic similarity retrieval): `sentence-transformers`

## Installation

```bash
# Option A: Install from source (recommended for now)
git clone https://github.com/weiruocan/triage.git
cd triage
pip install -e .

# Option B: Install from PyPI (coming soon)
pip install triage-agent
```

## Quick Start (5 minutes)

### Step 1: Set up your LLM API

```bash
export TRIAGE_API_KEY="sk-xxx"
export TRIAGE_BASE_URL="https://api.openai.com/v1"
export TRIAGE_MODEL="gpt-4"
```

### Step 2: Run the built-in example

```bash
# Simulated demo (no real LLM cost, shows the three-level mechanism)
python examples/quickstart.py

# Real LLM demo (end-to-end: cold start → trajectory accumulation → reuse)
python examples/end_to_end.py
```

### Step 3: Use in your own project

**Option A: Decorator (1 line, recommended)**

```python
from triage import triage_agent

@triage_agent
def my_agent(query: str) -> str:
    """Your original agent — no changes needed"""
    # Your ReAct logic here...
    return result

# Use exactly as before — TRIAGE handles caching automatically
print(my_agent("What is the online rate of sensors?"))
```

**Option B: CLI wrapper (0 code changes)**

```bash
# Run a single query through TRIAGE
triage run --query "What is the online rate of sensors?"

# Wrap your existing agent script
triage wrap --agent "python my_agent.py"

# Generate a comparison report
triage report
```

**Option C: Manual orchestration (full control)**

```python
from triage import TriageOrchestrator

orchestrator = TriageOrchestrator()
result = orchestrator.run(
    query="What is the online rate of sensors?",
    fallback=my_agent,
)
```

### Step 4: Verify it's working

```bash
# Check trajectory database status
triage report

# You should see:
# - Cold start: all queries go through full ReAct (auto-stored)
# - After 10+ queries: Level-2 edit reuse enabled
# - After 50+ queries: Level-1 direct reuse enabled
# - Token savings displayed in real-time
```

## How It Works

```
User Query
    │
    ▼
┌─────────────┐     Semantic Encoder (all-MiniLM-L6-v2)
│  LLM Router │───→ cosine similarity search
└──────┬──────┘     against trajectory DB
       │
    ┌──┴──┐
    │     │
    ▼     ▼
  L1:     L2:              L3:
  Direct  Edit             Full ReAct
  0 tokens ~200 tokens     (new query,
  (exact   (reference      auto-store
   match)  + adapt)        trajectory)

88.6% of queries benefit from reuse → 55.1% token reduction
```

### Three-Level Routing

| Level | Trigger | Token Cost | % of Queries |
|-------|---------|-----------|-------------|
| **L1 — Direct Reuse** | Cosine similarity ≥ 0.90 | **0** | 38.8% |
| **L2 — Edit Reuse** | Cosine similarity ≥ 0.65 | **~200** | 49.8% |
| **L3 — Full ReAct** | Cosine similarity < 0.65 | Full | 11.4% |

### Cold Start → Warm → Mature

| Phase | Trajectories | Behavior |
|-------|-------------|----------|
| ❄️ Cold | 0 | All → full ReAct, auto-store |
| 🔥 Warm | 10+ | L2 edit reuse enabled |
| 🏆 Mature | 50+ | L1 direct reuse enabled |

**No manual setup needed. TRIAGE learns from usage — the more you use it, the more tokens you save.**

## Core Concepts

### TaaS: Trajectory-as-a-Skill

TRIAGE's theoretical core: treat historical execution trajectories as reusable skills.

| Aspect | Skill (Function Calling) | TaaS (Trajectory-as-a-Skill) |
|--------|------------------------|-----------------------------|
| Definition | Developer pre-defines | Automatically accumulated |
| Granularity | Coarse (whole function) | Fine-grained (node-level) |
| Generalization | Weak (fixed schema) | Strong (semantic search + LLM edit) |
| Cold Start | Write all skills upfront | Auto from zero |
| Maintenance | High (new = new skill) | Low (auto-store, auto-decay) |
| Determinism | Strong | Probabilistic (LLM edit) |

### Auto Skill Extraction

High-frequency trajectory patterns (3+ occurrences) are automatically extracted into deterministic Skills:

```
Trajectories → pattern analysis → parameter extraction → Skill registration → 0-token execution
```

This creates a positive feedback loop: **the more you use it, the more efficient it becomes**.

## Project Structure

```
triage/
├── README.md                 # This file
├── AGENTS.md                 # AI behavioral guidelines (for Codex)
├── triage/                   # Core Python package
│   ├── model_api.py          # LLM API wrapper
│   ├── agent.py              # Agent orchestrator
│   ├── decorator.py          # @triage_agent decorator
│   ├── cli.py                # CLI entry point
│   ├── retrieval/            # Encoder, DB, editor, strategies
│   ├── router/               # LLM Router
│   ├── skills/               # ExecutionFlow, SkillExtractor
│   └── baseline/             # ReAct baseline
├── examples/                 # Quick start examples
├── experiments/              # Experiment scripts & data
├── docs/                     # Paper & experiment reports
└── tests/                    # Unit tests
```

## Paper

Full paper (Chinese) at `docs/paper/`. Key experimental data in `docs/experiments/`. English version will be submitted to arXiv.

```bibtex
@misc{triage2026,
  title={TRIAGE: Three-level Routing and Intelligent Agent Guidance for Efficient Execution},
  author={TRIAGE Contributors},
  year={2026},
  publisher={GitHub},
  howpublished={\\url{https://github.com/weiruocan/triage}}
}
```

## AI Assistance Declaration

This project's code and documentation were developed with the assistance of AI tools (Codex CLI). 
All core designs, algorithm selections, experimental plans, and data analysis were completed by human authors through repeated design, iteration, error correction, and refinement.
AI tools were primarily used for code implementation assistance, documentation translation, and text polishing.
The complete design records, experimental data, and iteration history are preserved in the project's documentation.

## License

MIT
