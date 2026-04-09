# CLAUDE.md — trader-personas

## What This Project Is

A barebones, well-documented implementation of persona-conditioned small language model (SLM)
agents for financial market simulation. Three trader personas are fine-tuned via LoRA on top of
a shared SLM base. They are deployed as heterogeneous agents in a minimal order-driven market.
The goal is to show that micro-level behavioral heterogeneity produces macro-level stylized facts
(fat tails, volatility clustering) that homogeneous populations do not.

This is an academic demonstration project built to accompany a talk at the University of Chicago
Financial Mathematics Program. Code clarity and documentation are as important as correctness.
Every file should be readable by a PhD student in an afternoon.

---

## Talk Section Mapping

Each source file should include a header comment indicating which section of the talk it
corresponds to. The sections are:

- **Section II** — Background: SLM fine-tuning for personas + LLM-based market simulation
- **Section III** — The Proposed Framework (data generation → LoRA fine-tuning → simulation → eval)
- **Section IV** — Open Problems (context drift, persona identifiability, equilibrium/stability)

Notebooks are the primary reading experience. `.py` files are clean implementations that
notebooks call into. README.md should be readable as a standalone document.

---

## Project Structure

```
trader-personas/
│
├── CLAUDE.md                        # this file
├── README.md                        # conceptual overview, maps to talk sections
├── requirements.txt
│
├── data/
│   ├── generate_personas.py         # [Section III] prompts an LLM to generate reasoning traces
│   └── personas/
│       ├── momentum.jsonl           # ~300 examples
│       ├── value.jsonl
│       └── noise.jsonl
│
├── training/
│   ├── finetune.py                  # [Section III] LoRA fine-tuning with PEFT + TRL SFTTrainer
│   └── configs/
│       ├── momentum.yaml
│       ├── value.yaml
│       └── noise.yaml
│
├── simulation/
│   ├── market.py                    # [Section III] minimal pure-Python order book
│   ├── agent.py                     # [Section III] wraps a fine-tuned SLM adapter as a trading agent
│   └── run_simulation.py            # orchestrates N agents, logs price series and order flow
│
├── eval/
│   ├── stylized_facts.py            # [Section III/IV] fat tails, volatility clustering checks
│   └── persona_drift.py             # [Section IV] tracks behavioral consistency over time
│
├── notebooks/
│   ├── 01_data_generation.ipynb     # walk through persona prompt design and data generation
│   ├── 02_finetuning_walkthrough.ipynb  # step-by-step LoRA fine-tuning for one persona
│   └── 03_simulation_and_eval.ipynb # run the hero experiment, plot results
│
└── app/
    ├── streamlit_app.py             # main entry point: `streamlit run app/streamlit_app.py`
    └── pages/
        ├── 01_persona_inspector.py  # interactive persona behavior explorer
        ├── 02_run_simulation.py     # configure and run a live simulation
        └── 03_hero_experiment.py    # pre-computed hero experiment results + stylized facts
```

---

## Stack

| Component | Library / Tool | Notes |
|---|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` | Small enough for CPU inference (quantized); fine-tunable on a single A100 |
| Fine-tuning | `peft` (LoRA) + `trl` (SFTTrainer) + `transformers` | Standard HuggingFace stack |
| Data generation | Anthropic API (`claude-sonnet-4-6`) | Called in `generate_personas.py`; requires `ANTHROPIC_API_KEY` env var |
| Market simulation | Pure Python | No external deps; clarity over realism |
| Evaluation | `numpy`, `scipy`, `matplotlib` | Stylized facts, return distribution plots |
| Notebooks | `jupyter` | Step-by-step reading/walkthrough interface |
| Demo app | `streamlit` | Interactive visualization and live simulation |

**Constraint**: The simulation and eval must always run end-to-end without a GPU. They do
this through automatic fallback: `simulation/run_simulation.py:create_agents` uses
`SLMAgent` (fine-tuned LoRA on `Qwen/Qwen2.5-1.5B-Instruct`) when adapters are present
under `<project_root>/adapters/<persona>/`, and falls back to `RuleBasedAgent` (a
hand-coded heuristic with no model weights) for any persona whose adapter directory is
missing or fails to load. Fine-tuning still requires a GPU and is documented with
estimated time on A100; the pre-trained adapters used in the talk are linked from the
README so users can drop them in and get the SLM path without doing the training run
themselves.

---

## The Three Personas

These are grounded in behavioral finance theory. Each persona has a name, a theoretical basis,
a decision style, and a prompt identity used during data generation and inference.

### 1. Momentum Trader
- **Theory**: Trend-following, recency bias (Jegadeesh & Titman 1993)
- **Behavior**: Buys recent price winners, sells recent losers; extrapolates short-term trends
- **Prompt identity**: "You are a momentum trader. You believe recent price trends persist in
  the short run. You buy assets that have risen recently and sell assets that have fallen.
  You do not anchor to fundamentals."

### 2. Value Investor
- **Theory**: Anchoring to fundamentals, loss aversion (Kahneman & Tversky; Shefrin & Statman)
- **Behavior**: Compares current price to an estimated fair value; buys on dips, sells on
  overextension; patient, low turnover
- **Prompt identity**: "You are a value investor. You estimate the intrinsic value of an asset
  from its fundamentals. You buy when the price is significantly below fair value and sell when
  it exceeds it. You are not influenced by short-term price momentum."

### 3. Noise Trader
- **Theory**: Sentiment-driven, overconfidence (De Long et al. 1990; Odean 1999)
- **Behavior**: Reacts to news headlines and social signals; high turnover; decisions are
  partially random and sentiment-amplified
- **Prompt identity**: "You are a noise trader. You react to market news and social sentiment.
  Your decisions are influenced by recent headlines and what you believe other traders are
  thinking. You trade frequently and are prone to overconfidence."

---

## Synthetic Data Format

Each `.jsonl` file contains training examples in the following format:

```json
{
  "persona": "momentum",
  "market_state": {
    "price_history": [100, 102, 105, 107, 110],
    "news": "Strong earnings report released this morning.",
    "fair_value": 104
  },
  "reasoning": "Prices have risen 10% over 5 periods with no reversal. The trend is intact. I will buy.",
  "action": "BUY",
  "quantity": 10
}
```

`reasoning` is the chain-of-thought. `action` is one of `BUY`, `SELL`, `HOLD`.
`quantity` is an integer 1–20.

---

## Market Simulation Design

The market is intentionally minimal. Do not add complexity that obscures the behavioral signal.

- **Assets**: One risky asset. Price determined by order imbalance each tick.
- **Tick structure**: Each tick, every agent receives the current market state (price history,
  latest news string, fair value signal) and emits an action + quantity.
- **Price update rule**: `price_t+1 = price_t * (1 + alpha * order_imbalance_t + epsilon)`
  where `order_imbalance = (buy_volume - sell_volume) / total_volume`, `alpha = 0.01`,
  and `epsilon ~ N(0, 0.001)` is a small noise term.
- **No short selling, no leverage, no transaction costs** in v1. These can be added later.
- **Agent endowment**: Each agent starts with 1000 cash and 0 shares.

---

## The Hero Experiment

This is the single result the talk shows. It must be reproducible from `notebooks/03_simulation_and_eval.ipynb`.

Run three simulation conditions, each with 30 agents, for 500 ticks:

1. **Homogeneous Momentum**: 30 momentum traders
2. **Homogeneous Value**: 30 value investors
3. **Mixed Population**: 10 momentum + 10 value + 10 noise traders

For each condition, compute and plot:
- Return distribution with fitted normal overlay (show fat tails in mixed condition)
- Autocorrelation of squared returns (show volatility clustering in mixed condition)
- Price series over time

The expected result: only the mixed population reproduces stylized facts. This is the empirical
payoff of the heterogeneous agent hypothesis.

---

## Open Problems Hooks (Section IV)

The following files are deliberately left as stubs with detailed docstring commentary explaining
the open problem. They are not full implementations — they are intellectual signposts.

- `eval/persona_drift.py`: Measures behavioral consistency of an agent over time as market
  conditions change. Hook into context drift / non-stationarity discussion.
- The README should include a section titled "Open Problems" that maps to Section IV of the talk:
  context drift, persona identifiability (inverse problem), equilibrium conditions.

---

## Documentation Standards

1. Every `.py` file begins with a module docstring that includes: purpose, talk section mapping,
   and key design decisions with brief justification.
2. Every function has a docstring with Args, Returns, and a one-line "Why this matters" note.
3. Notebooks have markdown cells between every code cell explaining what is happening and why.
4. Avoid clever one-liners. Prefer readable, explicit code even at the cost of brevity.
5. Where a simplification is made for clarity, leave a `# SIMPLIFICATION:` comment explaining
   what a production version would do differently.

---

## Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

`requirements.txt` should pin major versions. Include:
`transformers`, `peft`, `trl`, `torch`, `datasets`, `anthropic`, `numpy`, `scipy`,
`matplotlib`, `plotly`, `streamlit`, `jupyter`, `pyyaml`, `tqdm`

---

## Streamlit Demo App

The app is the primary presentation artifact — what gets shown on screen during the talk and
left running for attendees to explore afterward. It should look polished but remain simple.
Use Plotly for all charts (interactive, better for live demos than matplotlib).

Entry point: `streamlit run app/streamlit_app.py`

The app has three pages, accessible via the sidebar. Each page maps to a section of the talk.

---

### Page 1 — Persona Inspector (`pages/01_persona_inspector.py`)
**Talk section**: Section III (the fine-tuning framework)
**Purpose**: Let users browse pre-generated persona reasoning examples side by side.

This page loads from `eval/results/persona_examples.json` — a file generated offline by
running `python data/generate_personas.py --export-examples`. It contains a curated set of
market scenarios, each with the reasoning trace and action from all three personas, so
users can directly compare how each persona responds to the same inputs.

Layout:
- Sidebar controls:
  - Scenario selector: dropdown of available pre-generated scenarios
    (e.g. "Strong uptrend, bullish news", "Price below fair value, neutral news")
  - Persona filter: checkboxes to show/hide individual personas
- Main panel: three side-by-side cards, one per persona (or filtered subset)
  - Each card shows: persona name + color badge, the market state inputs, the full
    reasoning chain, and the final action/quantity in a highlighted callout
- Below cards: a "Why they disagree" expander with a one-paragraph explanation
  grounded in the behavioral finance theory behind each persona

This is the demo moment: the same market state, three completely different conclusions.
The side-by-side layout makes the divergence immediately legible to a non-technical audience.

If `eval/results/persona_examples.json` is missing, show `st.info()` directing the user
to run the export script.

---

### Page 2 — Simulation Explorer (`pages/02_run_simulation.py`)
**Talk section**: Section III (simulation framework)
**Purpose**: Explore pre-computed simulation runs across different persona compositions.

This page loads pre-saved simulation logs from `eval/results/simulations/` — a directory of
JSON files, one per run, generated offline by `run_simulation.py`. Each file encodes the full
tick-by-tick record: price series, per-agent actions, order flow, and final P&L.

Run `python simulation/run_simulation.py --precompute` offline to generate a library of runs
covering a range of persona compositions (all-momentum, all-value, mixed, noise-heavy, etc.).
These become the dataset the app explores. Aim for at least 6 pre-computed runs covering
meaningfully different compositions.

Layout:
- Sidebar controls:
  - Simulation selector: dropdown of available pre-computed runs, parsed from filenames
    (e.g. "10 Momentum / 10 Value / 10 Noise — seed 42")
  - Chart type toggles (checkboxes): Price Series, Order Flow, Agent P&L
- Main panel, shown after selecting a run:
  1. Price series over time (Plotly line chart), colored by dominant order flow each tick
  2. Per-tick order flow breakdown (stacked bar: buys vs sells, colored by persona)
  3. Agent P&L at end of simulation (bar chart grouped and colored by persona)
- Below charts: expandable "What drove this market?" section surfacing key moments
  from the log (largest single-tick price move, which persona type initiated it, etc.)

If `eval/results/simulations/` is empty or missing, show a clear `st.info()` directing
the user to run the precompute script. No live inference, no fallback modes needed.

---

### Page 3 — Hero Experiment (`pages/03_hero_experiment.py`)
**Talk section**: Section III/IV (results + open problems setup)
**Purpose**: Show the pre-computed hero experiment results with full stylized facts analysis.

This page loads pre-saved simulation results from `eval/results/hero_experiment.json`
(generated by `notebooks/03_simulation_and_eval.ipynb`). It does not run a live simulation.
If the results file does not exist, show a clear message directing the user to run the notebook first.

Layout:
- Header: brief framing of the three conditions (Homogeneous Momentum, Homogeneous Value, Mixed)
- Tab bar with three tabs, one per condition:
  - **Price Series**: Plotly line chart of full 500-tick price series
  - **Return Distribution**: histogram of log returns with fitted normal overlay;
    annotate excess kurtosis value
  - **Volatility Clustering**: autocorrelation of squared returns (bar chart, lags 1–20);
    annotate whether pattern is present
- Bottom panel: side-by-side kurtosis comparison across all three conditions (the money shot)
- Expander: "What are stylized facts?" — one paragraph explanation for non-specialist attendees

---

### App-Wide Design Conventions

- **Color palette**: assign a consistent color to each persona used everywhere
  - Momentum: `#E8A838` (amber)
  - Value: `#3A86FF` (blue)
  - Noise: `#FF6B6B` (coral)
- **Dark theme**: set via `.streamlit/config.toml` (`[theme] base = "dark"`)
- **Layout**: `layout="centered"` — cleaner on a projector, no need for wide mode
- **Sidebar**: always visible, contains page nav and any page-level controls
- **Error states**: every page must handle missing results files gracefully with
  `st.info()` messages pointing to the correct offline script to run. No crashes.
- **All data is pre-computed**: no model inference happens inside the app at runtime.
  The app is a viewer, not an executor. This keeps it fast, portable, and laptop-safe.
- **No authentication, no persistent state between sessions**

Add `streamlit` to requirements.txt and add the following to `.streamlit/config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#3A86FF"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1A1F2E"
textColor = "#FAFAFA"
```

---

## What NOT to Build

- No multi-asset simulation in v1
- No real market data ingestion
- No reinforcement learning (pure SFT only in v1)
- No distributed training
- No experiment tracking (MLflow, W&B) in v1 — keep dependencies minimal

These are all reasonable extensions but out of scope for this demonstration.
