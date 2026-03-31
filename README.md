# Trader Personas: Heterogeneous SLM Agents in a Financial Market Simulation

**Talk**: University of Chicago Financial Mathematics Program
**Thesis**: Micro-level behavioral heterogeneity produces macro-level stylized facts that
homogeneous agent populations cannot replicate.

---

## Talk Section Map

| Section | Topic | Key Files |
|---|---|---|
| II | Background: SLM fine-tuning + LLM-based market simulation | README (this section) |
| III | Proposed Framework: data → fine-tuning → simulation → eval | `data/`, `training/`, `simulation/`, `eval/`, `notebooks/` |
| IV | Open Problems: context drift, identifiability, equilibrium | `eval/persona_drift.py`, README (Open Problems section) |

---

## Section II — Background

### Why Language Models as Traders?

Traditional agent-based financial models use hand-coded behavioral rules. Each agent follows a
fixed decision function — for example, "buy if price rose more than 2% yesterday." These rules
are transparent but brittle: they cannot generalize to novel market regimes and they do not
capture the narrative reasoning that real traders report.

Large language models offer a different approach. They can be prompted to reason step-by-step
about a market state, incorporating price history, news, and fundamental signals into a coherent
decision. Several recent papers (Lim et al. 2024; Lopez-Lira & Tang 2023; Pelster & Val 2024)
have shown that LLMs can produce trading decisions that correlate with subsequent returns,
suggesting they encode economically relevant priors from their training data.

The challenge is **behavioral heterogeneity**. A single LLM prompted neutrally produces
near-identical reasoning for all agents in a simulation — the population collapses to a
homogeneous representative agent. Stylized facts (fat tails, volatility clustering) require
diverse, conflicting behavioral types.

### The LoRA Persona Approach

We solve this via **persona-conditioned fine-tuning**. Three distinct trader archetypes from
behavioral finance theory are each fine-tuned as a separate LoRA adapter on top of a shared
small language model (SLM) base — `Qwen/Qwen2.5-1.5B-Instruct`. The adapters are lightweight
(~2M parameters each) and can be hot-swapped at inference time, giving us a fleet of heterogeneous
agents that share the same base knowledge but reason in fundamentally different ways.

This is essentially **persona distillation**: a large, capable model (Claude) generates
high-quality reasoning traces in the voice of each persona, and those traces become supervised
training data for the smaller model.

---

## Section III — The Proposed Framework

### Step 1: Data Generation (`data/generate_personas.py`)

We prompt `claude-sonnet-4-6` with persona identity strings and diverse market scenarios to generate
chain-of-thought reasoning traces. Each example contains:

- A market state: price history, a news string, and a fair-value estimate
- A reasoning chain: the persona's internal monologue
- An action: BUY, SELL, or HOLD
- A quantity: 1–20 shares

About 300 examples per persona are generated and saved to `data/personas/{persona}.jsonl`.

### Step 2: LoRA Fine-Tuning (`training/finetune.py`)

Each persona adapter is trained independently using:

- **Base model**: `Qwen/Qwen2.5-1.5B-Instruct` — small enough for quantized CPU inference
- **PEFT LoRA**: rank-8 adapters on the query and value projection matrices
- **TRL SFTTrainer**: supervised fine-tuning on the chain-of-thought + action format

Training takes approximately 15 minutes per persona on an A100 GPU. Pre-trained adapters
can be loaded directly if you do not have a GPU (see `training/finetune.py` for instructions).

**Rule-based fallback**: `simulation/agent.py` provides a `RuleBasedAgent` that implements
each persona's logic without a model. All notebooks and the Streamlit app use rule-based
agents by default so the project runs on any laptop.

### Step 3: Market Simulation (`simulation/`)

The market is intentionally minimal. One risky asset. N agents. T ticks.

**Price update rule**:
```
price_{t+1} = price_t * (1 + alpha * order_imbalance_t + epsilon_t)
```
where:
- `order_imbalance = (buy_volume - sell_volume) / total_volume`
- `alpha = 0.01` (price impact coefficient)
- `epsilon ~ N(0, 0.001)` (exogenous noise)

Each tick, every agent receives the current market state and emits an action and quantity.
Orders are aggregated to compute order imbalance; the price updates; the cycle repeats.

### Step 4: Evaluation (`eval/stylized_facts.py`)

Two classic stylized facts from the empirical finance literature:

1. **Fat tails**: Real log-return distributions have excess kurtosis (kurtosis > 3). A normal
   distribution cannot explain the frequency of large moves.

2. **Volatility clustering**: Large moves tend to be followed by large moves (of either sign).
   Formally, the autocorrelation of squared returns is significantly positive at short lags.

We test for both using Jarque-Bera statistics and the ACF of squared returns.

---

## The Three Personas

### 1. Momentum Trader

**Theory**: Jegadeesh & Titman (1993) documented that stocks with high returns over the past
3–12 months continue to outperform over the next 3–12 months. Momentum traders exploit this
by buying recent winners and selling recent losers. Their behavior amplifies trends and
contributes to price overshooting.

**Behavior**: Looks at the recent price series; if it's trending up, buys; if trending down,
sells. Ignores fundamental value entirely.

**Prompt identity**:
> "You are a momentum trader. You believe recent price trends persist in the short run.
> You buy assets that have risen recently and sell assets that have fallen.
> You do not anchor to fundamentals."

### 2. Value Investor

**Theory**: Kahneman & Tversky's prospect theory and Shefrin & Statman's behavioral portfolio
theory both predict that investors anchor to a reference point (fair value) and become loss-
averse below it. Value investors provide a natural stabilizing force: they buy when prices
fall below fair value and sell when they rise above it.

**Behavior**: Compares current price to a fair value estimate. Buys on dips (price significantly
below fair value), sells on overextension (price significantly above fair value). Patient and
low-turnover.

**Prompt identity**:
> "You are a value investor. You estimate the intrinsic value of an asset from its fundamentals.
> You buy when the price is significantly below fair value and sell when it exceeds it.
> You are not influenced by short-term price momentum."

### 3. Noise Trader

**Theory**: De Long, Shleifer, Summers & Waldmann (1990) showed that noise traders — who trade
on pseudo-signals rather than information — can survive in equilibrium and affect prices. Odean
(1999) documented systematic overconfidence in retail investors. Noise traders add stochastic
variation and amplify sentiment-driven moves.

**Behavior**: Reacts to news headlines and vague sentiment signals. Partially random, high
turnover, prone to overconfidence. Adds entropy to the market.

**Prompt identity**:
> "You are a noise trader. You react to market news and social sentiment. Your decisions are
> influenced by recent headlines and what you believe other traders are thinking. You trade
> frequently and are prone to overconfidence."

---

## The Hero Experiment

Three simulation conditions, each with 30 agents, for 500 ticks:

| Condition | Composition | Expected behavior |
|---|---|---|
| Homogeneous Momentum | 30 momentum traders | Persistent trends, herding, price drift |
| Homogeneous Value | 30 value investors | Mean reversion to fair value, low volatility |
| Mixed Population | 10 momentum + 10 value + 10 noise | Fat tails, volatility clustering |

**Expected result**: Only the mixed population reproduces the two core stylized facts. The
momentum-only market trends excessively; the value-only market is too stable; only the conflict
between behavioral types generates the realistic, heteroskedastic price dynamics we observe in
real markets.

To reproduce:
```bash
python simulation/run_simulation.py --hero
jupyter notebook notebooks/03_simulation_and_eval.ipynb
```

---

## Open Problems (Section IV)

### 1. Context Drift

LLM-based agents accumulate market history in their context window as the simulation runs.
After hundreds of ticks, the context is dominated by recent price history, which can subtly
shift the agent's behavior — a momentum trader may start exhibiting mean-reversion tendencies
simply because the context has changed, not because its "beliefs" have changed.

**Research question**: Can we design context representations that are invariant to the length
of simulation history? Possible approaches include summarization (compress old history into
statistics), recency-weighted context, or explicit belief-state tracking.

`eval/persona_drift.py` is a stub that outlines a measurement framework for this problem.

### 2. Persona Identifiability (The Inverse Problem)

Given a sequence of observed trades from an anonymous agent, can we identify which persona it
belongs to? This is the inverse of the simulation: instead of simulating forward from a known
persona, we want to infer backward from behavior to type.

This problem is hard for several reasons:
- Individual trade decisions are noisy; persona signal emerges only over many trades
- The three personas have correlated behaviors in trending markets (both momentum and noise
  traders may buy in an uptrend)
- A fine-tuned SLM may not produce perfectly consistent persona behavior under all market states

**Research question**: What is the minimum number of observed trades required to reliably identify
a persona? Is there a market state that is maximally discriminating (i.e., the three personas
disagree most sharply)?

### 3. Equilibrium and Stability

In our minimal market, prices can diverge without bound if momentum traders dominate. Does the
market have a well-defined stationary distribution? Under what compositions of persona types
does the price process remain stationary?

Standard agent-based models often find that heterogeneous populations stabilize prices through
negative feedback (value investors act as arbitrageurs), but the conditions for stability in an
LLM-agent market are not well understood.

**Research question**: Is there a "phase transition" in persona composition — a threshold fraction
of value investors below which the market becomes non-stationary? Can we characterize this
analytically or only empirically?

---

## Setup

```bash
# 1. Clone and enter the project
cd trader-personas

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Anthropic API key (needed only for data generation)
export ANTHROPIC_API_KEY=your_key_here

# 5. Generate synthetic persona data (optional — placeholder data is included)
python data/generate_personas.py --persona momentum --n-examples 300
python data/generate_personas.py --persona value --n-examples 300
python data/generate_personas.py --persona noise --n-examples 300

# 6. Export the persona comparison file for the app
python data/generate_personas.py --export-examples

# 7. Pre-compute simulation results for the app
python simulation/run_simulation.py --precompute
python simulation/run_simulation.py --hero

# 8. Launch the Streamlit demo app
streamlit run app/streamlit_app.py

# 9. Or open the notebooks for a step-by-step walkthrough
jupyter notebook notebooks/
```

**GPU fine-tuning** (optional):
```bash
python training/finetune.py --persona momentum --config training/configs/momentum.yaml
python training/finetune.py --persona value    --config training/configs/value.yaml
python training/finetune.py --persona noise    --config training/configs/noise.yaml
```
Estimated time: ~15 minutes per persona on an A100. See `training/finetune.py` for instructions
on loading pre-trained adapters if you do not have a GPU.

---

## Project Structure

```
trader-personas/
├── README.md                        # this file
├── requirements.txt
├── data/
│   ├── generate_personas.py         # [Section III] LLM-based data generation
│   └── personas/
│       ├── momentum.jsonl           # ~300 examples (placeholder: 5)
│       ├── value.jsonl
│       └── noise.jsonl
├── training/
│   ├── finetune.py                  # [Section III] LoRA fine-tuning
│   └── configs/
│       ├── momentum.yaml
│       ├── value.yaml
│       └── noise.yaml
├── simulation/
│   ├── market.py                    # [Section III] order book + price dynamics
│   ├── agent.py                     # [Section III] rule-based and SLM agents
│   └── run_simulation.py            # orchestrator + CLI
├── eval/
│   ├── stylized_facts.py            # [Section III/IV] fat tails + volatility clustering
│   └── persona_drift.py             # [Section IV] open-problem stubs
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_finetuning_walkthrough.ipynb
│   └── 03_simulation_and_eval.ipynb
└── app/
    ├── streamlit_app.py
    └── pages/
        ├── 01_persona_inspector.py
        ├── 02_run_simulation.py
        └── 03_hero_experiment.py
```

---

## Citation

If you use this project, please cite the following behavioral finance foundations:

- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1), 65–91.
- Kahneman, D., & Tversky, A. (1979). Prospect theory. *Econometrica*, 47(2), 263–291.
- De Long, J. B., Shleifer, A., Summers, L. H., & Waldmann, R. J. (1990). Noise trader risk in financial markets. *Journal of Political Economy*, 98(4), 703–738.
- Odean, T. (1999). Do investors trade too much? *American Economic Review*, 89(5), 1279–1298.
