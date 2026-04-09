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

Training takes approximately 15 minutes per persona on an A100 GPU.

**Pre-trained adapters (download)**: The three LoRA adapters used in this talk have already
been fine-tuned and saved to a shared Google Drive folder so you do not need a GPU to run
inference yourself:

> [LoRA adapters — Google Drive](https://drive.google.com/drive/folders/1hOQKb-YTvjp_77FhwAyrMdN0I6DX85Nu?usp=share_link)

Download the three adapter directories so the project's `adapters/` folder ends up looking
like this (each persona directory must contain at minimum an `adapter_config.json`):

```
adapters/
├── momentum/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── value/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── noise/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

**That's all the configuration needed.** The simulation orchestrator
(`simulation/run_simulation.py`) auto-detects this directory and uses the fine-tuned
`SLMAgent` for any persona whose adapter is present. The detection is **per persona**:
if you only download the momentum adapter, momentum agents use the SLM and value/noise
agents use the rule-based fallback.

**Automatic fallback to rule-based agents.** If a persona's adapter directory is missing,
or if `transformers`/`peft` are not installed, that persona automatically falls back to
`RuleBasedAgent` (a hand-coded heuristic with no model weights required). This means the
project always runs end-to-end on any laptop, regardless of whether the adapters have been
downloaded — you simply get the SLM path when adapters are present and the rule-based
caricature when they are not. CLI flags to override the defaults:

```bash
# Default: SLM if adapters present, rules otherwise
python simulation/run_simulation.py --hero

# Force rules even if adapters are present (reproduce the rule-based baseline)
python simulation/run_simulation.py --hero --force-rules

# Fail loudly if any persona's adapter is missing (catch configuration mistakes)
python simulation/run_simulation.py --hero --require-adapters

# Point at a non-default adapters directory
python simulation/run_simulation.py --hero --adapters-dir /path/to/adapters

# Use a different base model (must match what the adapters were trained against)
python simulation/run_simulation.py --hero --base-model Qwen/Qwen2.5-1.5B-Instruct
```

The agent type that was actually used per persona is stamped into each saved JSON's
`metadata.agent_types` field, so the notebook and Streamlit app can faithfully report
which model produced any given run.

**Memory note for the SLM path.** Each persona's model + adapter takes ~6 GB of RAM
(float32, 1.5B parameters). All agents of the same persona share one model instance
via a process-wide cache, so a 30-agent mixed run loads three models (~18 GB), not 30.
On a 16 GB laptop the rule-based fallback is the practical default; on a workstation
or GPU box the SLM path is preferred and is what the talk's main results are intended
to be reproduced from.

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

**Going-in hypothesis**: Only the mixed population would reproduce the two core stylized facts.
The momentum-only market would trend excessively; the value-only market would be too stable;
only the conflict between behavioral types would generate the realistic, heteroskedastic price
dynamics we observe in real markets.

### Actual Results (from the current run)

The recorded run is in `eval/results/hero_experiment.json`. Computed log-return statistics
across the 500-tick series for each condition:

| Condition | Final price | Excess kurtosis | ACF(r²) lag-1 | Total trades |
|---|---|---|---|---|
| Homogeneous Momentum | $14,139.52 | **+17.27** | -0.004 | 216,870 buys / 0 sells |
| Homogeneous Value    | $99.62     | +0.17       | -0.054 | 0 buys / 0 sells |
| Mixed Population     | $103.36    | **−1.58**   | +0.122 | 46,683 buys / 45,448 sells |

The empirical picture is **more interesting and more nuanced than the going-in hypothesis**:

1. **Homogeneous Momentum became degenerate.** With only trend-followers and no contrarians,
   the population locked into a runaway upward trajectory: every agent bought every tick, no
   one ever sold (zero sell volume across all 500 ticks), and price multiplied by ~141×. The
   17.3 excess kurtosis comes from this pathology, not from healthy fat tails — the return
   distribution is dominated by a unipolar drift with occasional small downticks from the
   exogenous Gaussian noise.

2. **Homogeneous Value did nothing at all.** The value rule (`buy < 95% of fair value`,
   `sell > 105% of fair value`) is never triggered because the noise term never pushes the
   price outside `[$98.98, $101.06]`. With zero trades, the price series is just the cumulative
   exogenous noise — kurtosis ≈ 0, no clustering. Value investors are mean-reverting *only when
   the price actually deviates*; otherwise they are silent.

3. **The Mixed Population produced volatility but not fat tails.** Returns are roughly six
   times more volatile than the value-only condition (std 0.0069 vs 0.0010), and the lag-1
   autocorrelation of squared returns is +0.122 — modest but the only condition where it is
   meaningfully positive. **However, excess kurtosis is *negative* (−1.58).** The mixed market
   is platykurtic, not leptokurtic. Behavioral diversity in this minimal market is necessary
   to generate any volatility-clustering signal at all, but it is *not* sufficient to produce
   the fat tails of real markets.

**What this tells us.** The going-in hypothesis is partially refuted by these specific runs.
This is itself an honest empirical finding and a good Section IV motivator: a minimal market
with three rule-based personas reproduces *one* of the two stylized facts (weak volatility
clustering) but not the other (fat tails). The momentum-only pathology also tells us that the
absence of contrarian agents is not "trending too much" — it is loss of stationarity entirely.
Both observations point at the equilibrium-and-stability open problem.

### Did the eval use the actual fine-tuned model?

The agent type used in any saved run is now recorded in
`metadata.agent_types` inside each `eval/results/*.json` file. Open the file and check
that field — it will be `"slm"` for personas that used the fine-tuned LoRA adapter or
`"rules"` for personas that fell back to `RuleBasedAgent`.

For the **currently checked-in `eval/results/hero_experiment.json`**, the answer is
`{"momentum": "rules", "value": "rules", "noise": "rules"}` — every persona used the
rule-based fallback. (Earlier versions of `run_simulation.py` did not have an SLM code
path at all; the `agent_types` metadata field was added when the auto-detection landed.)
The numerical results in the table above therefore test the rule-based caricatures, not
the fine-tuned SLM.

To **re-run the hero experiment using the fine-tuned adapters**, download them from the
Drive link above into the project's `adapters/` folder so the layout matches the diagram
in Step 2, then:

```bash
python simulation/run_simulation.py --hero --require-adapters
jupyter notebook notebooks/03_simulation_and_eval.ipynb
```

The `--require-adapters` flag makes the run fail loudly if any persona's adapter is
missing, which is useful for catching download mistakes. Without it, missing personas
silently fall back to rules. Either way, the saved `hero_experiment.json` will record
which agent type was actually used, and the Streamlit hero experiment page reads that
field and labels the run accordingly.

The persona inspector page in the Streamlit app (powered by
`eval/results/persona_examples.json`) is a separate artifact: those reasoning traces
were generated by `claude-sonnet-4-6` via the Anthropic API in
`data/generate_personas.py --export-examples`, not by the fine-tuned SLM. Of the 30
reasoning traces in that file, 4 are placeholder strings from API failures during the
export — re-run the export script with a valid `ANTHROPIC_API_KEY` to populate them.

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

In our minimal market, prices can diverge without bound if momentum traders dominate.
The hero experiment makes this concrete: the homogeneous-momentum condition went from
$100 to $14,139 in 500 ticks with **zero sell volume across the entire run**. There was
no oscillation, no correction, no price discovery — just monotone exponential drift. This
is not "trending too much"; it is loss of stationarity entirely. The mixed condition, by
contrast, stays in a $94–$106 band but only generates *weak* volatility clustering and
no fat tails. Standard agent-based models often find that heterogeneous populations
stabilize prices through negative feedback (value investors act as arbitrageurs), but the
conditions under which an LLM-agent (or rule-based-agent) market produces both
stationarity *and* the full set of stylized facts remain unclear.

**Research question**: Is there a "phase transition" in persona composition — a threshold fraction
of value investors below which the market becomes non-stationary? Inside the stationary regime,
is there a sub-region of compositions that also produces fat tails, or do fat tails require
ingredients (e.g. asymmetric information, finite-memory traders, jumpy fundamentals) that this
minimal market does not have? Can either threshold be characterized analytically or only
empirically?

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
