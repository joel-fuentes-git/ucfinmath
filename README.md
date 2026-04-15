# Trader Personas: Heterogeneous SLM Agents in a Financial Market Simulation

**Thesis**: Micro-level behavioral heterogeneity produces macro-level stylized facts that
homogeneous agent populations cannot replicate.

---

## Background

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

## The Proposed Framework

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

### Current Results (fine-tuned LoRA SLM agents)

Every number in the table below was produced by `SLMAgent` — the fine-tuned
per-persona LoRA adapters on top of `Qwen/Qwen2.5-1.5B-Instruct`. The recorded
run is in `eval/results/hero_experiment.json` (all three conditions stamp
`metadata.agent_types` = `{..., "slm"}`). Log-return statistics across the
500-tick series:

| Condition | Final price | Excess kurtosis | ACF(r²) lag-1 | Total volume (buy / sell) |
|---|---|---|---|---|
| Homogeneous Momentum | $1.73   | +0.14  | −0.028 | 17,277 / 165,294 |
| Homogeneous Value    | $93.48  | −0.33  | **+0.134** | 40,494 / 43,083 |
| Mixed Population     | $3.43   | +0.06  | +0.028 | 29,202 / 149,172 |

What the SLM run actually shows — and why it is not what the rule-based
caricatures predicted:

1. **Homogeneous Momentum collapsed downward, not upward.** The rule-based
   momentum agent buys every uptick unconditionally; the fine-tuned SLM
   momentum persona, in contrast, reads any drawdown as a downtrend signal and
   sells into it. Sell volume was ~9.6× buy volume across 500 ticks and the
   price fell from $100 to $1.73. The failure mode is directional, not
   heavy-tailed: excess kurtosis is ≈ 0.14 (effectively Gaussian) with positive
   skew from the monotone decline. The stability pathology is real, but its
   sign is the opposite of what one-line momentum rules produce.

2. **Homogeneous Value is where the volatility-clustering signal lives.**
   Unlike the rule-based value agent (which was silent because prices never
   breached a 5% margin-of-safety band), the fine-tuned SLM value persona
   trades actively — ~84K total volume over 500 ticks. Price stayed in a tight
   $92–$101 corridor, and the ACF of squared returns at lag 1 is **+0.134**,
   well above the ±0.088 95% CI bound for n=500. This is the one condition in
   this run that cleanly exhibits a canonical stylized fact.

3. **Mixed Population also collapsed.** 10 value SLM agents were not enough
   ballast against 10 momentum SLM agents on the sell side; the mixed market
   ended at $3.43 with sell volume ~5× buy volume. Returns are near-normal
   (ek ≈ 0.06) and the lag-1 ACF of squared returns is only +0.028. The
   mixed-population hypothesis — that heterogeneity is what produces the
   stylized facts — is **not** supported here, and the condition that did
   produce clustering was homogeneous.

**What this tells us.** The fine-tuned SLM agents behave qualitatively
differently from the rule-based caricatures, and the places where stylized
facts appear are not where the going-in hypothesis predicted. The headline
results against the SLM baseline:

- *Fat tails:* no condition exhibits meaningful leptokurtosis in 500 ticks.
- *Volatility clustering:* present (and only present) in the homogeneous-value
  condition, driven by the internal dynamics of the value persona's active
  trading rather than by inter-persona disagreement.
- *Equilibrium and stability:* both the momentum-only and the mixed-population
  conditions lose stationarity under the SLM agents. The fraction of value
  agents required to stabilize a market with SLM momentum traders in it is
  almost certainly higher than 1/3, and the exact phase boundary is unknown.

This is the empirical starting point that the rest of the project — and the
open-problems section below — builds from.

### Constraints and limitations

Several design choices in this run bound how strongly any conclusion can be
drawn, and listing them explicitly is important because each one suggests a
different next experiment:

- **Single seed, single run per condition.** Every number above comes from one
  500-tick trajectory per condition (seed 42 for the hero experiment, seed 123
  for momentum-heavy). Stylized-fact statistics — especially kurtosis and the
  lag-1 ACF of squared returns — have high finite-sample variance, so a single
  run cannot distinguish "this condition has no fat tails" from "this seed did
  not produce fat tails." A proper test needs ensembles (≥20–50 independent
  seeds per condition) and confidence intervals on the statistics themselves,
  not just on the underlying returns.

- **Short horizon.** 500 ticks is roughly two trading years if one tick is a
  day, or a couple of days if one tick is a minute. Real-market stylized facts
  are typically measured over 10³–10⁵ observations; 499 log returns is at the
  very low end of where Jarque-Bera and Ljung-Box have power. The homogeneous-
  momentum and mixed runs also end in non-stationary collapse well before the
  500th tick, so most of their return series is dominated by the transient
  path to $1–$3 rather than by a stationary distribution. Ensembles *and*
  longer horizons — or post-collapse truncation — are both needed.

- **Small population.** 30 agents is small enough that individual-agent
  idiosyncrasy dominates. Neither the 10-agent sub-cohorts in the mixed
  condition nor the 30-agent homogeneous cohorts are large enough to talk
  about limits in the usual agent-based-modelling sense (N → ∞ with fixed
  persona fractions). Scaling N while holding fractions fixed would separate
  finite-size effects from genuine macro behavior.

- **Minimal market microstructure.** The price impact rule
  `p_{t+1} = p_t · (1 + α · imbalance + ε)` with α = 0.01 collapses the order
  book into a single linear-impact coefficient, has no inventory constraints,
  no funding, no short-selling rules, no transaction costs, and no bid/ask.
  Several real-market stylized facts (leverage effect, asymmetric volume
  response) are microstructure-driven and simply cannot emerge from this
  engine, regardless of agent behavior.

- **Exogenous noise is thin.** ε ∼ N(0, 0.001) per tick is Gaussian and
  memoryless. Fat tails and clustering in real markets are partly driven by
  fat-tailed, autocorrelated exogenous shocks (news arrivals, macro releases).
  With a Gaussian noise floor, the only route to fat tails is through
  endogenous agent dynamics, and the minimal engine limits how much of that
  can happen.

- **Context drift in the SLM prompt.** Each tick, the SLM agent receives a
  prompt that includes recent price history. Over 500 ticks, that context
  grows and shifts, which means a "momentum trader" at tick 400 is not being
  prompted identically to one at tick 10. The monotone sell-down in the
  momentum-only condition is consistent with context-drift: once a downtrend
  is in the window, every forward prompt reinforces it. Open Problem #1
  (below) flags this explicitly, but for the current run it is a confound,
  not a controlled variable.

- **Persona fidelity is not measured.** We have no independent test that the
  LoRA adapters actually implement their nominal theories — no holdout
  evaluation of (e.g.) "did the value adapter buy dips and sell
  overextensions in a held-out scenario set?" The adapters were trained on
  Claude-generated reasoning traces, so their "value" behavior is only as
  coherent as those traces were. Some of the anomalies above (momentum
  persona selling into weakness, rather than buying strength) may reflect
  persona drift from the training-data voice rather than a stable
  theory-consistent policy.

Taken together, these limits mean that the right reading of the current
results is **"preliminary; the hypothesis is not yet adequately tested,"**
rather than "the mixed-population hypothesis is falsified." The one result
that is comparatively robust is the qualitative stability picture — SLM
momentum populations are self-destabilizing in this engine, and a 1/3 value
cohort is not enough to stop it. The kurtosis and clustering results should
be treated as single-seed signals worth replicating, not as estimates.

### Next steps

Roughly ordered from cheapest / most diagnostic to most ambitious:

1. **Seed ensemble.** Re-run each of the three conditions for 30+ independent
   seeds and report the full distribution of (final price, excess kurtosis,
   lag-1 ACF of squared returns, lag-5 mean ACF). This is the single highest-
   value follow-up; every bullet above depends on knowing the within-condition
   variance before making any claim about between-condition differences.

2. **Longer horizons + stationarity-aware analysis.** Extend to 5,000–10,000
   ticks and compute stylized-fact statistics on the longest stationary window
   inside each run (detected via a rolling Augmented Dickey-Fuller or a simple
   drawdown cutoff). This separates "the SLM market has no fat tails" from
   "the SLM market crashed before a stationary distribution had time to form."

3. **Persona-fidelity audit.** Before running more dynamics, score each
   adapter on a held-out set of market scenarios with known theoretically-
   correct actions (e.g., "price is 20% below fair value, flat trend" should
   be a clean value-buy). This tells us whether the momentum persona is
   actually implementing momentum, or whether the training traces produced a
   subtly different policy that only looks like momentum on the training
   distribution.

4. **Composition sweep for the phase boundary.** Fix the total N = 30 and
   sweep the value fraction from 0 to 1 in steps of 0.1 (holding momentum and
   noise fractions equal over the remainder). Plot "fraction of runs that
   remain in a ±20% band after 500 ticks" vs. value fraction. This is a
   direct test of Open Problem #3 and gives a concrete number for "how many
   contrarians stabilize a market of SLM momentum traders."

5. **Context-drift ablations.** Re-run the hero experiment with (a) a
   fixed-length sliding context window, (b) a summarized-history context
   (last-K statistics rather than raw prices), and (c) a zero-history
   context (prompt only contains the current tick). If the momentum-only
   collapse persists in (c), the pathology is in the adapter; if it
   disappears, it is context-drift and belongs to Open Problem #1.

6. **Microstructure extensions.** Add a simple bid/ask spread, an inventory
   penalty, or an autocorrelated / fat-tailed ε. Each of these is a
   defensible candidate for "the missing ingredient" that lets fat tails
   emerge; running them individually isolates which additions matter.

7. **Scale-up.** Once (1)-(3) are in place, push N to 100–300. If the
   stylized-fact statistics do not stabilize by N = 300, that is itself a
   strong result (the finite-N dynamics are the whole story).

Items 1–3 are laptop-runnable with the current rule-based fallback for sanity
checks and require a GPU or long CPU time for the SLM path. Items 4–7 benefit
substantially from a GPU.

### Did the eval use the actual fine-tuned model?

Yes. The agent type used in any saved run is recorded in
`metadata.agent_types` inside each `eval/results/*.json` file. For the
currently checked-in `eval/results/hero_experiment.json`, all three conditions
have `agent_types` values that include `"slm"` (the fine-tuned LoRA adapter);
none fell back to the rule-based path. The same is true of the five
simulations under `eval/results/simulations/` — they are all SLM runs.

To **re-run the hero experiment locally** (e.g. with a different seed or tick
count), download the adapters from the Drive link above into the project's
`adapters/` folder and then:

```bash
python simulation/run_simulation.py --hero --require-adapters
jupyter notebook notebooks/03_simulation_and_eval.ipynb
```

The `--require-adapters` flag makes the run fail loudly if any persona's
adapter is missing, which is useful for catching download mistakes. The saved
`hero_experiment.json` records which agent type was actually used per persona.

The persona inspector page in the Streamlit app (powered by
`eval/results/persona_examples.json`) is a separate artifact: those reasoning traces
were generated by `claude-sonnet-4-6` via the Anthropic API in
`data/generate_personas.py --export-examples`, not by the fine-tuned SLM. Of the 30
reasoning traces in that file, 4 are placeholder strings from API failures during the
export — re-run the export script with a valid `ANTHROPIC_API_KEY` to populate them.

---

## Open Problems

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

In our minimal market, prices diverge without bound whenever SLM momentum
traders dominate. The hero experiment makes this concrete: the
homogeneous-momentum condition went from $100 to $1.73 in 500 ticks with sell
volume ~9.6× buy volume — the fine-tuned momentum persona, given any drawdown,
converges on a coordinated sell. Even the mixed condition (10 momentum + 10
value + 10 noise) collapses to $3.43; 1/3 value traders is not enough
negative feedback. Only the homogeneous-value condition stays stationary, and
that is the one condition where volatility clustering emerges (lag-1 ACF of
squared returns ≈ 0.134). Standard agent-based models often find that
heterogeneous populations stabilize prices through negative feedback (value
investors act as arbitrageurs), but the SLM run suggests the fraction of
arbitrageurs required to stabilize a market against fine-tuned momentum
traders is substantially higher than 1/3, and the conditions under which an
LLM-agent market produces both stationarity *and* the full set of stylized
facts remain unclear.

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
│   ├── generate_personas.py         # LLM-based data generation
│   └── personas/
│       ├── momentum.jsonl           # ~300 examples (placeholder: 5)
│       ├── value.jsonl
│       └── noise.jsonl
├── training/
│   ├── finetune.py                  # LoRA fine-tuning
│   └── configs/
│       ├── momentum.yaml
│       ├── value.yaml
│       └── noise.yaml
├── simulation/
│   ├── market.py                    # order book + price dynamics
│   ├── agent.py                     # rule-based and SLM agents
│   └── run_simulation.py            # orchestrator + CLI
├── eval/
│   ├── stylized_facts.py            # fat tails + volatility clustering
│   └── persona_drift.py             # open-problem stubs
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
