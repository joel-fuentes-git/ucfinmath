"""
streamlit_app.py — Main entry point for the Trader Personas demo app.

Purpose:
    Entry point for the Streamlit multi-page application. This page provides a conceptual
    overview of the project and navigation instructions. The actual content lives in the
    three pages under app/pages/.

    Run with: streamlit run app/streamlit_app.py

Key design decisions:
    - The main page is intentionally brief. Attendees should spend their time on the
      three content pages, not on meta-navigation.
    - All app configuration (dark theme, layout) is in .streamlit/config.toml so it
      applies consistently across all pages.
"""

import streamlit as st

st.set_page_config(
    page_title="Trader Personas",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Trader Personas")
    st.divider()
    st.markdown(
        """
        **Pages**

        - **Page 1** — Persona Inspector
          *Fine-tuning framework*

        - **Page 2** — Simulation Explorer
          *Simulation framework*

        - **Page 3** — Hero Experiment
          *Results + open problems*
        """
    )
    st.divider()
    st.markdown(
        """
        **Setup**
        ```bash
        # Generate data
        python data/generate_personas.py \\
            --export-examples

        # Pre-compute simulations
        python simulation/run_simulation.py \\
            --precompute
        python simulation/run_simulation.py \\
            --hero
        ```
        """
    )

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.title("Trader Personas")
st.markdown(
    "### Heterogeneous SLM Agents in a Financial Market Simulation"
)

st.markdown(
    """
    **Going-in thesis**: Micro-level behavioral heterogeneity — agents reasoning in fundamentally
    different ways — should be necessary and sufficient to produce the macro-level stylized facts
    observed in real financial markets (fat tails, volatility clustering).

    **Status of the demo**: every number shown on the Hero Experiment and
    Simulation Explorer pages was produced by the fine-tuned LoRA `SLMAgent`
    (per-persona adapters on top of `Qwen/Qwen2.5-1.5B-Instruct`). The results
    are surprising in ways the rule-based caricature did not predict: SLM
    momentum traders run the price *down* rather than up, the value-only
    market is the one that exhibits measurable volatility clustering, and no
    condition produces strong fat tails in 500 ticks. The Hero Experiment page
    walks through what each condition actually shows.
    """
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        **Background**

        Why language models as traders?
        The LoRA persona approach.
        How fine-tuning creates behavioral heterogeneity.
        """
    )

with col2:
    st.markdown(
        """
        **Framework**

        Data generation → LoRA fine-tuning → market simulation → evaluation.
        The three personas and their theoretical grounding.
        The hero experiment.
        """
    )

with col3:
    st.markdown(
        """
        **Open Problems**

        Context drift in long simulations.
        Persona identifiability (the inverse problem).
        Equilibrium and stability conditions.
        """
    )

st.divider()

st.markdown(
    """
    ### Navigate Using the Sidebar

    Use the page links in the left sidebar to explore:

    1. **Persona Inspector** — Compare how momentum, value, and noise traders
       reason about the same market state side by side.

    2. **Simulation Explorer** — Explore pre-computed simulation runs across
       different persona compositions. See price dynamics, order flow, and P&L.

    3. **Hero Experiment** — Pre-computed results across the three persona compositions,
       produced by the fine-tuned LoRA SLM. Reports actual per-condition price dynamics,
       excess kurtosis, and volatility-clustering diagnostics from the 500-tick runs.
    """
)

st.divider()

st.markdown(
    """
    #### The Three Personas

    | Persona | Theory | Key Behavior |
    |---|---|---|
    | **Momentum** | Jegadeesh & Titman (1993) | Buys winners, sells losers; trend-following |
    | **Value** | Kahneman & Tversky; Shefrin & Statman | Anchors to fair value; buys dips, sells overextension |
    | **Noise** | De Long et al. (1990); Odean (1999) | Sentiment-driven; high turnover; partially random |

    Each persona is fine-tuned as a separate LoRA adapter on `Qwen/Qwen2.5-1.5B-Instruct`.
    The fine-tuned adapters from the talk's training run are available for download here:
    [LoRA adapters — Google Drive](https://drive.google.com/drive/folders/1hOQKb-YTvjp_77FhwAyrMdN0I6DX85Nu?usp=share_link).

    **Agent type used at simulation time.** `simulation/run_simulation.py` auto-detects
    LoRA adapters under `<project_root>/adapters/<persona>/`. When a persona's adapter
    is present it uses `SLMAgent` (the fine-tuned LoRA SLM); when it is missing it
    automatically falls back to `RuleBasedAgent` (hand-coded heuristic). The decision is
    per persona, so a partial set of adapters works. The agent type that was actually
    used is recorded in each saved JSON's `metadata.agent_types` field, and the Hero
    Experiment and Simulation Explorer pages read that field and label every run
    accordingly. The persona inspector on the next page is a separate artifact: those
    reasoning traces were generated by Claude via the Anthropic API, not by either of
    the in-process agent types above.
    """
)
