"""
streamlit_app.py — Main entry point for the Trader Personas demo app.

Talk section: Sections II, III, IV (overview page)

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
    st.markdown("**University of Chicago**")
    st.markdown("Financial Mathematics Program")
    st.divider()
    st.markdown(
        """
        **Talk Section Map**

        - **Page 1** — Persona Inspector
          *Section III: Fine-tuning framework*

        - **Page 2** — Simulation Explorer
          *Section III: Simulation framework*

        - **Page 3** — Hero Experiment
          *Section III/IV: Results + open problems*
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
    This demo accompanies a talk at the **University of Chicago Financial Mathematics Program**.

    **Core thesis**: Micro-level behavioral heterogeneity — agents reasoning in fundamentally
    different ways — is necessary and sufficient to produce the macro-level stylized facts
    observed in real financial markets (fat tails, volatility clustering). Homogeneous
    agent populations cannot reproduce these facts regardless of their individual complexity.
    """
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        **Section II — Background**

        Why language models as traders?
        The LoRA persona approach.
        How fine-tuning creates behavioral heterogeneity.
        """
    )

with col2:
    st.markdown(
        """
        **Section III — Framework**

        Data generation → LoRA fine-tuning → market simulation → evaluation.
        The three personas and their theoretical grounding.
        The hero experiment.
        """
    )

with col3:
    st.markdown(
        """
        **Section IV — Open Problems**

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

    3. **Hero Experiment** — The central empirical result: only the mixed-population
       simulation reproduces financial stylized facts.
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
    The rule-based fallback in the demo uses hand-coded decision logic — no GPU required.
    """
)
