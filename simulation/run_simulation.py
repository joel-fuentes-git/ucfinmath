"""
run_simulation.py — Orchestrator for multi-agent market simulation runs.

Talk section: Section III — The Proposed Framework

Purpose:
    Runs the multi-agent market simulation and saves results to disk.
    Supports three operating modes via CLI flags:
    - --hero: Runs the three hero experiment conditions and saves to
      eval/results/hero_experiment.json. These results power the hero experiment
      notebook and Streamlit page.
    - --precompute: Runs 6 predefined compositions for the Simulation Explorer
      Streamlit page and saves each as a JSON file under eval/results/simulations/.
    - --persona-counts: Runs a custom composition specified on the command line.

Key design decisions:
    - Uses RuleBasedAgent by default. No model weights required. This keeps the
      simulation fast, portable, and runnable on any laptop.
    - The simulation returns a complete log dict that is JSON-serializable. This means
      results can be inspected, plotted, and shared without re-running the simulation.
    - We fix the random seed per run so results are reproducible. Different compositions
      use different seeds to avoid trivial correlations.
    - P&L is computed at the end of the simulation using the final market price.
"""

import argparse
import json
import sys
from pathlib import Path

# Add the project root to sys.path so we can import simulation and eval modules
# regardless of where the script is called from.
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from simulation.market import Market
from simulation.agent import RuleBasedAgent


# ---------------------------------------------------------------------------
# Core simulation function
# ---------------------------------------------------------------------------


def run_simulation(
    agents: list,
    n_ticks: int = 500,
    initial_price: float = 100.0,
    fair_value: float = 100.0,
    seed: int = 42,
) -> dict:
    """
    Run one complete simulation and return the full log.

    Each tick:
    1. Every agent observes the current market state.
    2. Every agent emits a decision (action + quantity).
    3. The market aggregates orders and updates the price.
    4. Agent portfolios are updated based on executed trades.

    Args:
        agents: A list of agent objects (RuleBasedAgent or SLMAgent).
        n_ticks: Number of simulation ticks to run.
        initial_price: Starting price of the asset.
        fair_value: Fundamental value of the asset (constant throughout).
        seed: Random seed for the market's noise term.

    Returns:
        A dict containing the full simulation log:
            - price_series: list of prices (length n_ticks + 1)
            - order_flow: list of per-tick order flow dicts
            - agent_decisions: list of per-tick agent decision lists
            - agent_pnl: dict mapping agent_id -> final P&L
            - agent_trade_counts: dict mapping agent_id -> number of executed trades
            - agent_personas: dict mapping agent_id -> persona
            - persona_composition: dict mapping persona -> count
            - metadata: dict with run parameters

    Why this matters:
        A complete, self-contained log is essential for the Streamlit app and for
        reproducing results. Every interesting quantity can be derived from this log
        without needing to re-run the simulation.
    """
    market = Market(
        initial_price=initial_price,
        fair_value=fair_value,
        seed=seed,
    )

    # Record the persona composition for metadata.
    persona_composition: dict[str, int] = {}
    for agent in agents:
        persona_composition[agent.persona] = persona_composition.get(agent.persona, 0) + 1

    # Per-tick log of all agent decisions.
    all_tick_decisions: list[list[dict]] = []

    for tick in range(n_ticks):
        market_state = market.get_state()

        # Collect all agent decisions for this tick.
        tick_orders = []
        tick_decisions = []

        for agent in agents:
            decision = agent.decide(market_state)
            tick_decisions.append(decision)

            # Convert decision to an order dict for the market.
            tick_orders.append({
                "action": decision["action"],
                "quantity": decision["quantity"],
                "persona": decision["persona"],
            })

        all_tick_decisions.append(tick_decisions)

        # Advance the market.
        new_price = market.step(tick_orders)

        # Execute trades and update agent portfolios.
        for agent, decision in zip(agents, tick_decisions):
            if decision["action"] != "HOLD":
                agent.execute_trade(decision["action"], decision["quantity"], new_price)

    # Final market price for P&L calculation.
    final_price = market.price_history[-1]

    # Compile P&L and trade counts for each agent.
    agent_pnl = {}
    agent_trade_counts = {}
    agent_personas = {}

    for agent in agents:
        agent_pnl[agent.agent_id] = agent.get_pnl(final_price)
        agent_trade_counts[agent.agent_id] = len(agent.trade_history)
        agent_personas[agent.agent_id] = agent.persona

    return {
        "price_series": market.price_history,
        "order_flow": market.order_flow_history,
        "news_history": market.news_history,
        "agent_decisions": all_tick_decisions,
        "agent_pnl": agent_pnl,
        "agent_trade_counts": agent_trade_counts,
        "agent_personas": agent_personas,
        "persona_composition": persona_composition,
        "market_summary": market.get_summary(),
        "metadata": {
            "n_ticks": n_ticks,
            "initial_price": initial_price,
            "fair_value": fair_value,
            "seed": seed,
            "n_agents": len(agents),
        },
    }


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_agents(persona_counts: dict[str, int], seed_offset: int = 0) -> list:
    """
    Create a list of RuleBasedAgent instances from a persona count dict.

    Args:
        persona_counts: A dict mapping persona name -> number of agents.
                        E.g., {"momentum": 10, "value": 10, "noise": 10}
        seed_offset: An integer offset added to each agent's seed to ensure
                     different agents have independent random states.

    Returns:
        A list of RuleBasedAgent instances.

    Why this matters:
        A factory function makes it easy to create named agent populations
        programmatically without repeating boilerplate.
    """
    agents = []
    agent_index = 0

    for persona, count in persona_counts.items():
        for i in range(count):
            agent_id = f"{persona}_{i:03d}"
            # Each agent gets a unique seed derived from its index and the offset.
            agent_seed = seed_offset + agent_index * 31 + hash(agent_id) % 1000
            agents.append(RuleBasedAgent(persona, agent_id, seed=agent_seed))
            agent_index += 1

    return agents


# ---------------------------------------------------------------------------
# Precompute functions
# ---------------------------------------------------------------------------


def run_hero_experiment(output_path: Path, n_ticks: int = 500) -> None:
    """
    Run the three hero experiment conditions and save results.

    The three conditions are:
    1. Homogeneous Momentum: 30 momentum traders
    2. Homogeneous Value: 30 value investors
    3. Mixed Population: 10 momentum + 10 value + 10 noise traders

    Args:
        output_path: Path to save the JSON results file.
        n_ticks: Number of simulation ticks (default: 500).

    Returns:
        None. Writes the results to output_path.

    Why this matters:
        The hero experiment is the central empirical result of this project.
        This function produces the canonical results used in the notebook and app.
    """
    conditions = [
        {
            "name": "Homogeneous Momentum",
            "label": "momentum_only",
            "persona_counts": {"momentum": 30},
            "seed": 42,
        },
        {
            "name": "Homogeneous Value",
            "label": "value_only",
            "persona_counts": {"value": 30},
            "seed": 42,
        },
        {
            "name": "Mixed Population",
            "label": "mixed",
            "persona_counts": {"momentum": 10, "value": 10, "noise": 10},
            "seed": 42,
        },
    ]

    results = {}

    for condition in conditions:
        label = condition["label"]
        print(f"Running hero condition: {condition['name']} ({label})...")

        agents = create_agents(condition["persona_counts"], seed_offset=condition["seed"])
        log = run_simulation(
            agents,
            n_ticks=n_ticks,
            initial_price=100.0,
            fair_value=100.0,
            seed=condition["seed"],
        )

        # Store only the data needed for analysis and plotting.
        results[label] = {
            "name": condition["name"],
            "persona_composition": log["persona_composition"],
            "price_series": log["price_series"],
            "order_flow": log["order_flow"],
            "market_summary": log["market_summary"],
            "metadata": log["metadata"],
        }

        print(
            f"  Done. Final price: {log['market_summary']['final_price']:.2f}, "
            f"N trades: {sum(log['agent_trade_counts'].values())}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nHero experiment results saved to: {output_path}")


def run_precompute_library(output_dir: Path, n_ticks: int = 500) -> None:
    """
    Run 6 predefined simulation compositions and save each as a JSON file.

    The 6 compositions cover a range of behavioral mixes for the Simulation Explorer
    page in the Streamlit app.

    Args:
        output_dir: Directory to save the JSON files.
        n_ticks: Number of simulation ticks per run.

    Returns:
        None. Writes one JSON file per run to output_dir.

    Why this matters:
        Pre-computing results makes the Streamlit app fast and portable. Users can
        explore different market conditions without waiting for simulations to run.
    """
    compositions = [
        {
            "name": "All Momentum (30)",
            "filename": "all_momentum_30.json",
            "persona_counts": {"momentum": 30},
            "seed": 42,
        },
        {
            "name": "All Value (30)",
            "filename": "all_value_30.json",
            "persona_counts": {"value": 30},
            "seed": 42,
        },
        {
            "name": "All Noise (30)",
            "filename": "all_noise_30.json",
            "persona_counts": {"noise": 30},
            "seed": 42,
        },
        {
            "name": "Mixed: 10 Momentum / 10 Value / 10 Noise",
            "filename": "mixed_10_10_10.json",
            "persona_counts": {"momentum": 10, "value": 10, "noise": 10},
            "seed": 42,
        },
        {
            "name": "Momentum Heavy: 20 Momentum / 5 Value / 5 Noise",
            "filename": "momentum_heavy_20_5_5.json",
            "persona_counts": {"momentum": 20, "value": 5, "noise": 5},
            "seed": 123,
        },
        {
            "name": "Value Heavy: 5 Momentum / 20 Value / 5 Noise",
            "filename": "value_heavy_5_20_5.json",
            "persona_counts": {"momentum": 5, "value": 20, "noise": 5},
            "seed": 456,
        },
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    for composition in compositions:
        filename = composition["filename"]
        print(f"Running: {composition['name']}...")

        agents = create_agents(composition["persona_counts"], seed_offset=composition["seed"])
        log = run_simulation(
            agents,
            n_ticks=n_ticks,
            initial_price=100.0,
            fair_value=100.0,
            seed=composition["seed"],
        )

        # Build the output record.
        record = {
            "name": composition["name"],
            "filename": filename,
            "persona_composition": log["persona_composition"],
            "price_series": log["price_series"],
            "order_flow": log["order_flow"],
            "agent_pnl": log["agent_pnl"],
            "agent_trade_counts": log["agent_trade_counts"],
            "agent_personas": log["agent_personas"],
            "market_summary": log["market_summary"],
            "metadata": log["metadata"],
        }

        out_path = output_dir / filename
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

        print(f"  Saved to: {out_path}")

    print(f"\nPrecomputed {len(compositions)} simulation runs in {output_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_persona_counts(counts_str: str) -> dict[str, int]:
    """
    Parse a persona counts string like "momentum=10,value=10,noise=10" into a dict.

    Args:
        counts_str: A comma-separated string of persona=count pairs.

    Returns:
        A dict mapping persona name -> count.

    Why this matters:
        A simple string format is easier to type on the command line than JSON.
    """
    result = {}
    for part in counts_str.split(","):
        part = part.strip()
        if "=" not in part:
            raise ValueError(f"Invalid persona count specification: '{part}'. Use 'persona=count'.")
        persona, count_str = part.split("=", 1)
        persona = persona.strip()
        count = int(count_str.strip())
        if persona not in ("momentum", "value", "noise"):
            raise ValueError(f"Unknown persona: '{persona}'. Must be momentum, value, or noise.")
        result[persona] = count
    return result


def main() -> None:
    """
    CLI entry point for the simulation orchestrator.

    Args:
        None (reads from sys.argv via argparse).

    Returns:
        None.

    Why this matters:
        The CLI makes it easy to run all necessary pre-computations with single commands
        before a talk or demo, without needing to open a notebook.
    """
    parser = argparse.ArgumentParser(
        description="Run trader persona market simulations."
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--hero",
        action="store_true",
        help="Run the three hero experiment conditions and save to eval/results/hero_experiment.json.",
    )
    mode_group.add_argument(
        "--precompute",
        action="store_true",
        help="Run 6 predefined compositions and save to eval/results/simulations/.",
    )
    mode_group.add_argument(
        "--persona-counts",
        type=str,
        metavar="COUNTS",
        help=(
            "Run a custom simulation. Specify as 'momentum=N,value=N,noise=N'. "
            "Example: --persona-counts momentum=10,value=10,noise=10"
        ),
    )

    parser.add_argument(
        "--n-ticks",
        type=int,
        default=500,
        help="Number of simulation ticks (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the simulation (default: 42). Used only with --persona-counts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path. Used only with --persona-counts (default: eval/results/custom_run.json).",
    )

    args = parser.parse_args()

    if args.hero:
        output_path = Path("eval/results/hero_experiment.json")
        run_hero_experiment(output_path, n_ticks=args.n_ticks)

    elif args.precompute:
        output_dir = Path("eval/results/simulations")
        run_precompute_library(output_dir, n_ticks=args.n_ticks)

    elif args.persona_counts:
        try:
            persona_counts = parse_persona_counts(args.persona_counts)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"Running custom simulation: {persona_counts}")
        agents = create_agents(persona_counts, seed_offset=args.seed)
        log = run_simulation(
            agents,
            n_ticks=args.n_ticks,
            initial_price=100.0,
            fair_value=100.0,
            seed=args.seed,
        )

        output_path = Path(args.output) if args.output else Path("eval/results/custom_run.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata fields for the app.
        n_agents = sum(persona_counts.values())
        composition_str = " / ".join(f"{v} {k.capitalize()}" for k, v in persona_counts.items())
        log["name"] = f"{composition_str} — seed {args.seed}"

        with open(output_path, "w") as f:
            json.dump(log, f, indent=2)

        print(f"\nSimulation complete.")
        print(f"  Final price: {log['market_summary']['final_price']:.2f}")
        print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
