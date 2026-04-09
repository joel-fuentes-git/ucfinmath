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
    - Uses SLMAgent (the fine-tuned LoRA SLM) when adapters are available on disk;
      automatically falls back to RuleBasedAgent (hand-coded heuristic) when they
      are not. Each persona is decided independently: a missing momentum adapter
      forces momentum agents onto the rule-based path while value/noise can still
      use their adapters.
    - Adapters are looked up under `<project_root>/adapters/<persona>/` by default.
      Override with --adapters-dir or set DEFAULT_ADAPTERS_DIR below.
    - The agent type actually used (per persona) is stamped into the saved JSON
      metadata under `agent_types`, so the Streamlit app and notebook can faithfully
      report which model produced any given run without re-deriving it.
    - Each tick dispatches decisions through simulation.agent.batched_decide(), which
      pools all SLMAgents sharing a loaded adapter into a single model.generate() call.
      This is the dominant wall-clock win for SLM-backed runs: a 30-agent homogeneous
      population runs 1 generate per tick instead of 30.
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
from typing import Optional

# Add the project root to sys.path so we can import simulation and eval modules
# regardless of where the script is called from.
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from simulation.market import Market
from simulation.agent import RuleBasedAgent, SLMAgent, batched_decide

# Default location for fine-tuned LoRA adapters. Each persona is expected to live
# in a subdirectory: adapters/momentum/, adapters/value/, adapters/noise/. Each
# subdirectory must contain at minimum an `adapter_config.json` (PEFT format).
# See README.md for the download link to the pre-trained adapters used in the talk.
DEFAULT_ADAPTERS_DIR = _project_root / "adapters"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


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

        # Collect all agent decisions for this tick. batched_decide() pools
        # every SLMAgent sharing a loaded adapter into a single model.generate()
        # call, so a 30-agent homogeneous population runs one generate per tick
        # and a 10/10/10 mixed population runs three (one per persona adapter),
        # instead of the 30 per-tick generates the naive per-agent loop did.
        # RuleBasedAgents are still dispatched individually on their fast path.
        tick_decisions = batched_decide(agents, market_state)

        tick_orders = [
            {
                "action": decision["action"],
                "quantity": decision["quantity"],
                "persona": decision["persona"],
            }
            for decision in tick_decisions
        ]

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


def _resolve_agent_class_for_persona(
    persona: str,
    adapters_dir: Optional[Path],
    base_model: str,
    force_rules: bool,
    require_adapters: bool,
) -> tuple[str, str]:
    """
    Decide whether a given persona will use SLMAgent or RuleBasedAgent.

    Args:
        persona: The persona label ("momentum", "value", "noise").
        adapters_dir: Root directory containing per-persona adapter subdirectories.
                      May be None when force_rules is set.
        base_model: HuggingFace name of the base model the adapters were trained against.
        force_rules: If True, always return ("rules", reason) regardless of adapter state.
        require_adapters: If True, raise instead of falling back when an adapter is missing.

    Returns:
        A tuple ("slm", "<resolved adapter path>") if the persona should use SLMAgent,
        or ("rules", "<short reason>") if it should fall back to RuleBasedAgent.

    Why this matters:
        Per-persona resolution lets a partial set of adapters work — e.g. if you have
        downloaded only the momentum adapter, momentum agents use the SLM and the other
        two personas fall back to rules — rather than forcing an all-or-nothing choice.
    """
    if force_rules:
        return "rules", "forced by --force-rules"
    if adapters_dir is None:
        return "rules", "no adapters directory configured"

    candidate_path = adapters_dir / persona
    ok, reason = SLMAgent.is_available(str(candidate_path))
    if ok:
        return "slm", str(candidate_path)

    if require_adapters:
        raise FileNotFoundError(
            f"Adapter for persona '{persona}' not usable: {reason}\n"
            f"Run without --require-adapters to fall back to RuleBasedAgent for this persona."
        )

    return "rules", reason


def create_agents(
    persona_counts: dict[str, int],
    seed_offset: int = 0,
    adapters_dir: Optional[Path] = DEFAULT_ADAPTERS_DIR,
    base_model: str = DEFAULT_BASE_MODEL,
    force_rules: bool = False,
    require_adapters: bool = False,
) -> tuple[list, dict[str, str]]:
    """
    Create a population of trader agents from a persona count dict.

    For each persona, this tries to instantiate SLMAgent against the adapter at
    `<adapters_dir>/<persona>/`. If that adapter is not present (or fails to load),
    the persona automatically falls back to RuleBasedAgent. The decision is per-persona,
    so a partial set of adapters works: missing personas use rules, present personas
    use the fine-tuned SLM.

    Args:
        persona_counts: A dict mapping persona name -> number of agents.
                        E.g., {"momentum": 10, "value": 10, "noise": 10}
        seed_offset: An integer offset added to each agent's seed to ensure
                     different agents have independent random states (used by
                     RuleBasedAgent's noise trader).
        adapters_dir: Root directory containing per-persona LoRA adapter subdirectories.
                      Defaults to <project_root>/adapters. Pass None together with
                      force_rules=True to skip adapter detection entirely.
        base_model: HuggingFace name of the base model the adapters were trained against.
        force_rules: If True, every persona uses RuleBasedAgent regardless of whether
                     adapters are present.
        require_adapters: If True, raise FileNotFoundError when any persona's adapter
                          is missing instead of silently falling back.

    Returns:
        A (agents, agent_types) tuple. `agents` is the list of agent instances. `agent_types`
        is a dict mapping persona -> "slm" or "rules", describing which agent class was
        actually instantiated for that persona. The agent_types dict is what gets stamped
        into saved simulation metadata so downstream consumers (notebook, Streamlit) can
        faithfully report what was running.

    Why this matters:
        A factory function with built-in adapter detection means the rest of the
        codebase (the hero experiment, the notebook, the precompute script) doesn't
        need to know whether the SLM or the rule-based fallback is in play — it just
        gets a working population.
    """
    # First decide once per persona which class to use, so we can print a clear banner
    # before any heavy model loading begins.
    resolution: dict[str, tuple[str, str]] = {}
    for persona in persona_counts:
        resolution[persona] = _resolve_agent_class_for_persona(
            persona,
            adapters_dir=adapters_dir,
            base_model=base_model,
            force_rules=force_rules,
            require_adapters=require_adapters,
        )

    print("[create_agents] Agent resolution per persona:")
    for persona, (kind, detail) in resolution.items():
        if kind == "slm":
            print(f"  {persona:9s} -> SLMAgent  (adapter: {detail})")
        else:
            print(f"  {persona:9s} -> RuleBasedAgent  ({detail})")

    agents: list = []
    agent_types: dict[str, str] = {}
    agent_index = 0

    for persona, count in persona_counts.items():
        kind, detail = resolution[persona]

        if kind == "slm":
            try:
                for i in range(count):
                    agent_id = f"{persona}_{i:03d}"
                    agents.append(
                        SLMAgent(
                            persona=persona,
                            agent_id=agent_id,
                            adapter_path=detail,
                            base_model_name=base_model,
                        )
                    )
                    agent_index += 1
                agent_types[persona] = "slm"
                continue
            except Exception as exc:  # noqa: BLE001 — we explicitly want broad coverage
                if require_adapters:
                    raise
                print(
                    f"[create_agents] WARNING: SLMAgent load failed for persona "
                    f"'{persona}': {exc}. Falling back to RuleBasedAgent for this persona."
                )
                # Drop any partially-constructed agents for this persona before fallback.
                agents = [a for a in agents if a.persona != persona]
                agent_index -= count

        # Rule-based path (either chosen or post-failure fallback).
        for i in range(count):
            agent_id = f"{persona}_{i:03d}"
            agent_seed = seed_offset + agent_index * 31 + hash(agent_id) % 1000
            agents.append(RuleBasedAgent(persona, agent_id, seed=agent_seed))
            agent_index += 1
        agent_types[persona] = "rules"

    return agents, agent_types


# ---------------------------------------------------------------------------
# Precompute functions
# ---------------------------------------------------------------------------


def run_hero_experiment(
    output_path: Path,
    n_ticks: int = 500,
    adapters_dir: Optional[Path] = DEFAULT_ADAPTERS_DIR,
    base_model: str = DEFAULT_BASE_MODEL,
    force_rules: bool = False,
    require_adapters: bool = False,
) -> None:
    """
    Run the three hero experiment conditions and save results.

    The three conditions are:
    1. Homogeneous Momentum: 30 momentum traders
    2. Homogeneous Value: 30 value investors
    3. Mixed Population: 10 momentum + 10 value + 10 noise traders

    Args:
        output_path: Path to save the JSON results file.
        n_ticks: Number of simulation ticks (default: 500).
        adapters_dir: Root directory containing per-persona LoRA adapters.
                      Forwarded to create_agents.
        base_model: HuggingFace base model name for the adapters.
        force_rules: Force RuleBasedAgent for every persona.
        require_adapters: Raise instead of falling back when adapters are missing.

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

        agents, agent_types = create_agents(
            condition["persona_counts"],
            seed_offset=condition["seed"],
            adapters_dir=adapters_dir,
            base_model=base_model,
            force_rules=force_rules,
            require_adapters=require_adapters,
        )
        log = run_simulation(
            agents,
            n_ticks=n_ticks,
            initial_price=100.0,
            fair_value=100.0,
            seed=condition["seed"],
        )

        # Stamp the per-persona agent type into metadata so downstream consumers
        # know whether the SLM or rules produced this run.
        log["metadata"]["agent_types"] = agent_types

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
            f"N trades: {sum(log['agent_trade_counts'].values())}, "
            f"agent types: {agent_types}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nHero experiment results saved to: {output_path}")


def run_precompute_library(
    output_dir: Path,
    n_ticks: int = 500,
    adapters_dir: Optional[Path] = DEFAULT_ADAPTERS_DIR,
    base_model: str = DEFAULT_BASE_MODEL,
    force_rules: bool = False,
    require_adapters: bool = False,
) -> None:
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

        agents, agent_types = create_agents(
            composition["persona_counts"],
            seed_offset=composition["seed"],
            adapters_dir=adapters_dir,
            base_model=base_model,
            force_rules=force_rules,
            require_adapters=require_adapters,
        )
        log = run_simulation(
            agents,
            n_ticks=n_ticks,
            initial_price=100.0,
            fair_value=100.0,
            seed=composition["seed"],
        )
        log["metadata"]["agent_types"] = agent_types

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

        print(f"  Saved to: {out_path} (agent types: {agent_types})")

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

    # Adapter / agent-type controls. By default the orchestrator looks for LoRA
    # adapters under <project_root>/adapters/<persona> and uses them when present;
    # missing personas fall back to RuleBasedAgent.
    parser.add_argument(
        "--adapters-dir",
        type=str,
        default=str(DEFAULT_ADAPTERS_DIR),
        help=(
            "Directory containing LoRA adapter subdirectories named after each persona "
            "(momentum/, value/, noise/). Defaults to <project_root>/adapters. "
            "If a persona's subdirectory is missing or invalid, that persona "
            "automatically falls back to RuleBasedAgent."
        ),
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=(
            f"HuggingFace name of the base model the adapters were trained against "
            f"(default: {DEFAULT_BASE_MODEL})."
        ),
    )
    parser.add_argument(
        "--force-rules",
        action="store_true",
        help=(
            "Force RuleBasedAgent for every persona even if adapters are present. "
            "Use this to reproduce the rule-based baseline without removing adapter files."
        ),
    )
    parser.add_argument(
        "--require-adapters",
        action="store_true",
        help=(
            "Fail with an error instead of falling back when any persona's adapter is missing. "
            "Useful for catching configuration mistakes when you intend to run the SLM path."
        ),
    )

    args = parser.parse_args()

    adapters_dir: Optional[Path]
    if args.force_rules:
        # Even when forced to rules, we still pass the path through so logging
        # can reference it; the resolver will short-circuit to "rules".
        adapters_dir = Path(args.adapters_dir) if args.adapters_dir else None
    else:
        adapters_dir = Path(args.adapters_dir) if args.adapters_dir else None

    if args.hero:
        output_path = Path("eval/results/hero_experiment.json")
        run_hero_experiment(
            output_path,
            n_ticks=args.n_ticks,
            adapters_dir=adapters_dir,
            base_model=args.base_model,
            force_rules=args.force_rules,
            require_adapters=args.require_adapters,
        )

    elif args.precompute:
        output_dir = Path("eval/results/simulations")
        run_precompute_library(
            output_dir,
            n_ticks=args.n_ticks,
            adapters_dir=adapters_dir,
            base_model=args.base_model,
            force_rules=args.force_rules,
            require_adapters=args.require_adapters,
        )

    elif args.persona_counts:
        try:
            persona_counts = parse_persona_counts(args.persona_counts)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"Running custom simulation: {persona_counts}")
        agents, agent_types = create_agents(
            persona_counts,
            seed_offset=args.seed,
            adapters_dir=adapters_dir,
            base_model=args.base_model,
            force_rules=args.force_rules,
            require_adapters=args.require_adapters,
        )
        log = run_simulation(
            agents,
            n_ticks=args.n_ticks,
            initial_price=100.0,
            fair_value=100.0,
            seed=args.seed,
        )
        log["metadata"]["agent_types"] = agent_types

        output_path = Path(args.output) if args.output else Path("eval/results/custom_run.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata fields for the app.
        composition_str = " / ".join(f"{v} {k.capitalize()}" for k, v in persona_counts.items())
        log["name"] = f"{composition_str} — seed {args.seed}"

        with open(output_path, "w") as f:
            json.dump(log, f, indent=2)

        print(f"\nSimulation complete.")
        print(f"  Final price: {log['market_summary']['final_price']:.2f}")
        print(f"  Agent types: {agent_types}")
        print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
