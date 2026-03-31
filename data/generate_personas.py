"""
generate_personas.py — Synthetic training data generation for trader personas.

Talk section: Section III — The Proposed Framework

Purpose:
    Uses the Anthropic API (claude-sonnet-4-6) to generate chain-of-thought reasoning traces
    for three trader personas: momentum, value, and noise. Each example consists of a market
    state, an in-persona reasoning chain, a trading action (BUY/SELL/HOLD), and a quantity.

    The output is saved to data/personas/{persona}.jsonl, one JSON object per line.

    With the --export-examples flag, also generates eval/results/persona_examples.json —
    a file containing curated cross-persona comparisons (same market state, all three personas).
    This file powers the Persona Inspector page in the Streamlit app.

Key design decisions:
    - We use a single API call per example (not batched) so that each reasoning trace is
      independent and the model does not see previous examples in context. This reduces
      correlation between examples.
    - We provide the persona identity string as a system prompt. This is the same identity
      string that will be used during simulation inference, ensuring train/test consistency.
    - We ask the model to produce structured output in a specific JSON format. We then
      parse and validate it before saving. Malformed outputs are logged and skipped.
    - Market scenarios are drawn from a predefined set of templates covering the major
      market conditions we want the agents to handle in simulation.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------

PERSONA_IDENTITIES = {
    "momentum": (
        "You are a momentum trader. You believe recent price trends persist in the short run. "
        "You buy assets that have risen recently and sell assets that have fallen. "
        "You do not anchor to fundamentals."
    ),
    "value": (
        "You are a value investor. You estimate the intrinsic value of an asset from its "
        "fundamentals. You buy when the price is significantly below fair value and sell when "
        "it exceeds it. You are not influenced by short-term price momentum."
    ),
    "noise": (
        "You are a noise trader. You react to market news and social sentiment. Your decisions "
        "are influenced by recent headlines and what you believe other traders are thinking. "
        "You trade frequently and are prone to overconfidence."
    ),
}

# ---------------------------------------------------------------------------
# Market scenario templates
# These define the space of situations the agents will face in simulation.
# Each scenario is a (price_history, news, fair_value) tuple.
# We generate many variants by parameterizing these templates.
# ---------------------------------------------------------------------------

SCENARIO_TYPES = [
    "strong_uptrend",
    "moderate_uptrend",
    "strong_downtrend",
    "moderate_downtrend",
    "price_below_fair_value",
    "price_above_fair_value",
    "positive_news_flat_price",
    "negative_news_flat_price",
    "neutral_at_fair_value",
    "volatile_choppy",
    "recovery_after_dip",
    "reversal_after_spike",
]

NEWS_TEMPLATES = {
    "positive": [
        "Strong quarterly earnings beat analyst expectations.",
        "Company announces record revenue and raises full-year guidance.",
        "Major institutional investor discloses a large new stake.",
        "Product launch receives enthusiastic market reception.",
        "Industry data shows sector-wide demand acceleration.",
    ],
    "negative": [
        "Earnings miss amid weakening demand signals.",
        "Analyst downgrades stock citing margin compression.",
        "Regulatory investigation announced into core business.",
        "Key executive departure raises strategic uncertainty.",
        "Sector-wide selloff accelerates on macro concerns.",
    ],
    "neutral": [
        "No significant company news today.",
        "Mixed economic data; market participants await clarity.",
        "Trading volume below average; no major catalysts.",
        "Quarterly earnings in line with expectations.",
        "Modest sector rotation with no clear direction.",
    ],
}


def generate_price_history(scenario_type: str, base_price: float = 100.0) -> list[float]:
    """
    Generate a 5-period price history consistent with a given scenario type.

    Args:
        scenario_type: One of the SCENARIO_TYPES strings.
        base_price: The starting price before the trend begins.

    Returns:
        A list of 5 prices (floats, rounded to 2 decimal places).

    Why this matters:
        The price history is the primary input for momentum and value decisions.
        Consistent, realistic histories are essential for training useful agents.
    """
    if scenario_type == "strong_uptrend":
        deltas = [0.02, 0.025, 0.02, 0.03, 0.025]
    elif scenario_type == "moderate_uptrend":
        deltas = [0.01, 0.008, 0.012, 0.009, 0.011]
    elif scenario_type == "strong_downtrend":
        deltas = [-0.02, -0.025, -0.02, -0.03, -0.025]
    elif scenario_type == "moderate_downtrend":
        deltas = [-0.01, -0.008, -0.012, -0.009, -0.011]
    elif scenario_type == "price_below_fair_value":
        # Flat or slight downtrend, ending well below fair value
        deltas = [-0.015, -0.01, -0.008, -0.012, -0.005]
    elif scenario_type == "price_above_fair_value":
        # Uptrend ending well above fair value
        deltas = [0.015, 0.01, 0.012, 0.008, 0.01]
    elif scenario_type == "positive_news_flat_price":
        deltas = [0.001, -0.001, 0.002, 0.0, 0.001]
    elif scenario_type == "negative_news_flat_price":
        deltas = [-0.001, 0.001, -0.002, 0.0, -0.001]
    elif scenario_type == "neutral_at_fair_value":
        deltas = [0.001, -0.001, 0.0, 0.001, -0.001]
    elif scenario_type == "volatile_choppy":
        deltas = [0.02, -0.025, 0.03, -0.02, 0.015]
    elif scenario_type == "recovery_after_dip":
        deltas = [-0.03, -0.02, 0.01, 0.025, 0.02]
    elif scenario_type == "reversal_after_spike":
        deltas = [0.03, 0.025, 0.01, -0.02, -0.025]
    else:
        deltas = [0.0, 0.0, 0.0, 0.0, 0.0]

    prices = [base_price]
    for delta in deltas:
        prices.append(round(prices[-1] * (1 + delta), 2))

    return prices[1:]  # Return only the last 5 prices (history window)


def generate_fair_value(price_history: list[float], scenario_type: str) -> float:
    """
    Generate a fair value estimate consistent with the scenario.

    Args:
        price_history: The 5-period price history.
        scenario_type: The scenario type, used to place fair value meaningfully.

    Returns:
        A fair value (float, rounded to 2 decimal places).

    Why this matters:
        Fair value is the key signal for value investors. It must be placed meaningfully
        relative to the current price so that the agent has a clear decision signal.
    """
    current_price = price_history[-1]

    if scenario_type == "price_below_fair_value":
        # Fair value significantly above current price
        fair_value = round(current_price * random.uniform(1.08, 1.15), 2)
    elif scenario_type == "price_above_fair_value":
        # Fair value significantly below current price
        fair_value = round(current_price * random.uniform(0.86, 0.93), 2)
    elif scenario_type in ("strong_uptrend", "moderate_uptrend", "recovery_after_dip"):
        # Price may have overshot fair value
        fair_value = round(current_price * random.uniform(0.92, 1.02), 2)
    elif scenario_type in ("strong_downtrend", "moderate_downtrend", "reversal_after_spike"):
        # Price may be below fair value
        fair_value = round(current_price * random.uniform(0.98, 1.08), 2)
    else:
        # Near fair value
        fair_value = round(current_price * random.uniform(0.97, 1.03), 2)

    return fair_value


def generate_news(scenario_type: str) -> str:
    """
    Pick a news string appropriate for the scenario type.

    Args:
        scenario_type: The scenario type string.

    Returns:
        A news headline string.

    Why this matters:
        News is the primary decision input for noise traders. Its sentiment should
        roughly align with the price direction, as in real markets.
    """
    if "uptrend" in scenario_type or scenario_type in ("recovery_after_dip", "positive_news_flat_price"):
        return random.choice(NEWS_TEMPLATES["positive"])
    elif "downtrend" in scenario_type or scenario_type in ("reversal_after_spike", "negative_news_flat_price"):
        return random.choice(NEWS_TEMPLATES["negative"])
    else:
        return random.choice(NEWS_TEMPLATES["neutral"])


def build_generation_prompt(persona: str, market_state: dict) -> str:
    """
    Build the user-turn prompt for generating one training example.

    Args:
        persona: One of "momentum", "value", "noise".
        market_state: A dict with keys: price_history, news, fair_value.

    Returns:
        A prompt string to send as the user turn to the API.

    Why this matters:
        The prompt format must exactly mirror what will be used at simulation inference time.
        Train/test prompt consistency is critical for the fine-tuned adapter to generalize.
    """
    price_history = market_state["price_history"]
    news = market_state["news"]
    fair_value = market_state["fair_value"]
    current_price = price_history[-1]
    price_change = round((current_price - price_history[0]) / price_history[0] * 100, 2)

    prompt = f"""You are analyzing a financial market. Here is the current market state:

Price history (last 5 periods): {price_history}
Current price: {current_price}
Price change over window: {price_change}%
Latest news: {news}
Estimated fair value: {fair_value}

Based on your identity as a trader, reason step-by-step about what you observe and what action you will take.

Then provide your decision in exactly this JSON format (and nothing else after it):
{{
  "reasoning": "<your step-by-step reasoning, 2-4 sentences>",
  "action": "<BUY or SELL or HOLD>",
  "quantity": <integer from 1 to 20, or 0 if HOLD>
}}

Remember: you must reason in character. Do not break character or hedge."""

    return prompt


def call_api_for_example(
    client: anthropic.Anthropic,
    persona: str,
    market_state: dict,
) -> dict | None:
    """
    Call the Anthropic API to generate one training example for a given persona.

    Args:
        client: An initialized Anthropic client.
        persona: One of "momentum", "value", "noise".
        market_state: A dict with keys: price_history, news, fair_value.

    Returns:
        A complete training example dict, or None if the API call fails or output is malformed.

    Why this matters:
        We validate the parsed output strictly before saving. A malformed example is worse
        than no example because it introduces noise into the training data.
    """
    system_prompt = PERSONA_IDENTITIES[persona]
    user_prompt = build_generation_prompt(persona, market_state)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = message.content[0].text.strip()
    except anthropic.APIError as exc:
        print(f"  [API error] {exc}", file=sys.stderr)
        return None

    # Parse the JSON block from the response.
    # The model is instructed to end with the JSON; we find the last '{' to be robust.
    json_start = raw_text.rfind("{")
    json_end = raw_text.rfind("}") + 1
    if json_start == -1 or json_end == 0:
        print(f"  [parse error] No JSON found in response: {raw_text[:100]}", file=sys.stderr)
        return None

    try:
        parsed = json.loads(raw_text[json_start:json_end])
    except json.JSONDecodeError as exc:
        print(f"  [parse error] {exc}: {raw_text[json_start:json_end][:100]}", file=sys.stderr)
        return None

    # Validate required fields and types.
    if parsed.get("action") not in ("BUY", "SELL", "HOLD"):
        print(f"  [validation error] Invalid action: {parsed.get('action')}", file=sys.stderr)
        return None

    quantity = parsed.get("quantity", 0)
    if not isinstance(quantity, int) or quantity < 0 or quantity > 20:
        print(f"  [validation error] Invalid quantity: {quantity}", file=sys.stderr)
        return None

    if "HOLD" == parsed["action"] and quantity != 0:
        # Normalize: HOLD always has quantity 0.
        parsed["quantity"] = 0

    return {
        "persona": persona,
        "market_state": market_state,
        "reasoning": parsed.get("reasoning", ""),
        "action": parsed["action"],
        "quantity": parsed["quantity"],
    }


def generate_persona_data(
    persona: str,
    n_examples: int,
    output_path: Path,
    client: anthropic.Anthropic,
) -> None:
    """
    Generate n_examples training examples for a persona and append to output_path.

    Args:
        persona: One of "momentum", "value", "noise".
        n_examples: How many examples to generate.
        output_path: Path to the .jsonl file to write examples into.
        client: An initialized Anthropic client.

    Returns:
        None. Writes directly to output_path.

    Why this matters:
        We append (not overwrite) so that interrupted runs can be resumed without losing
        already-generated data.
    """
    scenario_types = SCENARIO_TYPES
    base_prices = [90.0, 100.0, 110.0, 120.0, 80.0, 95.0]

    generated = 0
    attempts = 0
    max_attempts = n_examples * 2  # Allow up to 2x attempts to handle failures

    print(f"Generating {n_examples} examples for persona '{persona}'...")

    with open(output_path, "a") as f:
        while generated < n_examples and attempts < max_attempts:
            attempts += 1

            # Cycle through scenario types and base prices for variety.
            scenario_type = scenario_types[generated % len(scenario_types)]
            base_price = base_prices[generated % len(base_prices)]

            # Small random perturbation so examples within the same scenario type differ.
            base_price_jittered = base_price * random.uniform(0.95, 1.05)

            price_history = generate_price_history(scenario_type, base_price_jittered)
            fair_value = generate_fair_value(price_history, scenario_type)
            news = generate_news(scenario_type)

            market_state = {
                "price_history": price_history,
                "news": news,
                "fair_value": fair_value,
            }

            example = call_api_for_example(client, persona, market_state)

            if example is not None:
                f.write(json.dumps(example) + "\n")
                f.flush()
                generated += 1
                print(f"  [{generated}/{n_examples}] {example['action']} qty={example['quantity']}")
            else:
                print(f"  [attempt {attempts}] Failed — will retry with next scenario")

            # Respect API rate limits with a small delay.
            # SIMPLIFICATION: A production version would use exponential backoff with jitter.
            time.sleep(0.2)

    print(f"Done. Generated {generated} examples for '{persona}' -> {output_path}")


def export_persona_examples(client: anthropic.Anthropic, output_path: Path) -> None:
    """
    Generate a curated cross-persona comparison file for the Streamlit app.

    For each of a set of named scenarios, we generate one reasoning trace per persona
    using exactly the same market state. This lets the app show side-by-side comparisons
    of how different personas respond to identical information.

    Args:
        client: An initialized Anthropic client.
        output_path: Path to write the JSON output file.

    Returns:
        None. Writes directly to output_path.

    Why this matters:
        The side-by-side comparison is the key demo moment: same inputs, three completely
        different conclusions. This file is what makes that possible.
    """
    # Curated scenarios with human-readable labels.
    # Each scenario is fully specified (not randomly generated) for reproducibility.
    curated_scenarios = [
        {
            "label": "Strong uptrend, bullish news",
            "description": "Price has risen 10% over 5 periods. Earnings beat announced.",
            "market_state": {
                "price_history": [100.0, 102.0, 105.0, 107.0, 110.0],
                "news": "Strong quarterly earnings beat analyst expectations.",
                "fair_value": 104.0,
            },
        },
        {
            "label": "Strong downtrend, negative news",
            "description": "Price has fallen 10% over 5 periods. Analyst downgrade released.",
            "market_state": {
                "price_history": [110.0, 107.0, 105.0, 102.0, 99.0],
                "news": "Analyst downgrades stock citing margin compression.",
                "fair_value": 106.0,
            },
        },
        {
            "label": "Price significantly below fair value",
            "description": "Price at 92, fair value at 110. No strong trend.",
            "market_state": {
                "price_history": [98.0, 96.0, 94.0, 93.0, 92.0],
                "news": "Mixed economic data; market participants await clarity.",
                "fair_value": 110.0,
            },
        },
        {
            "label": "Price significantly above fair value",
            "description": "Price at 118, fair value at 100. Recent momentum positive.",
            "market_state": {
                "price_history": [108.0, 112.0, 114.0, 116.0, 118.0],
                "news": "Modest sector rotation with no clear direction.",
                "fair_value": 100.0,
            },
        },
        {
            "label": "Flat market, positive news surprise",
            "description": "Price flat at 100. Unexpected positive catalyst arrives.",
            "market_state": {
                "price_history": [100.0, 100.2, 99.8, 100.1, 100.0],
                "news": "Major institutional investor discloses a large new stake.",
                "fair_value": 101.0,
            },
        },
        {
            "label": "Flat market, negative news surprise",
            "description": "Price flat at 100. Unexpected negative catalyst arrives.",
            "market_state": {
                "price_history": [100.0, 99.8, 100.2, 99.9, 100.0],
                "news": "Regulatory investigation announced into core business.",
                "fair_value": 99.0,
            },
        },
        {
            "label": "Recovery after sharp dip",
            "description": "Price dropped then recovered. Neutral news.",
            "market_state": {
                "price_history": [105.0, 98.0, 95.0, 99.0, 103.0],
                "news": "Trading volume below average; no major catalysts.",
                "fair_value": 104.0,
            },
        },
        {
            "label": "Reversal after sharp spike",
            "description": "Price spiked then pulled back. Sentiment mixed.",
            "market_state": {
                "price_history": [95.0, 103.0, 108.0, 104.0, 100.0],
                "news": "Mixed economic data; market participants await clarity.",
                "fair_value": 97.0,
            },
        },
        {
            "label": "Choppy volatile market",
            "description": "Large swings in both directions. Uncertainty high.",
            "market_state": {
                "price_history": [100.0, 106.0, 98.0, 104.0, 97.0],
                "news": "Earnings miss amid weakening demand signals.",
                "fair_value": 100.0,
            },
        },
        {
            "label": "Neutral market at fair value",
            "description": "Price exactly at fair value, no news, no trend.",
            "market_state": {
                "price_history": [100.0, 100.5, 99.8, 100.2, 100.0],
                "news": "No significant company news today.",
                "fair_value": 100.0,
            },
        },
    ]

    personas = ["momentum", "value", "noise"]
    results = []

    for scenario in curated_scenarios:
        print(f"Generating comparisons for scenario: {scenario['label']}")
        scenario_result = {
            "label": scenario["label"],
            "description": scenario["description"],
            "market_state": scenario["market_state"],
            "personas": {},
        }

        for persona in personas:
            print(f"  -> {persona}...")
            example = call_api_for_example(client, persona, scenario["market_state"])
            if example is not None:
                scenario_result["personas"][persona] = {
                    "reasoning": example["reasoning"],
                    "action": example["action"],
                    "quantity": example["quantity"],
                }
            else:
                # If the API call fails, insert a placeholder so the file is still valid.
                scenario_result["personas"][persona] = {
                    "reasoning": "[Generation failed — re-run export to populate this entry.]",
                    "action": "HOLD",
                    "quantity": 0,
                }
            time.sleep(0.3)

        results.append(scenario_result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Exported {len(results)} scenarios to {output_path}")


def main() -> None:
    """
    CLI entry point for persona data generation.

    Args:
        None (reads from sys.argv via argparse).

    Returns:
        None.

    Why this matters:
        A clean CLI makes it easy to run data generation as a batch job on a remote
        machine and to incrementally add more examples without rewriting existing files.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic persona training data using the Anthropic API."
    )
    parser.add_argument(
        "--persona",
        type=str,
        choices=["momentum", "value", "noise"],
        default=None,
        help="Which persona to generate data for. If omitted with --export-examples, generates all.",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=300,
        help="Number of training examples to generate per persona (default: 300).",
    )
    parser.add_argument(
        "--export-examples",
        action="store_true",
        help=(
            "Export eval/results/persona_examples.json with curated cross-persona comparisons "
            "for the Streamlit app. Does not generate training data."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/personas",
        help="Directory to write .jsonl files into (default: data/personas).",
    )

    args = parser.parse_args()

    # Retrieve the API key from the environment.
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "Error: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it with: export ANTHROPIC_API_KEY=your_key_here",
            file=sys.stderr,
        )
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    if args.export_examples:
        output_path = Path("eval/results/persona_examples.json")
        export_persona_examples(client, output_path)
        return

    # Generate training data for one or all personas.
    personas_to_generate = [args.persona] if args.persona else ["momentum", "value", "noise"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for persona in personas_to_generate:
        output_path = output_dir / f"{persona}.jsonl"
        generate_persona_data(persona, args.n_examples, output_path, client)


if __name__ == "__main__":
    random.seed(42)
    main()
