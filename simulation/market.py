"""
market.py — Minimal order-driven market for trader persona simulation.

Talk section: Section III — The Proposed Framework

Purpose:
    Implements a single-asset order-driven market with a simple price update rule.
    This module is intentionally minimal: it exists to create a feedback loop between
    agent decisions and price, not to model any realistic market microstructure.

Key design decisions:
    - Price update is a deterministic function of aggregate order imbalance plus a small
      noise term. This is the simplest possible model that captures the core feedback:
      more buyers than sellers → price rises. Richer models (LOB, market impact functions,
      spread) would obscure the behavioral signal we are trying to demonstrate.
    - We track the full price history and order flow history so that the simulation log
      is complete and self-contained after the run.
    - News strings are generated procedurally from price movement templates. This avoids
      the need for an external data source and keeps the simulation self-contained.
    - There is no concept of "fills" or partial execution. An order of quantity Q always
      executes in full. This is a deliberate simplification.

Price update formula:
    price_{t+1} = price_t * (1 + alpha * order_imbalance_t + epsilon_t)
    where:
        order_imbalance = (buy_volume - sell_volume) / max(total_volume, 1)
        alpha = 0.01  (price impact coefficient)
        epsilon ~ N(0, 0.001)  (exogenous noise)
"""

import random
from typing import Optional


# ---------------------------------------------------------------------------
# News template system
# ---------------------------------------------------------------------------

# News strings are rotated based on recent price movement.
# This gives agents a plausible narrative to react to each tick.

_NEWS_POSITIVE = [
    "Buying interest remains strong on continued momentum.",
    "Market sentiment turns optimistic as prices advance.",
    "Volume confirms upward price move; bulls in control.",
    "Technical breakout observed; trend-followers pile in.",
    "Strong demand absorbs all available supply at current levels.",
]

_NEWS_NEGATIVE = [
    "Selling pressure intensifies as prices decline.",
    "Market sentiment deteriorates; bears gaining control.",
    "Volume confirms downward move; support levels tested.",
    "Technical breakdown observed; stop-losses triggered.",
    "Supply overwhelms demand at current price levels.",
]

_NEWS_NEUTRAL = [
    "Market activity subdued; no clear directional bias.",
    "Mixed signals from order flow; price action indecisive.",
    "Equilibrium conditions persist; balanced buy/sell activity.",
    "Low volatility environment; traders await catalyst.",
    "No significant order flow imbalance observed.",
]


def _generate_news(price_change_pct: float) -> str:
    """
    Select a news string based on the magnitude and direction of recent price change.

    Args:
        price_change_pct: The percentage price change over the last few ticks.

    Returns:
        A news string (str).

    Why this matters:
        News provides context for the noise trader's decisions. By correlating news
        sentiment with recent price moves, we create a plausible feedback narrative.
    """
    if price_change_pct > 0.3:
        return random.choice(_NEWS_POSITIVE)
    elif price_change_pct < -0.3:
        return random.choice(_NEWS_NEGATIVE)
    else:
        return random.choice(_NEWS_NEUTRAL)


# ---------------------------------------------------------------------------
# Market class
# ---------------------------------------------------------------------------


class Market:
    """
    A minimal single-asset order-driven market.

    Agents submit orders each tick. The market aggregates them, updates the price,
    and provides the new market state to agents for the next tick.

    Attributes:
        initial_price: The starting price of the asset.
        alpha: Price impact coefficient. Higher values mean larger price moves
               per unit of order imbalance.
        noise_std: Standard deviation of the exogenous noise term (epsilon).
        fair_value: The "true" fundamental value of the asset. Provided to agents
                    as a signal; the market itself does not enforce it.
        price_history: Full list of prices from tick 0 onwards.
        order_flow_history: List of per-tick order flow dicts for logging.
        news_history: List of news strings, one per tick.
        rng: A seeded Random instance for reproducible simulations.
    """

    def __init__(
        self,
        initial_price: float = 100.0,
        alpha: float = 0.01,
        noise_std: float = 0.001,
        fair_value: float = 100.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the market.

        Args:
            initial_price: Starting price for the asset.
            alpha: Price impact coefficient (how strongly order imbalance moves price).
            noise_std: Standard deviation of the exogenous Gaussian noise term.
            fair_value: The fundamental value of the asset (constant in v1).
            seed: Optional random seed for reproducibility.

        Returns:
            None.

        Why this matters:
            The seed is essential for reproducible hero experiment results. The same seed
            produces the same epsilon sequence, so differences across conditions are due
            purely to agent composition, not to random variation.
        """
        self.initial_price = initial_price
        self.alpha = alpha
        self.noise_std = noise_std
        self.fair_value = fair_value

        self.price_history: list[float] = [initial_price]
        self.order_flow_history: list[dict] = []
        self.news_history: list[str] = []

        # SIMPLIFICATION: We use Python's built-in random module for simplicity.
        # A production version would use numpy's Generator for better statistical properties.
        self.rng = random.Random(seed)

        # Seed Python's global random state too (used by news generation).
        if seed is not None:
            random.seed(seed)

    def step(self, orders: list[dict]) -> float:
        """
        Advance the market by one tick.

        Takes all agent orders, computes the aggregate order imbalance, updates the price,
        and records the tick in the history.

        Args:
            orders: A list of order dicts. Each dict has:
                    - "action": "BUY", "SELL", or "HOLD"
                    - "quantity": integer >= 0
                    - "persona": optional persona label for logging

        Returns:
            The new price after this tick (float).

        Why this matters:
            This is the core market mechanism. The simplicity here is intentional:
            the only feedback channel is aggregate order imbalance. This isolates the
            behavioral signal from microstructural noise.
        """
        buy_volume = 0
        sell_volume = 0
        persona_volumes: dict[str, dict[str, int]] = {}

        for order in orders:
            action = order.get("action", "HOLD")
            quantity = order.get("quantity", 0)
            persona = order.get("persona", "unknown")

            if persona not in persona_volumes:
                persona_volumes[persona] = {"buy": 0, "sell": 0, "hold": 0}

            if action == "BUY":
                buy_volume += quantity
                persona_volumes[persona]["buy"] += quantity
            elif action == "SELL":
                sell_volume += quantity
                persona_volumes[persona]["sell"] += quantity
            else:
                persona_volumes[persona]["hold"] += 1

        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            # No active trading this tick; price does not move from order flow.
            order_imbalance = 0.0
        else:
            order_imbalance = (buy_volume - sell_volume) / total_volume

        # Gaussian noise term.
        # SIMPLIFICATION: We sample from a Gaussian approximated by summing 12 uniform
        # random variables (Box-Muller would be cleaner but this is readable).
        epsilon = self._sample_gaussian(mean=0.0, std=self.noise_std)

        current_price = self.price_history[-1]
        new_price = current_price * (1 + self.alpha * order_imbalance + epsilon)
        new_price = round(max(new_price, 0.01), 4)  # Price cannot go below 0.01

        self.price_history.append(new_price)

        # Compute recent price change for news generation.
        if len(self.price_history) >= 4:
            recent_change_pct = (
                (self.price_history[-1] - self.price_history[-4]) / self.price_history[-4] * 100
            )
        else:
            recent_change_pct = 0.0

        news = _generate_news(recent_change_pct)
        self.news_history.append(news)

        # Record order flow for this tick.
        self.order_flow_history.append(
            {
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "total_volume": total_volume,
                "order_imbalance": round(order_imbalance, 4),
                "epsilon": round(epsilon, 6),
                "persona_volumes": persona_volumes,
            }
        )

        return new_price

    def get_state(self) -> dict:
        """
        Return the current observable market state.

        The state dict is what agents receive as input each tick. It contains everything
        a trader would observe in this minimal market: recent prices, the latest news,
        and the fair value signal.

        Args:
            None.

        Returns:
            A dict with keys:
                - "price_history": last 5 prices (list of floats)
                - "current_price": the most recent price (float)
                - "fair_value": the fundamental value (float)
                - "news": the most recent news string (str)
                - "tick": the current tick number (int)

        Why this matters:
            Agents only see this state dict — not the internal market object. This enforces
            information symmetry and keeps agent logic decoupled from market internals.
        """
        history_window = self.price_history[-5:] if len(self.price_history) >= 5 else self.price_history[:]
        latest_news = self.news_history[-1] if self.news_history else "Market just opened."
        tick = len(self.price_history) - 1  # Number of completed ticks

        return {
            "price_history": history_window,
            "current_price": self.price_history[-1],
            "fair_value": self.fair_value,
            "news": latest_news,
            "tick": tick,
        }

    def _sample_gaussian(self, mean: float, std: float) -> float:
        """
        Sample from a Gaussian distribution using the Box-Muller transform.

        Args:
            mean: Mean of the distribution.
            std: Standard deviation of the distribution.

        Returns:
            A single float sample.

        Why this matters:
            Using the seeded self.rng (not the global random state) ensures that the
            noise sequence is reproducible and isolated from other random calls.
        """
        # Box-Muller transform: converts two uniform samples to one Gaussian sample.
        u1 = self.rng.random()
        u2 = self.rng.random()

        # Avoid log(0).
        u1 = max(u1, 1e-10)

        import math
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mean + std * z

    def get_summary(self) -> dict:
        """
        Return a summary of the full simulation run.

        Args:
            None.

        Returns:
            A dict with summary statistics: initial price, final price, min price,
            max price, total ticks, total buy/sell volume.

        Why this matters:
            The summary is logged alongside full tick data in the simulation results
            file, giving a quick overview without needing to parse the full history.
        """
        if len(self.price_history) < 2:
            return {}

        total_buy = sum(tick["buy_volume"] for tick in self.order_flow_history)
        total_sell = sum(tick["sell_volume"] for tick in self.order_flow_history)

        return {
            "initial_price": self.price_history[0],
            "final_price": self.price_history[-1],
            "min_price": min(self.price_history),
            "max_price": max(self.price_history),
            "total_ticks": len(self.price_history) - 1,
            "total_buy_volume": total_buy,
            "total_sell_volume": total_sell,
        }
