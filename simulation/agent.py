"""
agent.py — Trader agents for the persona simulation.

Talk section: Section III — The Proposed Framework

Purpose:
    Defines two agent implementations:
    1. RuleBasedAgent: Implements each persona's logic as a simple hand-coded rule.
       This runs on CPU with no model weights required. It is the default for all
       notebooks, the hero experiment, and the Streamlit app.
    2. SLMAgent: Wraps a fine-tuned LoRA adapter and runs actual model inference.
       This requires a GPU (or at minimum a powerful CPU for quantized inference).

Key design decisions:
    - Both agents expose the same interface: decide(market_state) -> dict. This makes
      them interchangeable in the simulation without any changes to the orchestrator.
    - The RuleBasedAgent's rules are intentionally simple and transparent. They are
      meant to demonstrate the behavioral contrast between personas, not to be
      realistic trading strategies.
    - The SLMAgent includes a fallback to rule-based logic if model output is
      unparseable. This makes the simulation robust to occasional LLM failures.
    - All agents track their own cash, shares, trade history, and P&L. This keeps
      state encapsulated per-agent and makes the simulation easy to parallelize later.
    - Agents enforce a no-short-selling constraint: SELL is only executed if the agent
      currently holds shares. Similarly, BUY is only executed if the agent has cash.
"""

import json
import random
from abc import ABC, abstractmethod
from typing import Optional


# ---------------------------------------------------------------------------
# Persona identity strings (kept in sync with generate_personas.py and finetune.py)
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
# BaseAgent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """
    Abstract base class for all trader agents.

    Defines the interface that the simulation orchestrator expects from every agent.
    Subclasses must implement the decide() method.

    Attributes:
        persona: The persona label ("momentum", "value", or "noise").
        agent_id: A unique string identifier for this agent instance.
        cash: Current cash holdings (starts at INITIAL_CASH).
        shares: Current share holdings (starts at 0).
        trade_history: List of all executed trades (dicts).
        initial_cash: Starting cash (used to compute P&L).
    """

    INITIAL_CASH = 1000.0

    def __init__(self, persona: str, agent_id: str) -> None:
        """
        Initialize a trader agent.

        Args:
            persona: One of "momentum", "value", "noise".
            agent_id: A unique identifier string for this agent.

        Returns:
            None.

        Why this matters:
            Starting conditions are identical for all agents. Any differences in
            final P&L are attributable to persona behavior, not initial endowment.
        """
        if persona not in PERSONA_IDENTITIES:
            raise ValueError(f"Unknown persona: {persona}. Must be one of {list(PERSONA_IDENTITIES.keys())}")

        self.persona = persona
        self.agent_id = agent_id
        self.cash = self.INITIAL_CASH
        self.shares = 0
        self.trade_history: list[dict] = []
        self.initial_cash = self.INITIAL_CASH

    @abstractmethod
    def decide(self, market_state: dict) -> dict:
        """
        Make a trading decision based on the current market state.

        Args:
            market_state: A dict as returned by Market.get_state(), containing:
                          - price_history: list of recent prices
                          - current_price: float
                          - fair_value: float
                          - news: str
                          - tick: int

        Returns:
            A dict with keys:
                - "action": "BUY", "SELL", or "HOLD"
                - "quantity": integer >= 0 (0 for HOLD)
                - "persona": this agent's persona label
                - "agent_id": this agent's ID

        Why this matters:
            The decide() method is the only interface between the agent and the market.
            A clean, consistent return format simplifies the simulation orchestrator.
        """
        pass

    def execute_trade(self, action: str, quantity: int, price: float) -> dict:
        """
        Update the agent's portfolio after a trade is confirmed.

        This enforces the no-short-selling and no-leverage constraints.

        Args:
            action: "BUY", "SELL", or "HOLD".
            quantity: Number of shares to buy or sell.
            price: The execution price.

        Returns:
            A dict describing the executed trade (which may differ from the intended
            trade if constraints were binding).

        Why this matters:
            Constraints are enforced here, not in decide(). This separates intent
            (what the agent wants to do) from execution (what it is allowed to do).
        """
        executed_quantity = 0

        if action == "BUY" and quantity > 0:
            # Cannot spend more cash than available.
            # SIMPLIFICATION: In a real market, this would involve margin, fees, etc.
            max_affordable = int(self.cash / price)
            executed_quantity = min(quantity, max_affordable)
            cost = executed_quantity * price
            self.cash -= cost
            self.shares += executed_quantity

        elif action == "SELL" and quantity > 0:
            # Cannot sell more shares than held (no short selling).
            executed_quantity = min(quantity, self.shares)
            proceeds = executed_quantity * price
            self.cash += proceeds
            self.shares -= executed_quantity

        trade_record = {
            "action": action,
            "intended_quantity": quantity,
            "executed_quantity": executed_quantity,
            "price": price,
            "cash_after": round(self.cash, 2),
            "shares_after": self.shares,
        }

        if executed_quantity > 0:
            self.trade_history.append(trade_record)

        return trade_record

    def get_pnl(self, current_price: float) -> float:
        """
        Compute the agent's current P&L relative to the initial endowment.

        P&L = (current_cash + current_shares * current_price) - initial_cash

        Args:
            current_price: The current market price for marking shares to market.

        Returns:
            P&L as a float. Positive means profit, negative means loss.

        Why this matters:
            P&L is the ultimate measure of how well each persona performed.
            Comparing P&L across personas at the end of simulation shows which
            behavioral types are rewarded in which market conditions.
        """
        portfolio_value = self.cash + self.shares * current_price
        return round(portfolio_value - self.initial_cash, 2)


# ---------------------------------------------------------------------------
# RuleBasedAgent
# ---------------------------------------------------------------------------


class RuleBasedAgent(BaseAgent):
    """
    A trader agent that uses simple hand-coded rules to make decisions.

    This agent does not require a GPU or any model weights. It is the default
    for all simulation runs in this project.

    The rules are deliberately simple and transparent:
    - Momentum: follow the 3-tick trend
    - Value: buy below 95% of fair value, sell above 105% of fair value
    - Noise: random with slight directional bias from recent prices

    Attributes:
        (inherits from BaseAgent)
        rng: A seeded Random instance for the noise trader's random decisions.
    """

    def __init__(self, persona: str, agent_id: str, seed: Optional[int] = None) -> None:
        """
        Initialize the rule-based agent.

        Args:
            persona: One of "momentum", "value", "noise".
            agent_id: A unique identifier string.
            seed: Optional random seed (used only by noise trader).

        Returns:
            None.

        Why this matters:
            The seed allows different noise trader instances to make independent
            (but reproducible) decisions rather than all following the same random path.
        """
        super().__init__(persona, agent_id)
        # Each agent gets a unique seed derived from its ID so their noise is independent.
        # SIMPLIFICATION: A production version would use a proper PRNG with guaranteed independence.
        seed_value = seed if seed is not None else abs(hash(agent_id)) % (2**31)
        self.rng = random.Random(seed_value)

    def decide(self, market_state: dict) -> dict:
        """
        Make a trading decision using persona-specific rules.

        Dispatches to the appropriate rule method based on self.persona.

        Args:
            market_state: As returned by Market.get_state().

        Returns:
            A decision dict with keys: action, quantity, persona, agent_id.

        Why this matters:
            The dispatch pattern makes it easy to add new personas without changing
            the interface — just add a new _decide_{persona} method.
        """
        if self.persona == "momentum":
            action, quantity = self._decide_momentum(market_state)
        elif self.persona == "value":
            action, quantity = self._decide_value(market_state)
        elif self.persona == "noise":
            action, quantity = self._decide_noise(market_state)
        else:
            action, quantity = "HOLD", 0

        return {
            "action": action,
            "quantity": quantity,
            "persona": self.persona,
            "agent_id": self.agent_id,
        }

    def _decide_momentum(self, market_state: dict) -> tuple[str, int]:
        """
        Momentum rule: follow the 3-period price trend.

        If the last 3 prices are strictly increasing → BUY.
        If the last 3 prices are strictly decreasing → SELL.
        Otherwise → HOLD.

        Args:
            market_state: The current market state dict.

        Returns:
            A (action, quantity) tuple.

        Why this matters:
            This rule captures the essence of momentum trading: extrapolate the recent
            trend forward. It is simple enough to be transparent but generates the
            herding behavior and trend amplification we expect from this persona.
        """
        prices = market_state["price_history"]

        if len(prices) < 3:
            return "HOLD", 0

        # Look at the last 3 prices.
        p_minus_2 = prices[-3]
        p_minus_1 = prices[-2]
        p_now = prices[-1]

        trend_up = p_minus_2 < p_minus_1 < p_now
        trend_down = p_minus_2 > p_minus_1 > p_now

        if trend_up:
            # Buy with quantity proportional to trend strength.
            strength = (p_now - p_minus_2) / p_minus_2
            quantity = min(int(strength * 500) + 5, 20)  # Cap at 20
            quantity = max(quantity, 1)
            return "BUY", quantity

        elif trend_down:
            # Sell with quantity proportional to trend strength.
            strength = (p_minus_2 - p_now) / p_minus_2
            quantity = min(int(strength * 500) + 5, 20)
            quantity = max(quantity, 1)
            return "SELL", quantity

        else:
            return "HOLD", 0

    def _decide_value(self, market_state: dict) -> tuple[str, int]:
        """
        Value rule: buy below 95% of fair value, sell above 105% of fair value.

        The 5% margin of safety threshold is a standard value investing heuristic.
        Below the threshold, the agent HOLDs (neither cheap enough to buy nor
        expensive enough to sell).

        Args:
            market_state: The current market state dict.

        Returns:
            A (action, quantity) tuple.

        Why this matters:
            This rule creates mean-reversion pressure: value investors act as
            arbitrageurs who buy when prices are too low and sell when too high.
            In a homogeneous population, this produces a stable, low-volatility market.
        """
        current_price = market_state["current_price"]
        fair_value = market_state["fair_value"]

        if fair_value <= 0:
            return "HOLD", 0

        deviation = (current_price - fair_value) / fair_value

        if deviation < -0.05:
            # Price is more than 5% below fair value. Margin of safety achieved. BUY.
            # Quantity is proportional to the discount.
            discount = -deviation  # Positive number
            quantity = min(int(discount * 200) + 3, 20)
            quantity = max(quantity, 1)
            return "BUY", quantity

        elif deviation > 0.05:
            # Price is more than 5% above fair value. Overextended. SELL.
            premium = deviation  # Positive number
            quantity = min(int(premium * 200) + 3, 20)
            quantity = max(quantity, 1)
            return "SELL", quantity

        else:
            # Within 5% of fair value. Hold and wait.
            return "HOLD", 0

    def _decide_noise(self, market_state: dict) -> tuple[str, int]:
        """
        Noise rule: mostly random with slight bias toward recent price direction.

        The noise trader:
        - 60% probability of trading (BUY or SELL), 40% probability of HOLD
        - Given trading, 55% toward recent price direction, 45% against
        - Quantities are random in [1, 20]

        This produces high-turnover, sentiment-amplifying but erratic behavior.

        Args:
            market_state: The current market state dict.

        Returns:
            A (action, quantity) tuple.

        Why this matters:
            Noise traders add entropy to the market. Their erratic behavior, combined
            with sentiment-chasing, creates the non-linearities needed for fat tails.
        """
        prices = market_state["price_history"]

        # Determine the recent price direction for the sentiment bias.
        if len(prices) >= 2:
            recent_up = prices[-1] > prices[-2]
        else:
            recent_up = self.rng.random() > 0.5

        # 40% chance of HOLDing.
        if self.rng.random() < 0.40:
            return "HOLD", 0

        # 55% chance of following the recent direction (sentiment-chasing).
        follow_trend = self.rng.random() < 0.55

        if follow_trend:
            action = "BUY" if recent_up else "SELL"
        else:
            action = "SELL" if recent_up else "BUY"

        # Random quantity.
        quantity = self.rng.randint(1, 20)
        return action, quantity


# ---------------------------------------------------------------------------
# SLMAgent
# ---------------------------------------------------------------------------


class SLMAgent(BaseAgent):
    """
    A trader agent backed by a fine-tuned SLM with a LoRA persona adapter.

    This agent loads the base model and a LoRA adapter at initialization, then
    runs inference on each tick to generate a reasoning trace and trading decision.

    Note: This agent requires a GPU or a quantized model for practical use.
    Use RuleBasedAgent for CPU-only simulation runs.

    Attributes:
        (inherits from BaseAgent)
        adapter_path: Path to the LoRA adapter directory.
        base_model_name: HuggingFace model name for the base model.
        model: The loaded PEFT model (set after _load_model() is called).
        tokenizer: The loaded tokenizer.
        fallback_agent: A RuleBasedAgent used when model output is unparseable.
    """

    def __init__(
        self,
        persona: str,
        agent_id: str,
        adapter_path: str,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    ) -> None:
        """
        Initialize the SLM agent and load the model.

        Args:
            persona: One of "momentum", "value", "noise".
            agent_id: A unique identifier string.
            adapter_path: Path to the directory containing the saved LoRA adapter.
            base_model_name: HuggingFace model name for the base model.

        Returns:
            None.

        Why this matters:
            Loading the model at __init__ time (not at first inference) ensures that
            initialization failures are caught early, before the simulation starts.
        """
        super().__init__(persona, agent_id)
        self.adapter_path = adapter_path
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None

        # Fallback agent for when model output cannot be parsed.
        self.fallback_agent = RuleBasedAgent(persona, agent_id + "_fallback")

        self._load_model()

    def _load_model(self) -> None:
        """
        Load the base model and LoRA adapter.

        Args:
            None.

        Returns:
            None. Sets self.model and self.tokenizer.

        Why this matters:
            We use PEFT's PeftModel.from_pretrained() which handles the adapter merging
            transparently. The base model weights are not modified.
        """
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                f"SLMAgent requires transformers and peft: {exc}\n"
                "Install with: pip install transformers peft\n"
                "Or use RuleBasedAgent for CPU-only simulation."
            ) from exc

        print(f"[SLMAgent {self.agent_id}] Loading base model: {self.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        import torch
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU inference
            trust_remote_code=True,
        )

        print(f"[SLMAgent {self.agent_id}] Loading LoRA adapter from: {self.adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model.eval()
        print(f"[SLMAgent {self.agent_id}] Model loaded successfully.")

    def _build_prompt(self, market_state: dict) -> str:
        """
        Build the inference prompt from the current market state.

        This must match the format used during training in finetune.py exactly.

        Args:
            market_state: The current market state dict from Market.get_state().

        Returns:
            A formatted prompt string.

        Why this matters:
            Train/inference prompt consistency is the single most important factor for
            adapter generalization. Any deviation will degrade decision quality.
        """
        prices = market_state["price_history"]
        news = market_state["news"]
        fair_value = market_state["fair_value"]
        current_price = market_state["current_price"]
        price_change = round((current_price - prices[0]) / max(prices[0], 0.01) * 100, 2)

        return (
            f"You are analyzing a financial market. Here is the current market state:\n\n"
            f"Price history (last 5 periods): {prices}\n"
            f"Current price: {current_price}\n"
            f"Price change over window: {price_change}%\n"
            f"Latest news: {news}\n"
            f"Estimated fair value: {fair_value}\n\n"
            f"Based on your identity as a trader, reason step-by-step about what you observe "
            f"and what action you will take.\n\n"
            f"Then provide your decision in exactly this JSON format (and nothing else after it):\n"
            f'{{\n  "reasoning": "<your step-by-step reasoning, 2-4 sentences>",\n'
            f'  "action": "<BUY or SELL or HOLD>",\n'
            f'  "quantity": <integer from 1 to 20, or 0 if HOLD>\n}}\n\n'
            f"Remember: you must reason in character. Do not break character or hedge."
        )

    def _parse_model_output(self, output_text: str) -> Optional[tuple[str, int]]:
        """
        Parse the model's text output to extract action and quantity.

        Args:
            output_text: The raw text generated by the model.

        Returns:
            A (action, quantity) tuple if parsing succeeds, or None if it fails.

        Why this matters:
            LLM outputs are not always perfectly structured. Robust parsing with a
            clear failure path (return None → use fallback) makes the simulation
            resilient to occasional model misbehavior.
        """
        json_start = output_text.rfind("{")
        json_end = output_text.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            return None

        try:
            parsed = json.loads(output_text[json_start:json_end])
        except json.JSONDecodeError:
            return None

        action = parsed.get("action", "").upper()
        if action not in ("BUY", "SELL", "HOLD"):
            return None

        quantity = parsed.get("quantity", 0)
        if not isinstance(quantity, int) or quantity < 0 or quantity > 20:
            return None

        return action, quantity

    def decide(self, market_state: dict) -> dict:
        """
        Make a trading decision by running inference with the fine-tuned SLM.

        Falls back to rule-based logic if the model output cannot be parsed.

        Args:
            market_state: As returned by Market.get_state().

        Returns:
            A decision dict with keys: action, quantity, persona, agent_id.

        Why this matters:
            The fallback ensures the simulation never crashes due to a bad model output.
            In a production system, bad outputs would be logged for model improvement.
        """
        import torch

        system_prompt = PERSONA_IDENTITIES[self.persona]
        user_prompt = self._build_prompt(market_state)

        # Format as a chat using the tokenizer's chat template.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(input_text, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens (not the prompt).
            new_token_ids = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

            result = self._parse_model_output(generated_text)

        except Exception as exc:
            print(f"[SLMAgent {self.agent_id}] Inference error: {exc}. Using fallback.")
            result = None

        if result is not None:
            action, quantity = result
        else:
            # Fallback to rule-based agent.
            fallback_decision = self.fallback_agent.decide(market_state)
            action = fallback_decision["action"]
            quantity = fallback_decision["quantity"]

        return {
            "action": action,
            "quantity": quantity,
            "persona": self.persona,
            "agent_id": self.agent_id,
        }
