"""
agent.py — Trader agents for the persona simulation.

Talk section: Section III — The Proposed Framework

Purpose:
    Defines two agent implementations:
    1. SLMAgent: Wraps a fine-tuned LoRA adapter and runs actual model inference.
       This is the preferred agent type. The orchestrator (run_simulation.create_agents)
       uses SLMAgent by default whenever LoRA adapters are available on disk.
    2. RuleBasedAgent: Implements each persona's logic as a simple hand-coded rule.
       This runs on CPU with no model weights required and is the automatic fallback
       used by the orchestrator when adapters are missing or fail to load.

Key design decisions:
    - Both agents expose the same interface: decide(market_state) -> dict. This makes
      them interchangeable in the simulation without any changes to the orchestrator.
    - SLMAgent shares model + tokenizer across all agents that point at the same
      adapter directory via a module-level cache (_MODEL_CACHE). This means a
      30-agent population with three personas loads three models, not 30, which is
      essential for memory feasibility when running on a laptop.
    - Inference is batched at the tick level via the top-level batched_decide()
      helper. All SLMAgents that share a loaded model are pooled into a single
      model.generate() call per tick, so a 30-agent homogeneous population costs
      one generate call per tick instead of thirty. The run_simulation orchestrator
      calls batched_decide() directly; the per-agent SLMAgent.decide() method
      remains available for standalone / notebook use.
    - The RuleBasedAgent's rules are intentionally simple and transparent. They are
      meant to demonstrate the behavioral contrast between personas, not to be
      realistic trading strategies.
    - The SLMAgent includes a per-decision fallback to rule-based logic if a single
      model output is unparseable. This is independent of the orchestrator-level
      fallback, which kicks in only at startup if the adapter directory is missing.
    - All agents track their own cash, shares, trade history, and P&L. This keeps
      state encapsulated per-agent and makes the simulation easy to parallelize later.
    - Agents enforce a no-short-selling constraint: SELL is only executed if the agent
      currently holds shares. Similarly, BUY is only executed if the agent has cash.
"""

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
#
# Loading a 1.5B-parameter base model + a LoRA adapter takes ~30s on CPU and
# uses several GB of RAM. The hero experiment runs 30 agents at a time. We do
# NOT want to load 30 copies of the model — instead, we cache (model, tokenizer)
# tuples keyed by the resolved adapter path so all agents pointing at the same
# adapter share weights. The model state is read-only during inference, so
# sharing is safe.
#
# Cache key: (resolved adapter path string, base model name string)
# Cache value: (model, tokenizer)
_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def clear_model_cache() -> None:
    """
    Clear the in-process SLMAgent model cache.

    Args:
        None.

    Returns:
        None.

    Why this matters:
        Useful in long-lived processes (e.g. notebooks) where you want to free
        GPU/RAM after a simulation completes, or when re-loading adapters that
        have been updated on disk.
    """
    _MODEL_CACHE.clear()


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

    This agent loads the base model and a LoRA adapter at initialization (or
    retrieves them from the module-level cache if another agent already loaded
    the same adapter), then runs inference on each tick to generate a reasoning
    trace and trading decision.

    The orchestrator (run_simulation.create_agents) instantiates SLMAgent by
    default whenever a valid adapter directory is present, and falls back to
    RuleBasedAgent only if the adapter is missing or fails to load. Inference
    is feasible on CPU for the 1.5B-parameter Qwen base model — slow but
    workable on a laptop. A GPU is much faster.

    Attributes:
        (inherits from BaseAgent)
        adapter_path: Path to the LoRA adapter directory.
        base_model_name: HuggingFace model name for the base model.
        model: The loaded PEFT model (set after _load_model() is called).
        tokenizer: The loaded tokenizer.
        fallback_agent: A RuleBasedAgent used when a single model decoding is
                        unparseable. This is *per-decision* fallback, not the
                        startup-time orchestrator fallback.
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
        self.device = None

        # Fallback agent for when model output cannot be parsed.
        self.fallback_agent = RuleBasedAgent(persona, agent_id + "_fallback")

        self._load_model()

    @staticmethod
    def is_available(adapter_path: str) -> tuple[bool, str]:
        """
        Check whether a LoRA adapter directory looks usable, without loading it.

        Args:
            adapter_path: Path to the directory that should contain the LoRA adapter.

        Returns:
            A (ok, reason) tuple. `ok` is True if the directory exists and contains
            an `adapter_config.json`. `reason` is an empty string when ok is True,
            otherwise a short human-readable explanation suitable for logging.

        Why this matters:
            The orchestrator uses this to decide whether to instantiate SLMAgent
            or fall back to RuleBasedAgent. We deliberately do *not* import
            transformers/peft here — that check happens in _load_model so a
            missing adapter directory does not require the heavy ML stack to be
            installed for the fallback path to work.
        """
        path = Path(adapter_path)
        if not path.exists():
            return False, f"adapter directory does not exist: {adapter_path}"
        if not path.is_dir():
            return False, f"adapter path is not a directory: {adapter_path}"
        config_file = path / "adapter_config.json"
        if not config_file.exists():
            return False, (
                f"adapter_config.json missing in {adapter_path} "
                "(expected a PEFT/LoRA adapter directory)"
            )
        return True, ""

    def _load_model(self) -> None:
        """
        Load the base model and LoRA adapter, using the module-level cache.

        Args:
            None.

        Returns:
            None. Sets self.model and self.tokenizer.

        Why this matters:
            We use PEFT's PeftModel.from_pretrained() which handles the adapter merging
            transparently. The base model weights are not modified. The cache ensures
            that 30 agents pointing at the same adapter only load the model once.
        """
        cache_key = (str(Path(self.adapter_path).resolve()), self.base_model_name)
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            self.model, self.tokenizer = cached
            self.device = next(self.model.parameters()).device
            return

        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                f"SLMAgent requires transformers and peft: {exc}\n"
                "Install with: pip install transformers peft\n"
                "Or remove the adapters/ directory to fall back to RuleBasedAgent."
            ) from exc

        import torch

        # Pick the best available accelerator. CUDA gets bfloat16 on Ampere+
        # (matching the fine-tuning recipe in finetune.py so adapter weights map
        # cleanly onto base weights), float16 on older GPUs. Apple MPS uses
        # float16. CPU stays on float32 — slow but correct, preserving the
        # "must run end-to-end without a GPU" guarantee from CLAUDE.md.
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float16
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        print(
            f"[SLMAgent] Loading base model: {self.base_model_name} "
            f"(device={device}, dtype={dtype})"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        print(f"[SLMAgent] Loading LoRA adapter from: {self.adapter_path}")
        model = PeftModel.from_pretrained(base_model, self.adapter_path)
        model.to(device)

        # Merge LoRA weights into the base model for inference. Without this,
        # every forward pass pays an extra LoRA matmul per targeted linear
        # layer, which is the dominant per-token cost for a small model on
        # GPU. merge_and_unload() folds the delta weights into the base
        # linears and returns a plain transformers CausalLM — inference speed
        # now matches the base model. Safe because the cache is read-only
        # during simulation; we never train from it.
        print("[SLMAgent] Merging LoRA weights into base for inference...")
        model = model.merge_and_unload()
        model.eval()

        _MODEL_CACHE[cache_key] = (model, tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        print(
            f"[SLMAgent] Model + adapter cached as "
            f"'{Path(self.adapter_path).name}' "
            f"({len(_MODEL_CACHE)} model(s) now cached) on {device}."
        )

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
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
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


# ---------------------------------------------------------------------------
# Tick-level batched decisions
# ---------------------------------------------------------------------------


def batched_decide(agents: list, market_state: dict) -> list[dict]:
    """
    Make trading decisions for a whole population with batched SLM inference.

    SLMAgents are grouped by their shared (adapter_path, base_model) — the same
    key _MODEL_CACHE uses — so every group is guaranteed to touch one loaded
    model. Each group is dispatched as a single batched model.generate() call.
    RuleBasedAgents are handled individually on their existing fast path.

    Args:
        agents: A list of agent instances (any mix of SLMAgent / RuleBasedAgent).
        market_state: The current market state dict, shared across all agents.

    Returns:
        A list of decision dicts in the same order as `agents`.

    Why this matters:
        model.generate() per agent is the dominant cost of a tick. A homogeneous
        30-agent population collapses from 30 generate calls to 1; the 10/10/10
        mixed population collapses from 30 to 3 (one per persona adapter). This
        is the single largest wall-clock win available without changing the
        simulation semantics.
    """
    decisions: list[Optional[dict]] = [None] * len(agents)

    slm_groups: dict[tuple[str, str], list[tuple[int, "SLMAgent"]]] = {}
    for idx, agent in enumerate(agents):
        if isinstance(agent, SLMAgent):
            key = (str(Path(agent.adapter_path).resolve()), agent.base_model_name)
            slm_groups.setdefault(key, []).append((idx, agent))
        else:
            decisions[idx] = agent.decide(market_state)

    for group in slm_groups.values():
        group_decisions = _decide_slm_batch(group, market_state)
        for (idx, _), decision in zip(group, group_decisions):
            decisions[idx] = decision

    return decisions  # type: ignore[return-value]


def _decide_slm_batch(
    group: list[tuple[int, "SLMAgent"]],
    market_state: dict,
) -> list[dict]:
    """
    Run one batched model.generate() call for a group of SLMAgents.

    All agents in `group` share the same loaded model, tokenizer, and device by
    construction (batched_decide groups them on the cache key). Prompts are
    still built per-agent so the batch remains correct even if two agents in
    the group ever diverged (e.g. different personas pointed at the same
    adapter in a future experiment).

    Args:
        group: List of (original_index, SLMAgent) tuples. The index is only
               used by the caller to splice decisions back into the full list.
        market_state: The current market state dict.

    Returns:
        A list of decision dicts in the same order as `group`.

    Why this matters:
        Pooling prompts into one generate() call eliminates per-agent Python
        overhead and lets the GPU exploit true batch parallelism — this is
        where the tick-level speedup actually materializes.
    """
    import torch

    first_agent = group[0][1]
    tokenizer = first_agent.tokenizer
    model = first_agent.model
    device = first_agent.device

    # Build one chat-templated prompt per agent. Kept in the same format as
    # SLMAgent.decide() so batching does not drift from the train-time prompt.
    input_texts: list[str] = []
    for _, agent in group:
        messages = [
            {"role": "system", "content": PERSONA_IDENTITIES[agent.persona]},
            {"role": "user", "content": agent._build_prompt(market_state)},
        ]
        input_texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    # Left-padding is mandatory for batched generation: every prompt must end
    # at the same position in the tensor so the newly generated tail lines up
    # at `input_len:` for every row. We restore the prior padding_side after
    # tokenizing so we do not leak state into unrelated call sites.
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
        ).to(device)
    finally:
        tokenizer.padding_side = original_padding_side

    decisions: list[dict] = []
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for row, (_, agent) in enumerate(group):
            new_token_ids = outputs[row][input_len:]
            generated_text = tokenizer.decode(
                new_token_ids, skip_special_tokens=True
            )
            parsed = agent._parse_model_output(generated_text)
            if parsed is not None:
                action, quantity = parsed
            else:
                fb = agent.fallback_agent.decide(market_state)
                action, quantity = fb["action"], fb["quantity"]
            decisions.append(
                {
                    "action": action,
                    "quantity": quantity,
                    "persona": agent.persona,
                    "agent_id": agent.agent_id,
                }
            )
    except Exception as exc:
        # Preserve the per-decision fallback contract: if a batch blows up,
        # every agent in that batch falls back to its rule-based twin rather
        # than crashing the whole simulation.
        print(
            f"[SLMAgent batch] Inference error for {len(group)}-agent batch: "
            f"{exc}. Using rule-based fallback for the whole group."
        )
        for _, agent in group:
            fb = agent.fallback_agent.decide(market_state)
            decisions.append(
                {
                    "action": fb["action"],
                    "quantity": fb["quantity"],
                    "persona": agent.persona,
                    "agent_id": agent.agent_id,
                }
            )

    return decisions
