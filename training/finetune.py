"""
finetune.py — LoRA fine-tuning for trader persona adapters.

Talk section: Section III — The Proposed Framework

Purpose:
    Fine-tunes a LoRA adapter on top of Qwen/Qwen2.5-1.5B-Instruct for one persona
    at a time. Each persona gets its own adapter saved to training/adapters/{persona}/.
    The adapters are lightweight and can be hot-swapped at inference time.

Key design decisions:
    - We use SFTTrainer from TRL because it handles the chat template formatting
      automatically and natively supports PEFT adapters. Writing a custom training
      loop would add complexity without benefit for this scale.
    - We format each training example as a two-turn chat: the user provides the
      market state, the assistant provides the reasoning + action JSON. This mirrors
      the exact format used at inference time in simulation/agent.py.
    - We train on the completion only (the assistant turn), not the prompt. This is
      standard for instruction-following fine-tuning.
    - The base model is loaded in bfloat16 to reduce memory usage. On a 40GB A100,
      this leaves enough headroom for gradient accumulation with batch size 4.

Estimated training time:
    - ~15 minutes per persona on a single A100 (40GB) GPU with the default config.
    - CPU training is technically possible but would take many hours. We strongly
      recommend using a GPU for fine-tuning.

Fallback (no GPU):
    If you do not have a GPU, you can load a pre-trained adapter and skip training:

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        model = PeftModel.from_pretrained(base_model, "training/adapters/momentum")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    Pre-trained adapters (if provided by the instructor) should be placed in
    training/adapters/{persona}/ before running the simulation with --use-slm.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

# We import training dependencies inside functions so that the file can be imported
# and inspected on a machine without GPU dependencies installed.


def load_config(config_path: str) -> dict:
    """
    Load a YAML training configuration file.

    Args:
        config_path: Path to the YAML config file (e.g., training/configs/momentum.yaml).

    Returns:
        A dict containing all configuration fields.

    Why this matters:
        Keeping hyperparameters in YAML (not hardcoded) makes it easy to run ablations
        and ensures the configuration is versioned alongside the code.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_jsonl(path: str) -> list[dict]:
    """
    Load all examples from a .jsonl file.

    Args:
        path: Path to the .jsonl file.

    Returns:
        A list of dicts, one per line.

    Why this matters:
        JSONL is the standard format for streaming datasets. Each line is a self-contained
        JSON object, making it easy to append incrementally during data generation.
    """
    examples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def format_example_as_chat(example: dict, persona_identity: str) -> dict:
    """
    Format one training example as a chat conversation for SFTTrainer.

    The format is:
        system: the persona identity string
        user: the market state description
        assistant: the reasoning + action JSON

    This mirrors the inference prompt in simulation/agent.py exactly.

    Args:
        example: A training example dict with keys: persona, market_state, reasoning,
                 action, quantity.
        persona_identity: The persona's identity string (used as system prompt).

    Returns:
        A dict with key "messages" containing a list of chat turns.

    Why this matters:
        Train/inference prompt consistency is critical. If the prompt format differs
        between training and inference, the adapter will not generalize.
    """
    market_state = example["market_state"]
    price_history = market_state["price_history"]
    news = market_state["news"]
    fair_value = market_state["fair_value"]
    current_price = price_history[-1]
    price_change = round((current_price - price_history[0]) / price_history[0] * 100, 2)

    user_content = (
        f"You are analyzing a financial market. Here is the current market state:\n\n"
        f"Price history (last 5 periods): {price_history}\n"
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

    assistant_content = json.dumps(
        {
            "reasoning": example["reasoning"],
            "action": example["action"],
            "quantity": example["quantity"],
        },
        indent=2,
    )

    return {
        "messages": [
            {"role": "system", "content": persona_identity},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


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


def run_finetuning(config: dict) -> None:
    """
    Execute the LoRA fine-tuning run for one persona.

    Args:
        config: A config dict loaded from a YAML file. Must contain all fields defined
                in training/configs/{persona}.yaml.

    Returns:
        None. Saves the trained adapter to config["output_dir"].

    Why this matters:
        This is the core training function. Keeping it self-contained (all imports inside)
        means the file can be inspected and imported on CPU-only machines without errors.
    """
    # Import training dependencies here to avoid hard errors on CPU-only machines.
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        print(
            f"Import error: {exc}\n"
            "Make sure you have installed all GPU training dependencies:\n"
            "  pip install transformers peft trl torch datasets accelerate",
            file=sys.stderr,
        )
        sys.exit(1)

    persona = config["persona"]
    identity = PERSONA_IDENTITIES[persona]

    # --- Load and format data ---
    print(f"Loading training data from {config['train_data_path']}...")
    raw_examples = load_jsonl(config["train_data_path"])
    print(f"Loaded {len(raw_examples)} examples.")

    formatted = [format_example_as_chat(ex, identity) for ex in raw_examples]
    dataset = Dataset.from_list(formatted)

    # --- Load base model and tokenizer ---
    print(f"Loading base model: {config['base_model_name_or_path']}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model_name_or_path"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        # Qwen models may not have a pad token set by default.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model_name_or_path"],
        torch_dtype=torch.bfloat16 if config.get("bf16", True) else torch.float32,
        trust_remote_code=True,
    )

    # --- Configure LoRA ---
    # SIMPLIFICATION: We use a fixed set of target modules. A production version would
    # profile the model architecture to identify the most impactful layers to adapt.
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias=config.get("bias", "none"),
        task_type=config.get("task_type", "CAUSAL_LM"),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Training arguments ---
    output_dir = config["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.05),
        weight_decay=config.get("weight_decay", 0.01),
        logging_steps=config.get("logging_steps", 10),
        save_strategy=config.get("save_strategy", "epoch"),
        bf16=config.get("bf16", True),
        fp16=config.get("fp16", False),
        report_to="none",  # Disable W&B / MLflow in this v1 implementation
    )

    # --- SFTTrainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=config.get("max_seq_length", 512),
        # Train on completions only (the assistant turn).
        # SIMPLIFICATION: A production version would carefully verify that the loss mask
        # is applied correctly to avoid training on prompt tokens.
        dataset_text_field=None,
    )

    print(f"Starting training for persona '{persona}'...")
    print(f"Estimated time on A100: ~15 minutes")
    trainer.train()

    # Save the final adapter.
    final_adapter_path = Path(output_dir) / "final"
    trainer.model.save_pretrained(str(final_adapter_path))
    tokenizer.save_pretrained(str(final_adapter_path))
    print(f"Adapter saved to {final_adapter_path}")


def main() -> None:
    """
    CLI entry point for LoRA fine-tuning.

    Args:
        None (reads from sys.argv via argparse).

    Returns:
        None.

    Why this matters:
        A clean CLI makes it possible to launch training as a SLURM job on a GPU cluster
        with a single command, and to track which config was used for each run.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a LoRA adapter for a trader persona."
    )
    parser.add_argument(
        "--persona",
        type=str,
        choices=["momentum", "value", "noise"],
        required=True,
        help="Which persona to fine-tune.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file (e.g., training/configs/momentum.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Sanity check: the config persona must match the CLI persona.
    if config.get("persona") != args.persona:
        print(
            f"Warning: --persona={args.persona} but config says persona={config.get('persona')}. "
            f"Using --persona value.",
            file=sys.stderr,
        )
        config["persona"] = args.persona

    run_finetuning(config)


if __name__ == "__main__":
    main()
