"""
ppo_config.py
=============
Central configuration constants for PPO fine-tuning and LoRA/PEFT setup.
All notebooks import from here to keep settings consistent.
"""

from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt2-medium"          # ~345M parameters
REWARD_MODEL_DIR = "reward_model"   # where notebook 02 saves the reward model
PPO_MODEL_DIR = "ppo_model"         # where notebook 03 saves the PPO model

# ---------------------------------------------------------------------------
# LoRA configuration  (matches resume: rank 8, alpha 16)
# ---------------------------------------------------------------------------
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["c_attn"]    # GPT-2's combined Q/K/V attention weight
LORA_BIAS = "none"

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 200
MIN_NEW_TOKENS = 30
TEMPERATURE = 0.9
TOP_P = 0.92
REPETITION_PENALTY = 1.3

# ---------------------------------------------------------------------------
# PPO training hyperparameters
# ---------------------------------------------------------------------------
PPO_LEARNING_RATE = 1.41e-5
PPO_BATCH_SIZE = 8          # number of prompts per PPO step
PPO_MINI_BATCH_SIZE = 4
PPO_GRADIENT_ACCUMULATION = 1
PPO_EPOCHS = 4              # inner epochs per PPO step
PPO_STEPS = 200             # total PPO steps (T4 budget ~90 min)
PPO_INIT_KL_COEF = 0.2
PPO_TARGET_KL = 6.0
PPO_CLIP_EPS = 0.2
PPO_VF_COEF = 0.1
PPO_MAX_GRAD_NORM = 1.0
REWARD_BASELINE = 0.0       # subtract mean reward as baseline

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
PREFERENCES_CSV = "data/preferences.csv"
MAX_SEQ_LENGTH = 512
SEED = 42

# ---------------------------------------------------------------------------
# Dataclass wrappers for clean import
# ---------------------------------------------------------------------------
@dataclass
class LoRAConfig:
    r: int = LORA_R
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    target_modules: List[str] = field(default_factory=lambda: LORA_TARGET_MODULES)
    bias: str = LORA_BIAS
    task_type: str = "CAUSAL_LM"


@dataclass
class PPOHyperParams:
    learning_rate: float = PPO_LEARNING_RATE
    batch_size: int = PPO_BATCH_SIZE
    mini_batch_size: int = PPO_MINI_BATCH_SIZE
    gradient_accumulation_steps: int = PPO_GRADIENT_ACCUMULATION
    ppo_epochs: int = PPO_EPOCHS
    steps: int = PPO_STEPS
    init_kl_coef: float = PPO_INIT_KL_COEF
    target_kl: float = PPO_TARGET_KL
    cliprange: float = PPO_CLIP_EPS
    vf_coef: float = PPO_VF_COEF
    max_grad_norm: float = PPO_MAX_GRAD_NORM


def trainable_param_percent(model) -> float:
    """Return percentage of trainable parameters (for LoRA verification)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    return pct


if __name__ == "__main__":
    print("=== ppo_config.py ===")
    print(f"Model: {MODEL_NAME}")
    print(f"LoRA r={LORA_R}, alpha={LORA_ALPHA}, target={LORA_TARGET_MODULES}")
    print(f"PPO steps={PPO_STEPS}, batch={PPO_BATCH_SIZE}")
    cfg = LoRAConfig()
    print(f"LoRAConfig: {cfg}")
    ppo = PPOHyperParams()
    print(f"PPOHyperParams: {ppo}")
