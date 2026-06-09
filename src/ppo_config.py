from dataclasses import dataclass, field
from typing import List


MODEL_NAME = "gpt2-medium"
REWARD_MODEL_DIR = "reward_model"
PPO_MODEL_DIR = "ppo_model"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["c_attn"]
LORA_BIAS = "none"

MAX_NEW_TOKENS = 200
MIN_NEW_TOKENS = 30
TEMPERATURE = 0.9
TOP_P = 0.92
REPETITION_PENALTY = 1.3

PPO_LEARNING_RATE = 1.41e-5
PPO_BATCH_SIZE = 8
PPO_MINI_BATCH_SIZE = 4
PPO_GRADIENT_ACCUMULATION = 1
PPO_EPOCHS = 4
PPO_STEPS = 200
PPO_INIT_KL_COEF = 0.2
PPO_TARGET_KL = 6.0
PPO_CLIP_EPS = 0.2
PPO_VF_COEF = 0.1
PPO_MAX_GRAD_NORM = 1.0
REWARD_BASELINE = 0.0

PREFERENCES_CSV = "data/preferences.csv"
MAX_SEQ_LENGTH = 512
SEED = 42


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
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    return pct
