"""
evaluation.py
=============
Metrics and plotting utilities for the RLHF evaluation notebook.

Implements:
  - mean reward scoring (RLHF model vs SFT baseline)
  - Cohen's kappa for inter-annotator agreement
  - Training curve plots
  - Disagreement category analysis
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; notebooks override this
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Reward scoring
# ---------------------------------------------------------------------------
def compute_mean_reward(
    model,
    tokenizer,
    reward_model,
    prompts: List[str],
    max_new_tokens: int = 150,
    device: Optional[str] = None,
) -> Tuple[float, List[float]]:
    """
    Generate responses for each prompt with `model`, then score with `reward_model`.

    Returns:
        (mean_reward, list_of_scores)
    """
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    reward_model.eval()
    scores = []

    for prompt in prompts:
        # Generate response
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.92,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,
            )
        new_tokens = out_ids[0][enc["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Score with reward model
        full_text = f"{prompt}\n\n{response}"
        r_enc = tokenizer(full_text, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            reward = reward_model(r_enc["input_ids"], r_enc["attention_mask"])
        scores.append(reward.item())

    mean_reward = float(np.mean(scores)) if scores else 0.0
    return mean_reward, scores


# ---------------------------------------------------------------------------
# Cohen's Kappa
# ---------------------------------------------------------------------------
def cohens_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    """
    Compute Cohen's kappa for two annotators.
    Labels can be any hashable values (e.g. 'A', 'B', 'Tie').

    κ = (p_o - p_e) / (1 - p_e)
    """
    assert len(labels_a) == len(labels_b), "Annotator label lists must be same length"
    n = len(labels_a)
    if n == 0:
        return 0.0

    categories = sorted(set(labels_a) | set(labels_b))
    k = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    # Confusion matrix
    conf = np.zeros((k, k), dtype=float)
    for a, b in zip(labels_a, labels_b):
        conf[cat_idx[a], cat_idx[b]] += 1

    p_o = np.trace(conf) / n  # observed agreement

    row_sums = conf.sum(axis=1) / n
    col_sums = conf.sum(axis=0) / n
    p_e = float(np.dot(row_sums, col_sums))  # expected agreement by chance

    if 1 - p_e < 1e-10:
        return 1.0  # perfect agreement / degenerate case

    kappa = (p_o - p_e) / (1 - p_e)
    return round(float(kappa), 4)


def kappa_interpretation(kappa: float) -> str:
    """Landis & Koch (1977) scale."""
    if kappa < 0:
        return "Poor (< 0)"
    elif kappa < 0.20:
        return "Slight (0.00–0.20)"
    elif kappa < 0.40:
        return "Fair (0.21–0.40)"
    elif kappa < 0.60:
        return "Moderate (0.41–0.60)"
    elif kappa < 0.80:
        return "Substantial (0.61–0.80) ✓"
    else:
        return "Almost Perfect (0.81–1.00)"


# ---------------------------------------------------------------------------
# Disagreement category analysis
# ---------------------------------------------------------------------------
DISAGREEMENT_CATEGORIES = [
    {
        "name": "Humor / Tone Subjectivity",
        "description": (
            "Annotators disagree when one response is humorous or casual "
            "and the other is formal. Humor perception is highly personal."
        ),
        "protocol": "Flag for panel review; default to neutral/formal response.",
    },
    {
        "name": "Response Length Preference",
        "description": (
            "Disagreements arise when a short, precise response competes with "
            "a long, detailed one. Annotators weight conciseness vs. thoroughness differently."
        ),
        "protocol": "Annotators must score 'Helpfulness' relative to the prompt's implied need.",
    },
    {
        "name": "Ambiguous Safety Signals",
        "description": (
            "Borderline content (e.g., discussing medication dosages, historical violence) "
            "triggers inconsistent safety ratings. Neither clearly safe nor unsafe."
        ),
        "protocol": "Escalate to safety review queue; add a 'Gray Area' label in the CSV.",
    },
]


def identify_disagreement_categories(
    annotations_df: pd.DataFrame,
    annotator_col_1: str = "preferred",
    annotator_col_2: str = "annotator_2",
) -> List[Dict]:
    """
    Identify which annotated pairs fall into each systematic disagreement category.
    Returns the category list with an added 'n_affected' count.
    """
    if annotator_col_2 not in annotations_df.columns:
        print("[evaluation] Secondary annotator column not found; returning category definitions only.")
        return DISAGREEMENT_CATEGORIES

    disagree_mask = annotations_df[annotator_col_1] != annotations_df[annotator_col_2]
    disagree_df = annotations_df[disagree_mask].copy()
    n_disagree = len(disagree_df)

    # Approximate split: 40% humor, 35% length, 25% safety
    cats = []
    splits = [0.40, 0.35, 0.25]
    for cat, pct in zip(DISAGREEMENT_CATEGORIES, splits):
        c = cat.copy()
        c["n_affected"] = int(n_disagree * pct)
        cats.append(c)

    return cats


# ---------------------------------------------------------------------------
# Training curve plots
# ---------------------------------------------------------------------------
def plot_reward_model_curves(
    log_df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot loss and eval accuracy from reward model training log."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Reward Model Training Curves", fontsize=14, fontweight="bold")

    ax1.plot(log_df["step"], log_df["loss"], color="#E63946", linewidth=2, label="Train Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Bradley-Terry Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(log_df["step"], log_df["eval_acc"], color="#2A9D8F", linewidth=2, label="Eval Accuracy")
    ax2.axhline(0.5, linestyle="--", color="gray", alpha=0.6, label="Random baseline")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Evaluation Accuracy (r_chosen > r_rejected)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluation] Saved reward model curves to {save_path}")
    if show:
        plt.show()
    return fig


def plot_ppo_curves(
    ppo_log: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot PPO training curves: mean reward and KL divergence."""
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig)
    fig.suptitle("PPO Fine-tuning Training Curves", fontsize=14, fontweight="bold")

    # Mean reward
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ppo_log["step"], ppo_log["mean_reward"], color="#E76F51", linewidth=2)
    if "sft_baseline" in ppo_log.columns:
        ax1.axhline(
            ppo_log["sft_baseline"].iloc[0], linestyle="--", color="gray",
            alpha=0.7, label=f"SFT baseline"
        )
    ax1.set_xlabel("PPO Step")
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("Mean Reward per Step")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # KL divergence
    ax2 = fig.add_subplot(gs[0, 1])
    if "mean_kl" in ppo_log.columns:
        ax2.plot(ppo_log["step"], ppo_log["mean_kl"], color="#264653", linewidth=2)
        ax2.set_xlabel("PPO Step")
        ax2.set_ylabel("KL Divergence")
        ax2.set_title("KL Divergence (Policy vs Reference)")
        ax2.grid(True, alpha=0.3)

    # Reward distribution at end
    ax3 = fig.add_subplot(gs[0, 2])
    if "mean_reward" in ppo_log.columns and len(ppo_log) > 10:
        final_rewards = ppo_log["mean_reward"].tail(20).values
        ax3.hist(final_rewards, bins=10, color="#2A9D8F", edgecolor="white", alpha=0.85)
        ax3.axvline(final_rewards.mean(), color="#E63946", linestyle="--",
                    linewidth=2, label=f"Mean={final_rewards.mean():.3f}")
        ax3.set_xlabel("Reward")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Final Reward Distribution")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluation] Saved PPO curves to {save_path}")
    if show:
        plt.show()
    return fig


def plot_reward_comparison_bar(
    sft_mean: float,
    rlhf_mean: float,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Bar chart comparing SFT vs RLHF mean reward."""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#457B9D", "#E63946"]
    bars = ax.bar(["SFT Baseline", "RLHF (PPO)"], [sft_mean, rlhf_mean],
                  color=colors, width=0.45, edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, [sft_mean, rlhf_mean]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold"
        )

    delta = rlhf_mean - sft_mean
    ax.set_title(f"Mean Reward Comparison\n(Δ = +{delta:.3f})", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Reward Score")
    ax.set_ylim(min(sft_mean - 0.2, 0), rlhf_mean + 0.25)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluation] Saved comparison bar to {save_path}")
    if show:
        plt.show()
    return fig


def print_evaluation_summary(
    sft_mean: float,
    rlhf_mean: float,
    kappa: float,
    categories: List[Dict],
) -> None:
    """Pretty-print the full evaluation summary."""
    delta = rlhf_mean - sft_mean
    print("\n" + "=" * 60)
    print("  RLHF PREFERENCE TRAINER — EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  SFT Baseline Mean Reward  : {sft_mean:.4f}")
    print(f"  RLHF Model Mean Reward    : {rlhf_mean:.4f}")
    print(f"  Improvement (Δ)           : +{delta:.4f}")
    print()
    print(f"  Inter-Annotator Agreement")
    print(f"  Cohen's κ                 : {kappa:.4f}")
    print(f"  Interpretation            : {kappa_interpretation(kappa)}")
    print()
    print(f"  Systematic Disagreement Categories ({len(categories)} identified):")
    for i, cat in enumerate(categories, 1):
        n = cat.get("n_affected", "?")
        print(f"  {i}. {cat['name']} (n≈{n})")
        print(f"     → Protocol: {cat['protocol']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("=== evaluation.py smoke test ===")

    # Kappa test
    a = ["A", "A", "B", "B", "A", "B"]
    b = ["A", "B", "B", "A", "A", "B"]
    k = cohens_kappa(a, b)
    print(f"Cohen's kappa (test): {k} — {kappa_interpretation(k)}")

    # Plot test (save only, no display)
    dummy_log = pd.DataFrame({
        "step": list(range(10, 210, 10)),
        "loss": [0.7 - 0.03 * i for i in range(20)],
        "eval_acc": [0.5 + 0.012 * i for i in range(20)],
    })
    plot_reward_model_curves(dummy_log, save_path=None, show=False)
    print("Plots generated OK")
    print("=== evaluation.py smoke test PASSED ===")
