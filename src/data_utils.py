import os
import csv
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional


SEED_PROMPTS = [
    "Explain how neural networks learn from data.",
    "What are the main causes of climate change?",
    "How does a vaccine work in the human body?",
    "Describe the water cycle in simple terms.",
    "What is the difference between machine learning and deep learning?",
    "How do computers store information?",
    "Explain the concept of supply and demand.",
    "What causes earthquakes and how are they measured?",
    "How does the internet work?",
    "What is CRISPR and how does it edit genes?",
    "Explain photosynthesis to a 10-year-old.",
    "What are the benefits and risks of artificial intelligence?",
    "How does encryption protect data online?",
    "Describe the process of how stars are formed.",
    "What is the role of the immune system?",
    "Explain quantum computing in simple terms.",
    "How do social media algorithms work?",
    "What is the greenhouse effect?",
    "How does the stock market work?",
    "Explain the difference between DNA and RNA.",
    "What is machine learning bias and why does it matter?",
    "How do self-driving cars navigate roads?",
    "What is blockchain technology?",
    "Explain how antibiotics work.",
    "How does the human brain process language?",
    "What is the difference between renewable and non-renewable energy?",
    "How do search engines rank websites?",
    "What is natural language processing?",
    "Explain the Big Bang theory.",
    "How does GPS determine your location?",
]

PREFERENCE_COLUMNS = [
    "prompt",
    "response_a",
    "response_b",
    "preferred",
    "helpfulness_score",
    "factuality_score",
    "safety_score",
    "fluency_score",
    "annotator_id",
    "timestamp",
]


def init_preferences_csv(csv_path: str) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PREFERENCE_COLUMNS)
            writer.writeheader()
        print(f"Initialized new preferences CSV at {csv_path}")
    else:
        existing = pd.read_csv(csv_path)
        print(f"Found existing CSV with {len(existing)} annotations.")


def save_annotation(row: Dict, csv_path: str) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREFERENCE_COLUMNS)
        if not file_exists:
            writer.writeheader()
        safe_row = {col: row.get(col, "") for col in PREFERENCE_COLUMNS}
        writer.writerow(safe_row)


def load_preferences(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Preferences CSV not found at {csv_path}. "
            "Run notebook 01_annotation_ui.ipynb first."
        )
    df = pd.read_csv(csv_path)
    df = df[df["preferred"].isin(["A", "B", "Tie"])].copy()
    df_no_ties = df[df["preferred"] != "Tie"].copy()
    print(
        f"Loaded {len(df)} annotations "
        f"({len(df_no_ties)} usable pairs after removing ties)."
    )
    return df_no_ties


def build_reward_dataset(df: pd.DataFrame) -> Dict[str, List]:
    prompts, chosen_texts, rejected_texts = [], [], []
    for _, row in df.iterrows():
        prompt = str(row["prompt"])
        resp_a = str(row["response_a"])
        resp_b = str(row["response_b"])
        if row["preferred"] == "A":
            chosen, rejected = resp_a, resp_b
        else:
            chosen, rejected = resp_b, resp_a
        prompts.append(prompt)
        chosen_texts.append(f"{prompt}\n\n{chosen}")
        rejected_texts.append(f"{prompt}\n\n{rejected}")
    return {"prompt": prompts, "chosen": chosen_texts, "rejected": rejected_texts}


def split_dataset(
    data: Dict[str, List],
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict, Dict]:
    random.seed(seed)
    n = len(data["prompt"])
    indices = list(range(n))
    random.shuffle(indices)
    split_idx = int(n * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    def _subset(idx_list):
        return {k: [v[i] for i in idx_list] for k, v in data.items()}

    train_data = _subset(train_idx)
    test_data = _subset(test_idx)
    print(f"Train: {len(train_data['prompt'])}, Test: {len(test_data['prompt'])}")
    return train_data, test_data


def generate_response(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 150,
    temperature: float = 0.9,
    top_p: float = 0.92,
    seed: Optional[int] = None,
) -> str:
    import torch
    if seed is not None:
        torch.manual_seed(seed)
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=256
    ).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_prompt_pairs(
    model,
    tokenizer,
    prompts: List[str],
    n_pairs: int = 20,
    seed_a: int = 42,
    seed_b: int = 99,
) -> List[Dict]:
    pairs = []
    selected = random.sample(prompts, min(n_pairs, len(prompts)))
    print(f"Generating {len(selected)} prompt pairs...")
    for i, prompt in enumerate(selected):
        resp_a = generate_response(prompt, model, tokenizer, seed=seed_a + i)
        resp_b = generate_response(prompt, model, tokenizer, seed=seed_b + i)
        pairs.append({"prompt": prompt, "response_a": resp_a, "response_b": resp_b})
    return pairs


def generate_synthetic_annotations(
    df: pd.DataFrame,
    agreement_rate: float = 0.67,
    seed: int = 123,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    choices = df["preferred"].values.copy()
    n_disagree = int(len(choices) * (1 - agreement_rate))
    disagree_idx = rng.choice(len(choices), size=n_disagree, replace=False)
    secondary = choices.copy()
    for idx in disagree_idx:
        current = secondary[idx]
        options = [x for x in ["A", "B"] if x != current]
        secondary[idx] = rng.choice(options)
    return pd.Series(secondary, name="annotator_2")
