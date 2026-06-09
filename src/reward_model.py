import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM


class BradleyTerryRewardModel(nn.Module):
    """GPT-2 Medium backbone with a scalar linear reward head."""

    def __init__(self, model_name: str = "gpt2-medium", freeze_backbone: bool = False):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name, output_hidden_states=True
        )
        hidden_size = self.backbone.config.n_embd
        self.reward_head = nn.Linear(hidden_size, 1, bias=True)
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.hidden_states[-1]
        seq_len = attention_mask.sum(dim=1) - 1
        B = input_ids.shape[0]
        last_h = hidden[torch.arange(B, device=input_ids.device), seq_len]
        return self.reward_head(last_h).squeeze(-1)

    def save_pretrained(self, save_dir: str) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(path))
        torch.save(self.reward_head.state_dict(), str(path / "reward_head.pt"))
        print(f"Saved to {save_dir}")

    @classmethod
    def from_pretrained(cls, save_dir: str, device: str = "cpu") -> "BradleyTerryRewardModel":
        m = cls.__new__(cls)
        super(BradleyTerryRewardModel, m).__init__()
        m.backbone = AutoModelForCausalLM.from_pretrained(save_dir, output_hidden_states=True)
        hidden_size = m.backbone.config.n_embd
        m.reward_head = nn.Linear(hidden_size, 1, bias=True)
        m.reward_head.load_state_dict(
            torch.load(Path(save_dir) / "reward_head.pt", map_location=device)
        )
        return m.to(device)


def bradley_terry_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(r_chosen - r_rejected).mean()


class PreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, data: Dict[str, List[str]], tokenizer, max_length: int = 512):
        self.chosen = data["chosen"]
        self.rejected = data["rejected"]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.chosen)

    def _enc(self, text: str) -> Dict[str, torch.Tensor]:
        out = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in out.items()}

    def __getitem__(self, idx):
        c = self._enc(self.chosen[idx])
        r = self._enc(self.rejected[idx])
        return {
            "chosen_input_ids": c["input_ids"],
            "chosen_attention_mask": c["attention_mask"],
            "rejected_input_ids": r["input_ids"],
            "rejected_attention_mask": r["attention_mask"],
        }


def evaluate_reward_model(model, loader, device: str) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            r_c = model(batch["chosen_input_ids"].to(device), batch["chosen_attention_mask"].to(device))
            r_r = model(batch["rejected_input_ids"].to(device), batch["rejected_attention_mask"].to(device))
            correct += (r_c > r_r).sum().item()
            total += r_c.shape[0]
    model.train()
    return correct / total if total > 0 else 0.0


def train_reward_model(
    model: BradleyTerryRewardModel,
    tokenizer,
    train_data: Dict,
    eval_data: Dict,
    save_dir: str = "reward_model",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    grad_clip: float = 1.0,
    log_every: int = 20,
    device: Optional[str] = None,
) -> List[Dict]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    model = model.to(device)
    train_ds = PreferenceDataset(train_data, tokenizer, max_length)
    eval_ds = PreferenceDataset(eval_data, tokenizer, max_length)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(train_loader)
    )

    logs = []
    step = 0
    for epoch in range(num_epochs):
        for batch in train_loader:
            r_c = model(batch["chosen_input_ids"].to(device), batch["chosen_attention_mask"].to(device))
            r_r = model(batch["rejected_input_ids"].to(device), batch["rejected_attention_mask"].to(device))
            loss = bradley_terry_loss(r_c, r_r)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % log_every == 0:
                acc = evaluate_reward_model(model, eval_loader, device)
                logs.append({"epoch": epoch + 1, "step": step, "loss": round(loss.item(), 4), "eval_acc": round(acc, 4)})
                print(f"  E{epoch+1} S{step} | loss={loss.item():.4f} | eval_acc={acc:.4f}")

        print(f"Epoch {epoch+1}/{num_epochs} done")

    model.save_pretrained(save_dir)
    import pandas as pd
    pd.DataFrame(logs).to_csv(os.path.join(save_dir, "training_log.csv"), index=False)
    return logs


def score_response(
    model, tokenizer, prompt: str, response: str,
    max_length: int = 512, device: Optional[str] = None
) -> float:
    if device is None:
        device = str(next(model.parameters()).device)
    enc = tokenizer(f"{prompt}\n\n{response}", truncation=True, max_length=max_length, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        r = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return r.item()
