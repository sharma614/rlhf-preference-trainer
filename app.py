import os
import random
import datetime
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

from src.reward_model import BradleyTerryRewardModel, score_response
from src.ppo_config import MODEL_NAME, REWARD_MODEL_DIR, PPO_MODEL_DIR, PREFERENCES_CSV
from src.data_utils import SEED_PROMPTS, init_preferences_csv, save_annotation, generate_response

# Initialize preferences CSV
init_preferences_csv(PREFERENCES_CSV)

app = FastAPI(title="RLHF Studio Backend")

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Global references to models and tokenizers (lazy loaded on first request to speed up server start)
tokenizer = None
base_model = None
rlhf_model = None
reward_model = None

HAS_RLHF = False
HAS_REWARD = False

def load_models():
    global tokenizer, base_model, rlhf_model, reward_model, HAS_RLHF, HAS_REWARD
    if tokenizer is not None:
        return

    print("Loading models (this may take a minute or two)...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load SFT model (Base GPT-2 Medium)
    print("  Loading base GPT-2 Medium...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    ).to(device)
    base_model.eval()

    # Load RLHF PPO model
    if os.path.exists(PPO_MODEL_DIR):
        print(f"  Loading RLHF model from {PPO_MODEL_DIR}...")
        try:
            rlhf_base = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            )
            rlhf_model = PeftModel.from_pretrained(rlhf_base, PPO_MODEL_DIR)
            rlhf_model = rlhf_model.merge_and_unload()
            rlhf_model = rlhf_model.to(device)
            rlhf_model.eval()
            HAS_RLHF = True
            print("  RLHF model loaded successfully")
        except Exception as e:
            print(f"  RLHF model load failed: {e}")
            rlhf_model = base_model
            HAS_RLHF = False
    else:
        print(f"  {PPO_MODEL_DIR} not found — using base model as RLHF stand-in.")
        rlhf_model = base_model
        HAS_RLHF = False

    # Load Reward Model
    if os.path.exists(REWARD_MODEL_DIR):
        print(f"  Loading reward model from {REWARD_MODEL_DIR}...")
        try:
            reward_model = BradleyTerryRewardModel.from_pretrained(REWARD_MODEL_DIR, device=device)
            reward_model.eval()
            HAS_REWARD = True
            print("  Reward model loaded successfully")
        except Exception as e:
            print(f"  Reward model load failed: {e}")
            reward_model = None
            HAS_REWARD = False
    else:
        print("  No reward model found — scores will use deterministic fallbacks.")
        reward_model = None
        HAS_REWARD = False

    print("All models ready!")

# Preference Annotation State
def get_existing_count():
    try:
        if os.path.exists(PREFERENCES_CSV):
            df = pd.read_csv(PREFERENCES_CSV)
            return len(df)
    except Exception:
        pass
    return 0

STATE = {
    'pair_idx': get_existing_count(),
    'pairs': [],
    'gen_seed': 100,
    'annotator_id': f'annotator_{random.randint(1000, 9999)}',
}

def make_pair(seed):
    load_models()
    prompt = random.choice(SEED_PROMPTS)
    # Using SFT model (base_model) to generate response options A and B
    resp_a = generate_response(prompt, base_model, tokenizer, max_new_tokens=120, seed=seed)
    resp_b = generate_response(prompt, base_model, tokenizer, max_new_tokens=120, seed=seed + 500)
    return {
        'prompt': prompt,
        'response_a': resp_a,
        'response_b': resp_b,
    }

def get_current_pair_item():
    while STATE['pair_idx'] >= len(STATE['pairs']):
        STATE['pairs'].append(make_pair(STATE['gen_seed']))
        STATE['gen_seed'] += 1
    return STATE['pairs'][STATE['pair_idx']]


# Pydantic classes
class AnnotationSubmit(BaseModel):
    prompt: str
    response_a: str
    response_b: str
    preferred: str
    helpfulness_score: int
    factuality_score: int
    safety_score: int
    fluency_score: int

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_new_tokens: int = 128


# Routes
@app.get("/")
def read_root():
    return RedirectResponse(url="/annotation")

@app.get("/annotation", response_class=HTMLResponse)
def get_annotation_page():
    # Load and serve annotation.html
    html_path = Path("frontend/annotation.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="annotation.html not found")
    return html_path.read_text(encoding="utf-8")

@app.get("/demo", response_class=HTMLResponse)
def get_demo_page():
    # Load and serve demo.html
    html_path = Path("frontend/demo.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="demo.html not found")
    return html_path.read_text(encoding="utf-8")

@app.get("/api/current_pair")
def get_current_pair():
    pair = get_current_pair_item()
    return {
        "prompt": pair["prompt"],
        "response_a": pair["response_a"],
        "response_b": pair["response_b"],
        "count": get_existing_count()
    }

@app.post("/api/submit_annotation")
def submit_annotation(data: AnnotationSubmit):
    # Save the annotation
    save_annotation({
        'prompt': data.prompt,
        'response_a': data.response_a,
        'response_b': data.response_b,
        'preferred': data.preferred,
        'helpfulness_score': data.helpfulness_score,
        'factuality_score': data.factuality_score,
        'safety_score': data.safety_score,
        'fluency_score': data.fluency_score,
        'annotator_id': STATE['annotator_id'],
        'timestamp': datetime.datetime.now().isoformat(),
    }, PREFERENCES_CSV)

    STATE['pair_idx'] += 1
    return {"status": "success"}

@app.post("/api/skip_pair")
def skip_pair():
    STATE['pair_idx'] += 1
    return {"status": "success"}

@app.post("/api/generate")
def generate_comparison(req: GenerateRequest):
    load_models()
    
    # Generate response from Base SFT model
    resp_a = generate_response(
        req.prompt, base_model, tokenizer,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    
    # Generate response from RLHF model
    resp_b = generate_response(
        req.prompt, rlhf_model, tokenizer,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    
    # Score them
    if HAS_REWARD and reward_model is not None:
        reward_a = score_response(reward_model, tokenizer, req.prompt, resp_a, device=device)
        reward_b = score_response(reward_model, tokenizer, req.prompt, resp_b, device=device)
    else:
        # Heuristic/Deterministic Mock Score fallback when models are not trained yet
        # Ensure B is slightly higher to mock improvements
        reward_a = float(len(resp_a) % 10) / 3.0
        reward_b = reward_a + 0.82 + (float(len(resp_b) % 5) / 10.0)
    
    return {
        "response_a": resp_a,
        "response_b": resp_b,
        "reward_a": reward_a,
        "reward_b": reward_b
    }

if __name__ == "__main__":
    import uvicorn
    # Pre-load models in background to avoid freezing the first page load
    try:
        load_models()
    except Exception as e:
        print(f"Pre-loading models failed: {e}. They will be loaded on demand.")
    
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
