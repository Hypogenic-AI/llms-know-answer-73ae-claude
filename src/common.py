"""Shared utilities: model loading, dataset loading, answer parsing, seeding.

All experiments import from here so model handling and parsing are identical
across base / instruct / RL-reasoning models (fair apples-to-apples comparison).
"""
import os, json, random, re
import numpy as np

# Pin to free GPUs (GPU index 2 was busy at session start). Adjust if needed.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,3")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

# The matched post-training gradient (same Qwen2.5 family lineage).
MODELS = {
    "base":     "Qwen/Qwen2.5-1.5B",
    "instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "rl":       "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}
MODEL_LABEL = {  # for plots
    "base": "Base (Qwen2.5-1.5B)",
    "instruct": "Instruct (RLHF)",
    "rl": "RL-reasoning (R1-Distill)",
}


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_dataset_local(name):
    """Return list of dicts for a local dataset."""
    if name == "arc":
        return load_jsonl(os.path.join(DATA, "arc_challenge", "test.jsonl"))
    if name == "gsm8k":
        return load_jsonl(os.path.join(DATA, "gsm8k", "test.jsonl"))
    if name == "strategyqa":
        return load_jsonl(os.path.join(DATA, "strategyqa", "test.jsonl"))
    if name == "hotpotqa":
        return load_jsonl(os.path.join(DATA, "hotpotqa", "sample2000.jsonl"))
    raise ValueError(name)


_CACHE = {}

def get_model(key, output_hidden_states=False):
    """Load (and cache) a model + tokenizer by MODELS key."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if key in _CACHE:
        return _CACHE[key]
    name = MODELS[key]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="cuda:0",
        output_hidden_states=output_hidden_states,
    )
    model.eval()
    _CACHE[key] = (model, tok)
    return model, tok


def free_model(key):
    if key in _CACHE:
        del _CACHE[key]
        torch.cuda.empty_cache()


# ---------------- Answer parsing ----------------

def parse_mc(text, labels=("A", "B", "C", "D", "E")):
    """Extract a multiple-choice letter from free text. Returns letter or None.

    Strategy: prefer explicit 'answer is X' / 'answer: X'; else last standalone
    letter mention. Robust to both base (continuation) and RL (verbose) outputs.
    """
    if not text:
        return None
    labs = "".join(labels)
    # Explicit answer markers (take the LAST such mention).
    pats = [
        rf"answer\s*(?:is|:)?\s*\(?([{labs}])\b",
        rf"answer\s*(?:is|:)?\s*\(?([{labs}])\)",
        rf"\\boxed\{{\s*([{labs}])\s*\}}",
        rf"\boption\s*\(?([{labs}])\b",
    ]
    best = None
    for p in pats:
        for m in re.finditer(p, text, re.IGNORECASE):
            best = m.group(1).upper()
    if best:
        return best
    # Fallback: last standalone capital letter that is a valid label.
    cands = re.findall(rf"(?<![A-Za-z])([{labs}])(?![A-Za-z])", text)
    return cands[-1].upper() if cands else None


def parse_yesno(text):
    """Return 'yes'/'no'/None from free text (last explicit mention)."""
    if not text:
        return None
    best = None
    for m in re.finditer(r"\b(yes|no|true|false)\b", text, re.IGNORECASE):
        w = m.group(1).lower()
        best = "yes" if w in ("yes", "true") else "no"
    return best


def parse_number(text):
    """Extract the final numeric answer (gsm8k style)."""
    if not text:
        return None
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def gsm8k_gold(answer_field):
    return parse_number(answer_field)
