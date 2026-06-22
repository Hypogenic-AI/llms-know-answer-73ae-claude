"""Experiment 3 — Rotating certainty / distributing uncertainty (claim C3, novel).

StrategyQA items are multi-hop yes/no questions with a `facts` field that decomposes
the reasoning chain. We split it into two hops (fact A, fact B) and present each item
under conditions that ROTATE which hop is held certain:
  - baseline : question only
  - +A       : "Given that <fact A>." prepended
  - +B       : "Given that <fact B>." prepended

Uncertainty is measured model-agnostically as PREDICTIVE ENTROPY over the final yes/no
answer from K stochastic samples (works for base, instruct AND the reasoning model,
whose answer lives at the end of its chain). Binary entropy H(p_yes).

Tests:
  (1) Does fixing a hop reduce answer entropy?  H(base) vs H(+A), H(+B).
  (2) ASYMMETRY: |ΔH(+A) - ΔH(+B)| -> uncertainty was concentrated on one hop, and
      rotating which is certain moves the residual uncertainty (the core C3 claim).
  (3) Base vs RL: do RL models REDISTRIBUTE LESS (rigid pre-commitment => smaller
      entropy response to injected certainty)?
"""
import os, json, argparse, time, re
import numpy as np
import torch
from src.common import (get_model, free_model, set_seed, load_dataset_local,
                        parse_yesno, RESULTS, MODELS)

GEN_TOK = {"base": 48, "instruct": 128, "rl": 512}
K = 6          # samples per (item, condition)
TEMP = 0.7


def split_facts(facts):
    """Split the facts paragraph into sentence-level hops."""
    sents = re.split(r"(?<=[.!?])\s+", (facts or "").strip())
    sents = [s.strip() for s in sents if len(s.strip()) > 10]
    return sents


def build(items):
    out = []
    for it in items:
        sents = split_facts(it.get("facts", ""))
        if len(sents) < 2:
            continue
        a = sents[0]
        b = " ".join(sents[1:]) if len(sents) <= 3 else sents[1]
        gold = "yes" if it["answer"] else "no"
        out.append(dict(qid=it["qid"], q=it["question"], factA=a, factB=b, gold=gold))
    return out


def conditions(item):
    instr = "Answer the question with only 'yes' or 'no', then a brief reason."
    base_u = f"{instr}\n\nQuestion: {item['q']}"
    a_u = f"{instr}\n\nGiven that {item['factA']}\n\nQuestion: {item['q']}"
    b_u = f"{instr}\n\nGiven that {item['factB']}\n\nQuestion: {item['q']}"
    return {"baseline": base_u, "+A": a_u, "+B": b_u}


def to_prompt(tok, user, chat):
    if chat:
        return tok.apply_chat_template([{"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)
    return user + "\nAnswer:"


@torch.no_grad()
def sample_batch(model, tok, prompts, max_new, bs=48):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                             temperature=TEMP, top_p=0.95,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            outs.append(tok.decode(gen[j][enc.input_ids.shape[1]:],
                                   skip_special_tokens=True))
    return outs


def binary_entropy(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def run_model(key, items):
    model, tok = get_model(key)
    chat = key != "base"
    conds = ["baseline", "+A", "+B"]
    # Build flat list of (item_idx, cond, prompt) replicated K times.
    flat, meta = [], []
    for idx, it in enumerate(items):
        cu = conditions(it)
        for c in conds:
            p = to_prompt(tok, cu[c], chat)
            for _ in range(K):
                flat.append(p); meta.append((idx, c))
    set_seed(123)
    texts = sample_batch(model, tok, flat, GEN_TOK[key])
    # aggregate
    agg = {}  # (idx,cond) -> list of yes/no/none
    for (idx, c), t in zip(meta, texts):
        agg.setdefault((idx, c), []).append(parse_yesno(t))
    free_model(key)

    recs = []
    for idx, it in enumerate(items):
        row = dict(qid=it["qid"], gold=it["gold"])
        ok = True
        for c in conds:
            ans = agg[(idx, c)]
            valid = [a for a in ans if a is not None]
            if len(valid) < 2:
                ok = False
                break
            p_yes = sum(a == "yes" for a in valid) / len(valid)
            row[f"pyes_{c}"] = p_yes
            row[f"H_{c}"] = binary_entropy(p_yes)
            row[f"maj_{c}"] = "yes" if p_yes >= 0.5 else "no"
            row[f"nvalid_{c}"] = len(valid)
        if ok:
            recs.append(row)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    set_seed(42)
    raw = load_dataset_local("strategyqa")
    items = build(raw)[:args.n]
    print(f"[exp3] usable items={len(items)} models={args.models} K={K}")

    out = {}
    for key in args.models:
        t0 = time.time()
        recs = run_model(key, items)
        out[key] = recs
        H0 = np.mean([r["H_baseline"] for r in recs])
        HA = np.mean([r["H_+A"] for r in recs])
        HB = np.mean([r["H_+B"] for r in recs])
        print(f"  {key}: {len(recs)} items, {time.time()-t0:.0f}s | "
              f"H_base={H0:.3f} H_+A={HA:.3f} H_+B={HB:.3f}")

    os.makedirs(RESULTS + "/model_outputs", exist_ok=True)
    with open(f"{RESULTS}/model_outputs/exp3_strategyqa.json", "w") as f:
        json.dump(dict(K=K, temp=TEMP, n=len(items), results=out), f)
    print(f"[exp3] saved {RESULTS}/model_outputs/exp3_strategyqa.json")


if __name__ == "__main__":
    main()
