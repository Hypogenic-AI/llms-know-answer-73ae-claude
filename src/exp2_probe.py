"""Experiment 2 — Answer-commitment earliness probe (claims C1 + C2a).

For each ARC item and each model:
  1. Greedy-generate the answer; parse the model's OWN final letter (the answer it
     "wants to give").
  2. Forward-pass the full [prompt + generation]; extract last-token hidden states
     at (a) the prompt-end position for ALL layers, and (b) a chosen mid-layer at
     5 relative points along the generation timeline (0/25/50/75/100%).
  3. Train a logistic-regression probe to predict the model's final answer letter
     from each hidden state (80/20 split).

C1: probe accuracy >> chance at the prompt position => the answer is pre-encoded.
C2a: compare prompt-position accuracy base vs instruct vs RL => does post-training
     make the answer decodable EARLIER (stronger pre-commitment)?

Reuses the Emergent-Response-Planning probing protocol (prompt-position MLP/linear
probe predicting the final answer), extended to a matched base-vs-RL contrast.
"""
import os, json, argparse, time
import numpy as np
import torch
from src.common import (get_model, free_model, set_seed, load_dataset_local,
                        parse_mc, RESULTS, MODELS)

GEN_TOK = {"base": 64, "instruct": 200, "rl": 640}  # enough to reach the answer
FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]


def build_arc(items):
    out = []
    for it in items:
        labels = it["choices"]["label"]
        texts = it["choices"]["text"]
        if list(labels) != ["A", "B", "C", "D"]:
            continue  # keep clean A-D 4-way for the classifier (some ARC use 1-4)
        body = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
        instr = ("Answer the multiple-choice science question. "
                 "End with 'Answer: <letter>'.")
        plain = f"{instr}\n\nQuestion: {it['question']}\n{body}\nAnswer:"
        chat_user = f"{instr}\n\nQuestion: {it['question']}\n{body}"
        out.append(dict(qid=it["id"], plain=plain, chat_user=chat_user,
                        labels=labels, gold=it["answerKey"]))
    return out


@torch.no_grad()
def _batch_generate(model, tok, prompts, max_new, bs=48):
    """Batched greedy generation; returns list of (decoded_new_text, full_ids_1d)."""
    tok.padding_side = "left"
    res = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        plen = enc.input_ids.shape[1]
        for j in range(len(chunk)):
            full = gen[j]
            new = full[plen:]
            # strip left-padding from the front for the clean (unpadded) sequence
            attn = enc.attention_mask[j]
            real_prompt = enc.input_ids[j][attn.bool()]
            clean_full = torch.cat([real_prompt, new[new != tok.pad_token_id]])
            res.append((tok.decode(new, skip_special_tokens=True),
                        clean_full, len(real_prompt)))
    return res


@torch.no_grad()
def extract(key, items, layer_frac=0.5):
    model, tok = get_model(key, output_hidden_states=True)
    chat = key != "base"
    nlayers = model.config.num_hidden_layers
    mid = int(nlayers * layer_frac)

    prompts = []
    for it in items:
        if chat:
            prompts.append(tok.apply_chat_template(
                [{"role": "user", "content": it["chat_user"]}],
                tokenize=False, add_generation_prompt=True))
        else:
            prompts.append(it["plain"])

    # Step 1: batched greedy generation (fast).
    gens = _batch_generate(model, tok, prompts, GEN_TOK[key])

    # Step 2: per-item single forward pass for hidden states (cheap, no generation).
    rows = []
    for it, (new_text, clean_full, plen) in zip(items, gens):
        ans = parse_mc(new_text, labels=tuple(it["labels"]))
        if ans is None:
            continue
        out = model(clean_full.unsqueeze(0).to(model.device))
        hs = out.hidden_states  # tuple (nlayers+1) of [1, seq, H]
        pe = plen - 1           # prompt-end position
        prompt_all = np.stack([hs[l][0, pe].float().cpu().numpy()
                               for l in range(len(hs))])  # [L+1, H]
        G = clean_full.shape[0] - plen
        timeline = []
        for fr in FRACS:
            pos = min(plen - 1 + int(round(fr * max(G, 1))), clean_full.shape[0] - 1)
            timeline.append(hs[mid][0, pos].float().cpu().numpy())
        timeline = np.stack(timeline)  # [5, H]
        rows.append(dict(answer=ans, gold=it["gold"], correct=int(ans == it["gold"]),
                         prompt_all=prompt_all, timeline=timeline))
    free_model(key)
    return rows, mid, nlayers


def probe_accuracy(X, y, seed=42):
    """Balanced 80/20 logistic-regression probe accuracy + chance baseline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score
    y = np.asarray(y)
    X = np.asarray(X)
    # Drop classes with <2 members (can't be stratified); they are parse artifacts.
    classes, counts = np.unique(y, return_counts=True)
    keep_classes = set(classes[counts >= 2])
    mask = np.array([v in keep_classes for v in y])
    X, y = X[mask], y[mask]
    classes = np.unique(y)
    if len(classes) < 2:
        return None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed,
                                          stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    acc = balanced_accuracy_score(yte, pred)
    chance = 1.0 / len(classes)
    return dict(acc=float(acc), chance=float(chance), n=len(y),
                n_test=len(yte), nclasses=int(len(classes)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    set_seed(42)
    raw = load_dataset_local("arc")[:args.n]
    items = build_arc(raw)
    print(f"[exp2] N={len(items)} models={args.models}")

    summary = {}
    for key in args.models:
        t0 = time.time()
        rows, mid, nlayers = extract(key, items)
        print(f"  {key}: extracted {len(rows)}/{len(items)} (parsed), "
              f"mid-layer={mid}/{nlayers}, {time.time()-t0:.0f}s")
        y = [r["answer"] for r in rows]
        # layer sweep at prompt position
        prompt_all = np.stack([r["prompt_all"] for r in rows])  # [N, L+1, H]
        layer_acc = []
        for l in range(prompt_all.shape[1]):
            res = probe_accuracy(prompt_all[:, l, :], y)
            layer_acc.append(res["acc"] if res else None)
        # timeline at mid layer
        timeline = np.stack([r["timeline"] for r in rows])  # [N,5,H]
        tl_acc = [probe_accuracy(timeline[:, i, :], y) for i in range(len(FRACS))]
        # best-layer prompt-position probe (report) + correctness probe
        best_layer = int(np.nanargmax([a if a is not None else -1 for a in layer_acc]))
        res_best = probe_accuracy(prompt_all[:, best_layer, :], y)
        corr_res = probe_accuracy(prompt_all[:, best_layer, :],
                                  [r["correct"] for r in rows])
        summary[key] = dict(
            n_parsed=len(rows), nlayers=nlayers, mid_layer=mid,
            chance=1.0 / len(set(y)),
            self_consistency_acc=float(np.mean([r["correct"] for r in rows])),
            layer_acc=layer_acc, best_layer=best_layer,
            prompt_probe_best=res_best,
            timeline_fracs=FRACS,
            timeline_acc=[(r["acc"] if r else None) for r in tl_acc],
            correctness_probe=corr_res,
        )
        print(f"    prompt-pos best-layer acc={res_best['acc']:.3f} "
              f"(chance {res_best['chance']:.3f}); "
              f"timeline={[round(r['acc'],3) if r else None for r in tl_acc]}")

    os.makedirs(RESULTS + "/evaluations", exist_ok=True)
    with open(f"{RESULTS}/evaluations/exp2_probe.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[exp2] saved {RESULTS}/evaluations/exp2_probe.json")


if __name__ == "__main__":
    main()
