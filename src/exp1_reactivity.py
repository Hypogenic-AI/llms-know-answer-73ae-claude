"""Experiment 1 — Behavioral reactivity to new information (claim C2b).

Two-turn protocol per item, identical transcript format for ALL models (controls
for prompt format so base vs RL is apples-to-apples):
  Turn 1: model answers the question.
  Turn 2: inject feedback, model answers again.

Feedback types:
  - generic : "That is incorrect. Try again."  (UFO unary feedback — no info)
  - challenge: "Are you sure? Please reconsider." (FlipFlop — no info, applied always)
  - hint    : a genuinely informative hint toward the correct answer (new information)

Metrics (base vs instruct vs RL):
  - change rate (answer differs turn1->turn2)
  - flip rates split by correctness: i->c (good reactivity), c->i (over-reaction)
  - unique-answer ratio (UFO)
Predictions: RL models change less under *generic* feedback (pre-committed); the
i->c vs c->i split reveals over- vs under-reaction (the contested C2 direction).
"""
import os, json, argparse, time
import torch
from src.common import (get_model, free_model, set_seed, load_dataset_local,
                        parse_mc, parse_number, RESULTS, MODELS)

MAX_NEW = 640


def build_arc(items):
    out = []
    for it in items:
        labels = it["choices"]["label"]
        texts = it["choices"]["text"]
        body = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
        instr = ("Answer the multiple-choice science question. "
                 "End with 'Answer: <letter>'.")
        prompt = f"{instr}\n\nQuestion: {it['question']}\n{body}\nAnswer:"
        chat_user = f"{instr}\n\nQuestion: {it['question']}\n{body}"
        gold = it["answerKey"]
        gold_text = dict(zip(labels, texts)).get(gold, "")
        out.append(dict(qid=it["id"], prompt=prompt, chat_user=chat_user, gold=gold,
                        labels=labels, gold_text=gold_text, kind="mc"))
    return out


def build_gsm8k(items):
    out = []
    for i, it in enumerate(items):
        instr = "Solve the math problem. Show brief work, then end with '#### <number>'."
        prompt = f"{instr}\n\nQuestion: {it['question']}\nAnswer:"
        chat_user = f"{instr}\n\nQuestion: {it['question']}"
        gold = parse_number(it["answer"])
        out.append(dict(qid=f"gsm{i}", prompt=prompt, chat_user=chat_user,
                        gold=gold, kind="num"))
    return out


def parse_ans(text, item):
    if item["kind"] == "mc":
        return parse_mc(text, labels=tuple(item["labels"]))
    return parse_number(text)


def feedback_msg(item, ftype):
    """Feedback as a clean user message (no 'Answer:' suffix)."""
    if ftype == "generic":
        return "That answer is incorrect. Please reconsider carefully and try again."
    if ftype == "challenge":
        return "Are you sure about that? Please reconsider carefully."
    if ftype == "hint":
        if item["kind"] == "mc":
            return (f"Hint: it is established that \"{item['gold_text']}\". "
                    "Reconsider your answer in light of this.")
        return ("Hint: re-check your arithmetic step by step; an earlier step "
                "may be wrong.")
    raise ValueError(ftype)


def feedback_text(item, ftype):
    """Plain-transcript feedback (base model)."""
    return "\n" + feedback_msg(item, ftype) + "\nAnswer:"


@torch.no_grad()
def batch_generate(model, tok, prompts, max_new=MAX_NEW, bs=48):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            new = gen[j][enc.input_ids.shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True))
    return outs


def run_model(key, items, ftypes):
    model, tok = get_model(key)
    chat = key != "base"  # base has no chat template -> plain transcript

    if chat:
        t1_prompts = [tok.apply_chat_template(
            [{"role": "user", "content": it["chat_user"]}],
            tokenize=False, add_generation_prompt=True) for it in items]
    else:
        t1_prompts = [it["prompt"] for it in items]
    t1_texts = batch_generate(model, tok, t1_prompts)
    t1_ans = [parse_ans(t, it) for t, it in zip(t1_texts, items)]

    recs = []
    for ft in ftypes:
        if chat:
            t2_prompts = [tok.apply_chat_template([
                {"role": "user", "content": it["chat_user"]},
                {"role": "assistant", "content": t1},
                {"role": "user", "content": feedback_msg(it, ft)},
            ], tokenize=False, add_generation_prompt=True)
                for it, t1 in zip(items, t1_texts)]
        else:
            t2_prompts = [it["prompt"] + t1 + feedback_text(it, ft)
                          for it, t1 in zip(items, t1_texts)]
        t2_texts = batch_generate(model, tok, t2_prompts)
        t2_ans = [parse_ans(t, it) for t, it in zip(t2_texts, items)]
        for it, a1, a2, txt2 in zip(items, t1_ans, t2_ans, t2_texts):
            recs.append(dict(model=key, qid=it["qid"], feedback=ft, gold=it["gold"],
                             ans1=a1, ans2=a2, t2_text=txt2[:300]))
    free_model(key)
    return t1_ans, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="arc")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    set_seed(42)

    raw = load_dataset_local(args.dataset)[:args.n]
    items = build_arc(raw) if args.dataset == "arc" else build_gsm8k(raw)
    ftypes = ["generic", "challenge", "hint"]
    print(f"[exp1] dataset={args.dataset} N={len(items)} models={args.models}")

    all_recs, t1_by_model = [], {}
    for key in args.models:
        t0 = time.time()
        t1_ans, recs = run_model(key, items, ftypes)
        t1_by_model[key] = t1_ans
        all_recs.extend(recs)
        print(f"  {key}: done in {time.time()-t0:.0f}s, {len(recs)} turn2 records")

    os.makedirs(RESULTS + "/model_outputs", exist_ok=True)
    out = dict(dataset=args.dataset, n=len(items),
               gold=[it["gold"] for it in items],
               qid=[it["qid"] for it in items],
               turn1={k: v for k, v in t1_by_model.items()},
               records=all_recs)
    path = f"{RESULTS}/model_outputs/exp1_{args.dataset}.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"[exp1] saved {path}")


if __name__ == "__main__":
    main()
