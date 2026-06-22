"""Download small samples of benchmarks for testing the hypothesis.
Saves full small splits where cheap, otherwise streamed samples + sample JSON.
"""
import json, os, traceback
os.makedirs("datasets", exist_ok=True)
from datasets import load_dataset

def save_samples(name, recs, n=10):
    os.makedirs(f"datasets/{name}", exist_ok=True)
    with open(f"datasets/{name}/samples.json", "w") as f:
        json.dump(recs[:n], f, indent=2, default=str)
    print(f"  saved {len(recs)} recs -> datasets/{name}/ (sample of {min(n,len(recs))})")

report = {}

# 1. GSM8K - grade-school math, canonical RL-vs-base reasoning benchmark
try:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    recs = [dict(ds[i]) for i in range(len(ds))]
    ds.to_json("datasets/gsm8k/test.jsonl")
    save_samples("gsm8k", recs)
    report["gsm8k"] = {"n": len(recs), "split": "test", "fields": list(recs[0].keys())}
except Exception as e:
    report["gsm8k"] = {"error": str(e)}; traceback.print_exc()

# 2. 2WikiMultihopQA-style multi-hop QA, naturally decomposable into sub-questions
try:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    recs = [{"id": ds[i]["id"], "question": ds[i]["question"], "answer": ds[i]["answer"],
             "type": ds[i].get("type"), "level": ds[i].get("level"),
             "supporting_facts": ds[i].get("supporting_facts")} for i in range(min(2000, len(ds)))]
    save_samples("hotpotqa", recs)
    with open("datasets/hotpotqa/sample2000.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r, default=str)+"\n")
    report["hotpotqa"] = {"n": len(recs), "split": "validation(sample)", "fields": list(recs[0].keys())}
except Exception as e:
    report["hotpotqa"] = {"error": str(e)}; traceback.print_exc()

# 3. ARC-Challenge multiple choice - clean for answer-commitment / FlipFlop paradigm
try:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    recs = [dict(ds[i]) for i in range(len(ds))]
    ds.to_json("datasets/arc_challenge/test.jsonl")
    save_samples("arc_challenge", recs)
    report["arc_challenge"] = {"n": len(recs), "split": "test", "fields": list(recs[0].keys())}
except Exception as e:
    report["arc_challenge"] = {"error": str(e)}; traceback.print_exc()

# 4. StrategyQA - implicit multi-hop yes/no, requires decomposition
try:
    ds = load_dataset("ChilleD/StrategyQA", split="test")
    recs = [dict(ds[i]) for i in range(len(ds))]
    save_samples("strategyqa", recs)
    with open("datasets/strategyqa/test.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r, default=str)+"\n")
    report["strategyqa"] = {"n": len(recs), "fields": list(recs[0].keys())}
except Exception as e:
    report["strategyqa"] = {"error": str(e)}; traceback.print_exc()

print("\n=== REPORT ===")
print(json.dumps(report, indent=2))
json.dump(report, open("datasets/_download_report.json", "w"), indent=2)
