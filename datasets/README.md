# Downloaded Datasets

Benchmarks selected to test the hypothesis. Each is small (<500 KB) so samples are committed;
full data is reproducible from the snippets below. Data files are git-ignored (see `.gitignore`);
`samples.json` files and this README are kept.

Rationale for selection:
- **Decomposable multi-hop QA** (HotpotQA, StrategyQA): naturally splits into sub-questions, so
  we can "rotate what is certain" and test distributing uncertainty across parts.
- **Clean multiple-choice** (ARC-Challenge): discrete answer ⇒ ideal for the FlipFlop / answer-
  commitment paradigm and for probing a pre-committed answer token.
- **Math reasoning** (GSM8K): canonical RL-vs-base reasoning benchmark; pairs with the reasoning-
  strength-planning and UFO setups.

These complement (do not replace) the benchmarks named across the literature — MATH/MATH500,
AIME2024, GPQA, MMLU/MMLU-Pro, TheoremQA, TruthfulQA, SciQ, MBPP/HumanEval, CommonsenseQA,
SocialIQA, MedMCQA, CREAK/FEVER — which the experiment runner can pull on demand the same way.

---

## 1. GSM8K (`gsm8k/`)
- **Source**: HuggingFace `openai/gsm8k` (config `main`), split `test` (1,319 problems)
- **Format**: JSONL, fields `question`, `answer` (answer ends with `#### <number>`)
- **Task**: grade-school multi-step arithmetic word problems
- **Download**:
  ```python
  from datasets import load_dataset
  ds = load_dataset("openai/gsm8k", "main", split="test")
  ds.to_json("datasets/gsm8k/test.jsonl")
  ```

## 2. HotpotQA (`hotpotqa/`)
- **Source**: HuggingFace `hotpotqa/hotpot_qa` (config `distractor`), split `validation`
  (2,000-record sample saved here; full validation = 7,405)
- **Format**: JSONL, fields `id`, `question`, `answer`, `type` (comparison/bridge),
  `level` (easy/medium/hard), `supporting_facts`
- **Task**: 2-hop Wikipedia QA — decomposable into sub-questions
- **Download** (needs `trust_remote_code=True`):
  ```python
  from datasets import load_dataset
  ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", trust_remote_code=True)
  ```

## 3. ARC-Challenge (`arc_challenge/`)
- **Source**: HuggingFace `allenai/ai2_arc` (config `ARC-Challenge`), split `test` (1,172)
- **Format**: JSONL, fields `id`, `question`, `choices` ({`text`,`label`}), `answerKey`
- **Task**: hard grade-school science multiple-choice (clean answer-commitment testbed; also
  used by both Emergent Response Planning and FlipFlop)
- **Download**:
  ```python
  from datasets import load_dataset
  ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
  ```

## 4. StrategyQA (`strategyqa/`)
- **Source**: HuggingFace `ChilleD/StrategyQA`, split `test` (687)
- **Format**: JSONL, fields `qid`, `term`, `description`, `question`, `answer` (yes/no), `facts`
- **Task**: implicit multi-hop yes/no reasoning requiring decomposition into reasoning steps
- **Download**:
  ```python
  from datasets import load_dataset
  ds = load_dataset("ChilleD/StrategyQA", split="test")
  ```

## Loading any saved file
```python
import json
recs = [json.loads(l) for l in open("datasets/gsm8k/test.jsonl")]      # full split
sample = json.load(open("datasets/gsm8k/samples.json"))                 # first 10
```

`_download_report.json` records counts, splits, and field names for all four datasets.
Regenerate everything with `python download_datasets.py` from the workspace root.
