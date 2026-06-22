# Code Walkthrough

All experiments use **real local Qwen2.5-1.5B-family models** (no simulation) on GPU.

## Structure
```
src/
├── common.py            # model/dataset loading, answer parsing, seeding (shared)
├── exp1_reactivity.py   # Exp 1: two-turn behavioral reactivity (C2)
├── exp2_probe.py        # Exp 2: answer-commitment earliness probe (C1+C2a)
├── exp3_rotating.py     # Exp 3: rotating-certainty / uncertainty redistribution (C3)
└── analyze.py           # metrics, statistical tests, figures (reads results/, writes figures/)
```

## Models (matched post-training gradient, same lineage)
| key | HF id | post-training |
|-----|-------|---------------|
| `base` | `Qwen/Qwen2.5-1.5B` | none (pretrained) |
| `instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | SFT + RLHF/DPO |
| `rl` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | distilled from R1 RL reasoning traces |

Each model is run in its **native interface**: base = plain transcript (no chat template
exists), instruct/rl = their chat templates (multi-turn `apply_chat_template`). This is how
each is actually deployed and what the UFO/FlipFlop protocols do.

## Key components

### `common.py`
- `get_model(key, output_hidden_states=False)` — loads + caches model/tokenizer in bf16 on
  `cuda:0` (GPUs pinned to 0,1,3 since GPU 2 was busy). `free_model` clears the cache.
- `parse_mc / parse_yesno / parse_number` — robust answer extraction (prefer explicit
  "Answer: X" / `\boxed{}`, else last valid mention). Handles both terse base completions and
  verbose reasoning outputs. Parse-failure rate is logged, never silently dropped.

### `exp1_reactivity.py`
Two-turn protocol. Turn 1: model answers. Turn 2: inject feedback ∈ {`generic` (UFO unary
"that's incorrect, try again"), `challenge` (FlipFlop "are you sure?"), `hint` (genuine new
information toward the correct answer)}. Batched greedy generation (`do_sample=False`, bs=48,
640 new tokens to let the reasoning model finish). Saves per-item turn-1/turn-2 answers.

### `exp2_probe.py`
For each ARC item: greedy-generate → parse the model's **own** final letter (the answer it
"wants to give"). Forward-pass `[prompt+generation]` with `output_hidden_states`; take the
last-token hidden state at the **prompt-end** position for every layer, plus a mid-layer state
at 5 points along the generation timeline. A balanced logistic-regression probe (80/20 split,
`StandardScaler`) predicts the final letter from each state. **Prompt-position accuracy ≫ chance
⇒ the answer is pre-encoded before any token is emitted** (C1); comparing across models tests
whether post-training makes it decodable earlier (C2a).

### `exp3_rotating.py`
StrategyQA `facts` split into two hops (A, B). Conditions rotate which hop is held certain:
`baseline`, `+A` ("Given that <fact A> …"), `+B`. Uncertainty = **predictive entropy** of the
final yes/no answer over K=6 stochastic samples (T=0.7) — model-agnostic, so it works for the
reasoning model whose answer lives at the end of its chain. Tests: (1) does fixing a hop reduce
answer entropy; (2) asymmetry |ΔA−ΔB| (rotating certainty moves residual uncertainty); (3)
base-vs-RL difference in how much certainty is absorbed (reactivity to injected certainty).

### `analyze.py`
Bootstrap 95% CIs (10k resamples), Fisher exact tests (Exp1 change rates), Wilcoxon signed-rank
(Exp3 paired entropy reductions), Mann-Whitney (cross-model). Writes
`results/evaluations/summary.json` and all figures.

## How to reproduce
```bash
uv venv && source .venv/bin/activate
uv add torch transformers accelerate scikit-learn scipy numpy pandas matplotlib seaborn
python -m src.exp1_reactivity --dataset arc --n 200
python -m src.exp2_probe --n 400
python -m src.exp3_rotating --n 120
python -m src.analyze
```
Seeds fixed at 42 (123 for Exp3 sampling). Hardware: NVIDIA RTX A6000 (bf16, ~3 GB/model).
Runtimes: Exp1 ≈ 20 min (3 models), Exp2 ≈ per-item forward passes, Exp3 ≈ sampling sweep.
```
```
