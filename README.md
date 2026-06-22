# LLMs Know the Answer They Want to Give

Do RL/post-trained LLMs *pre-commit* to their answer and react less to new information than base
LLMs? We test this with a matched **base → instruct(RLHF) → RL-reasoning** gradient of the
Qwen2.5-1.5B family across three experiments using **real local models** (no simulation).

## Key findings (the hypothesis is inverted)

- **The *base* model most "knows the answer it wants to give."** A linear probe decodes its final
  answer from the **prompt hidden state at 87%** (chance 25%); instruct 71%; **RL-reasoning only
  33%** — the reasoning model decides *through* generation, not before it. (Exp 2)
- **The base model is the most rigid behaviorally** — it changes its answer just **2%** of the
  time given a genuinely informative hint, vs ~47% for both post-trained models (p≈10⁻²⁷). (Exp 1)
- **Post-trained models react, but the direction is task-dependent** — on ARC, RLHF over-reacts
  (sycophantic c→i 33–38%) while the reasoning model is selective; on GSM8K the reasoning model
  over-reacts. The one invariant across both tasks is **base-model rigidity**. (Exp 1, ARC+GSM8K)
- **Rotating which sub-fact is held certain redistributes uncertainty** — strongly for the base
  model (−0.32 bits, p≈10⁻¹³), marginally for instruct, **not at all** for the reasoning model
  (p=0.84). (Exp 3, novel)
- **Takeaway:** RL post-training *reduces* internal pre-commitment and *increases* reactivity;
  the "pre-plans, can't react" property belongs to base models and to info arriving after commit.

See **[REPORT.md](REPORT.md)** for full results, statistics, and discussion.

## Reproduce
```bash
uv venv && source .venv/bin/activate
uv add torch transformers accelerate scikit-learn scipy numpy pandas matplotlib seaborn
python -m src.exp1_reactivity --dataset arc --n 200      # behavioral reactivity (C2)
python -m src.exp2_probe       --n 450                    # commitment-earliness probe (C1+C2a)
python -m src.exp3_rotating    --n 120                    # rotating certainty (C3)
python -m src.analyze                                     # stats + figures
```
Hardware: NVIDIA RTX A6000 (bf16). Seeds fixed (42 / 123). Runtime ≈ 45 min total.

## Structure
```
src/            experiment + analysis code (see CODE_WALKTHROUGH.md)
datasets/       ARC-Challenge, GSM8K, StrategyQA, HotpotQA (pre-gathered)
results/        model_outputs/ (raw) + evaluations/ (summary.json, probe results)
figures/        exp1_reactivity_arc, exp1_flipdir_arc, exp2_layer_sweep, exp2_timeline, exp3_rotating
planning.md     Phase 0/1 plan + motivation & novelty
REPORT.md       full research report   ·   literature_review.md / resources.md  pre-gathered
```
