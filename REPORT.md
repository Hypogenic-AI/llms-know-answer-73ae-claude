# LLMs Know the Answer They Want to Give — Research Report

**Date:** 2026-06-22 · **Domain:** Artificial Intelligence · **Models:** Qwen2.5-1.5B family
(base / instruct / RL-reasoning), real local inference on NVIDIA RTX A6000.

---

## 1. Executive Summary

We tested whether **RL/post-trained LLMs pre-commit to the answer they will give and are
therefore less able to react to new information than base LLMs** — the submitter's hypothesis —
and whether **rotating which sub-fact is held certain redistributes a model's uncertainty**.
Using a matched post-training gradient of the *same* model family (base → instruct/RLHF →
RL-reasoning), three experiments measured (1) behavioral reactivity to new information, (2) the
internal earliness of answer commitment via hidden-state probes, and (3) uncertainty
redistribution under rotated certainty.

**The headline finding inverts the stated hypothesis.** It is the **base** model — not the
RL model — that most "knows the answer it wants to give." A linear probe decodes the base
model's final answer from its **prompt hidden state with 87% balanced accuracy** (chance 25%),
versus 71% for instruct and only **33% for the RL-reasoning model**, which decides its answer
*through* generation rather than pre-encoding it. Behaviorally, the base model is the **most
rigid**: it changes its answer only 2% of the time when handed a genuinely informative hint
(vs ~47% for both post-trained models, p≈10⁻²⁷). Post-training does **not** increase
pre-commitment — it **reduces** it. The two post-trained models differ in *how* they react:
instruct **over-reacts** (sycophantic, flips correct→incorrect 33–38% under content-free
pushback) while the RL-reasoning model is the most **selective** (reacts to good information,
resists empty pushback). For uncertainty (Exp 3), the base model's answer entropy is the most
**redistributable** by injected certainty (−0.32 bits when a hop is fixed, p≈10⁻¹³), while the
RL-reasoning model's is essentially **immovable** (≈0 bits, p=0.84).

**Practical implication:** the "pre-plans and can't react" property the hypothesis describes is
**real, but it is a property of base/pretrained models and of information that arrives *after*
the model has committed**. RL post-training largely *mitigates* it — at the cost (for plain
RLHF) of introducing the opposite failure, sycophantic over-reaction.

---

## 2. Research Question & Motivation

**Hypothesis (submitter).** RL-fine-tuned LLMs pre-plan the end of their turn and so react less
to new information than base LLMs; rotating what is held certain across a question may force the
model to distribute uncertainty across its parts.

Three separable claims:
- **C1 — Pre-planning exists:** the answer is internally decodable before it is emitted.
- **C2 — RL increases pre-commitment / reduces reactivity** (internal C2a + behavioral C2b).
- **C3 — Rotating certainty redistributes uncertainty** (novel; untested in the literature).

**Why it matters.** If post-training makes deployed models lock in their intended answer, they
would be the *least* able to absorb corrections or newly revealed facts mid-conversation — a
direct reliability/safety concern (the "lost-in-conversation" failure). **Gap:** prior work
establishes C1 (Emergent Response Planning; Future Lens) but never isolates RL with a matched
base contrast; reactivity work (UFO, SCoRe, FlipFlop) is purely behavioral and never ties to
internal pre-commitment; and the *direction* of the deficit is contested (RL→under-react vs.
widespread over-reaction). **C3 has no precedent.** We supply the missing base-vs-RL
pre-planning-probe contrast and the first test of C3.

---

## 3. Experimental Setup

### Models (matched post-training gradient, same lineage)
| key | HF id | post-training stage |
|-----|-------|---------------------|
| base | `Qwen/Qwen2.5-1.5B` | none (pretrained only) |
| instruct | `Qwen/Qwen2.5-1.5B-Instruct` | SFT + RLHF/DPO |
| rl | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | distilled from R1's RL reasoning traces |

Same architecture/tokenizer lineage → the systematic difference is post-training intensity.
Each model is run in its **native interface** (base = plain transcript, since no chat template
exists; instruct/rl = their chat templates, multi-turn). bf16, greedy decoding for commitment
measures, temperature 0.7 sampling for entropy. Seeds fixed (42; 123 for Exp 3 sampling).

### Datasets (pre-gathered, local)
ARC-Challenge (science MC, clean letter answers; Exp 1 & 2), GSM8K (numeric; Exp 1 robustness),
StrategyQA (decomposable yes/no with a `facts` chain; Exp 3).

### Experiments & metrics
- **Exp 1 — Reactivity (C2b).** Two-turn: model answers → feedback injected → answers again.
  Feedback ∈ {`generic` (UFO "that's incorrect, try again" — no info), `challenge` (FlipFlop
  "are you sure?"), `hint` (genuine information toward the correct answer)}. Metrics: answer
  change rate, flip rates i→c (good fix) / c→i (over-react). N=200 ARC items × 3 feedback types.
- **Exp 2 — Commitment-earliness probe (C1+C2a).** Greedy-generate → parse the model's *own*
  final letter. A balanced logistic-regression probe (80/20 split) predicts that letter from
  the **prompt-end** hidden state (all layers) and along the generation timeline. N=440 A–D
  ARC items. Prompt-position accuracy ≫ chance ⇒ answer pre-encoded.
- **Exp 3 — Rotating certainty (C3).** StrategyQA `facts` split into two hops (A, B).
  Conditions rotate which is held certain: `baseline`, `+A`, `+B`. Uncertainty = predictive
  binary entropy of the yes/no answer over K=6 samples (model-agnostic). N=120 items.

### Statistics
Bootstrap 95% CIs (10k resamples), Fisher exact (Exp 1), Wilcoxon signed-rank (Exp 3 paired
reductions), Mann-Whitney (cross-model). Raw outputs in `results/`, figures in `figures/`.

---

## 4. Results

### Exp 1 — Reactivity to new information (ARC, N=200)

Answer **change rate** turn1→turn2 (95% CI), and turn-1 accuracy:

| model | turn-1 acc | generic | challenge | hint |
|-------|-----------:|--------:|----------:|-----:|
| base | 0.65 | **0.17** [.12,.22] | 0.43 [.35,.50] | **0.02** [.01,.04] |
| instruct | 0.72 | 0.52 [.44,.59] | 0.56 [.49,.64] | 0.47 [.40,.54] |
| rl | 0.37 | 0.47 [.41,.53] | 0.43 [.36,.50] | 0.46 [.39,.54] |

Fisher exact: base vs instruct and base vs rl differ hugely under **generic** (p=2.6×10⁻¹³,
3.1×10⁻¹⁰) and **hint** (p=4.3×10⁻²⁸, 3.8×10⁻²⁷). **The base model barely revises — and almost
entirely ignores a genuinely informative hint appended after its answer (2% change).**

**Flip direction** (`figures/exp1_flipdir_arc.png`) reveals *how* each reacts:

| | generic i→c / c→i | hint i→c / c→i |
|---|---|---|
| base | 0.06 / 0.04 | 0.02 / 0.00 |
| instruct | 0.11 / **0.33** | 0.25 / 0.20 |
| rl | **0.24** / 0.11 | **0.30** / 0.08 |

Instruct (RLHF) **over-reacts**: under content-free pushback it flips correct→incorrect 33%
(challenge: 38%) — classic sycophancy. The RL-reasoning model is the most **selective**: it
fixes wrong answers (i→c) far more than it breaks correct ones (c→i), especially given a hint
(0.30 vs 0.08).

#### Exp 1 robustness — GSM8K (numeric, N=150)

turn-1 accuracy: base 0.22, instruct 0.67, **rl 0.77** (the math-distilled reasoning model is
best at math). Change rates (and c→i over-reaction):

| model | generic | challenge | hint* | dominant flip |
|-------|--------:|----------:|------:|---------------|
| base | 0.15 | 0.13 | 0.08 | — (rigid) |
| instruct | 0.16 | 0.18 | 0.07 | — (rigid) |
| rl | **0.57** | 0.51 | 0.56 | **c→i 0.35–0.39 (over-reacts)** |

\*The GSM8K "hint" is non-informative ("re-check your arithmetic") — unlike ARC's answer-pointing
hint — so it functions as a third content-free nudge; GSM8K therefore mainly probes reactivity to
*content-free* feedback. Fisher: base/instruct vs rl differ at p≤5.5×10⁻¹² on every condition;
base vs instruct is **not** significant (p>0.3). **Two findings replicate / extend:** (i) the
**base model is rigid on both tasks** (the most consistent result); (ii) the *direction* of
post-training's reactivity effect is **task- and correctness-dependent** — here the high-accuracy
RL-reasoning model **over-reacts**, re-deriving and breaking its (mostly correct) solutions under
empty pushback, the mirror image of its *selective* behavior on ARC.

### Exp 2 — Answer pre-encoding (ARC, N=440; chance = 0.25)

Probe accuracy predicting the model's **own** final answer:

| model | **prompt-position** acc (best layer) | self-consistency (acc vs gold) |
|-------|------------------------:|-------------------------------:|
| base | **0.873** | 0.67 |
| instruct | 0.712 | 0.68 |
| rl | **0.331** | 0.39 |

`figures/exp2_layer_sweep.png`: the base model's answer becomes strongly decodable from the
prompt hidden state in late layers (→0.87); instruct →0.71; **the RL-reasoning model stays near
chance at every layer** — its answer is not pre-encoded. `figures/exp2_timeline.png` (mid-layer
trajectory) shows base commits **early** (0.72 by 25% of generation), instruct commits **late**
(rises only after 75%), and rl's answer is decodable (mid-layer) only at the very end.
**Monotonic with post-training: more post-training ⇒ the answer is committed *later*.**

### Exp 3 — Rotating certainty (StrategyQA, N=120; entropy in bits)

| model | H(baseline) | H(+A) | H(+B) | ΔH(+A) | ΔH(+B) | Wilcoxon p |
|-------|------------:|------:|------:|-------:|-------:|-----------:|
| base | 0.722 | 0.404 | 0.471 | **+0.318** | **+0.250** | 1.8×10⁻¹³ |
| instruct | 0.420 | 0.341 | 0.327 | +0.079 | +0.093 | 0.019 |
| rl | 0.555 | 0.607 | 0.528 | −0.052 | +0.027 | 0.84 (n.s.) |

`figures/exp3_rotating.png`: fixing a hop **redistributes** uncertainty — strongly for the base
model (large, significant entropy drop; CIs clear of 0), marginally for instruct, and **not at
all** for the RL-reasoning model (its answer entropy is robust to injected sub-fact certainty;
it reasons through the question regardless). Base vs rl reduction: Mann-Whitney p=8.6×10⁻⁹.

---

## 5. Analysis & Discussion

**A coherent, three-experiment story that inverts the hypothesis.** The mechanism (Exp 2)
explains the behavior (Exp 1 & 3):

- The **base** model encodes its answer in the prompt representation before generating
  (Exp 2: 0.87). It is therefore behaviorally **rigid** — it ignores both empty pushback (0.17)
  and genuine hints (0.02) delivered *after* it has answered (Exp 1) — and yet its uncertainty
  is highly **shaped by what is in the prompt** (Exp 3: −0.32 bits when a fact is prepended).
  This is precisely "pre-plans from the prompt, can't react to information arriving mid-turn,"
  the property the hypothesis describes — but it belongs to the **base** model.
- **Post-training moves the commitment point later.** Instruct decides during generation
  (Exp 2 timeline), which makes it reactive — but plain RLHF makes it *indiscriminately* so:
  it **over-reacts**, abandoning correct answers under content-free challenge (Exp 1: c→i up to
  0.38). This is the sycophancy/FlipFlop failure, the *opposite* of pre-commitment.
- The **RL-reasoning** model commits latest of all (Exp 2: 0.33, ≈chance) and is the most
  **selective** reactor (Exp 1: high i→c, low c→i) and the most **robust** in its uncertainty
  (Exp 3: injecting one hop's certainty does not move its answer entropy — it reasons over the
  whole question). It "knows the answer it wants to give" the *least*.

**Reconciling Exp 1 and Exp 3 for the base model.** The base model ignores a hint *appended
after* its turn-1 answer (Exp 1) yet is strongly moved by a fact placed *in the prompt before
it answers* (Exp 3). Both follow from Exp 2: the answer is determined by the prompt
representation. Information in the prompt shapes the commitment; information after it does not.

**Task-dependence of the behavioral direction (from GSM8K).** Exp 1's *internal* mechanism
(Exp 2) is clean and monotonic, but the *behavioral* reactivity direction among post-trained
models depends on task and on whether the model was right to begin with. On ARC (where models
are often wrong and the hint is informative) the RL-reasoning model reacts *beneficially*; on
GSM8K (where it is usually right and feedback is content-free) the same model *over-reacts* and
breaks correct solutions. What is invariant across tasks is **base-model rigidity**. This is
exactly the "contested direction" the literature flagged — under- and over-reaction coexist and
are selected by conditions — so we report both rather than claiming a single direction.

**Relation to the literature.** This is the base-vs-RL pre-planning-probe contrast the review
flagged as missing, and it lands on the **mitigation** side of the contested-direction debate:
RL post-training reduces internal pre-commitment. We also reproduce both failure modes the
literature warns coexist — UFO/SCoRe-style under-reaction (here, the **base** model) and
FlipFlop-style over-reaction (here, **instruct**) — and show they sit at opposite ends of the
same post-training axis. On C3, rotating certainty **does** redistribute uncertainty, but the
effect is largest in the base model and vanishes in the reasoning model.

---

## 6. Limitations

- **One model family, one scale (1.5B).** Lineage is matched, but `instruct` and `rl` differ in
  *type* of post-training (RLHF vs reasoning distillation), not a clean RL ablation; results
  should be confirmed at 7B and on other families (SimpleRL-Zoo, Qwen-UFO).
- **Interface differs by model** (base=transcript, others=chat) because base models have no chat
  mode — inherent to the comparison, but a confound for Exp 1 absolute rates. The internal probe
  (Exp 2), measured identically, is the cleaner contrast and agrees.
- **RL-reasoning model's low ARC accuracy (37%)** reflects a math-distilled model on science MC
  under a 640-token budget (reasoning sometimes truncated); the probe targets the model's *own*
  answer so this does not bias Exp 2, but turn-1 accuracy should be read with care.
- **Exp 3 entropy from K=6 samples** is coarse per item; we rely on 120-item aggregates and
  paired tests (robust), and replaced the noise-inflated per-item |ΔH| metric with signed
  bootstrap-CI reductions.
- **Decoding budget** (greedy, fixed max tokens) and StrategyQA fact-splitting (sentence-level)
  are heuristics; parse-failure rates are logged (≤6%, mostly the reasoning model) and reported,
never silently dropped.

---

## 7. Conclusions & Next Steps

**Answer to the research question.** Within this matched family, the hypothesis is **inverted**:
RL/post-training does **not** increase answer pre-commitment or reduce reactivity — it **reduces**
internal pre-commitment (answer decoded from the prompt: base 0.87 → instruct 0.71 → RL 0.33)
and **increases** reactivity. The "knows-the-answer / can't-react" property is real but belongs
to the **base** model (rigid on both ARC and GSM8K) and to information arriving *after*
commitment. Among post-trained models the reactivity *direction* is task- and correctness-
dependent — beneficially selective on ARC, over-reactive on GSM8K — so the deficit has no single
direction, only one invariant: base-model rigidity.
**Rotating certainty does redistribute uncertainty (C3 confirmed), but only where the answer is
prompt-driven — strongly in the base model, not in the reasoning model.**

**Next steps:** (1) replicate at 7B and on a clean RL ablation (Qwen2.5-base vs SimpleRL-Zoo,
vs Qwen-UFO); (2) causally steer the late-layer "answer direction" found in Exp 2 to test
whether forcing earlier commitment *induces* base-like rigidity; (3) extend C3 to multi-hop QA
(HotpotQA) and test whether rotating certainty can be used as a *decoding-time* intervention to
surface a reasoning model's residual per-hop uncertainty.

---

## References (used resources)
Emergent Response Planning in LLMs (2502.06258); Future Lens (2311.04897); Reasoning-Strength
Planning (2506.08390); "Try Again"/UFO (2507.14295); SCoRe (2409.12917); FlipFlop (2311.08596);
Uncertainty Profiles (2505.07309). Datasets: ARC-Challenge, GSM8K, StrategyQA. Tooling:
PyTorch 2.12, Transformers 5.12, scikit-learn 1.9, SciPy 1.18. Full literature synthesis in
`literature_review.md`; resource catalog in `resources.md`.
