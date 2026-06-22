# Research Plan: LLMs Know the Answer They Want to Give

## Research Question

Do RL/post-trained LLMs **pre-commit** to the answer they will give — encoding it early
and resisting revision when new information arrives — *more* than their base counterparts?
And can we **rotate which sub-answer is held certain** across a decomposable question to
**redistribute the model's uncertainty** onto the remaining parts?

The hypothesis (from the submitter) has three separable claims:
- **C1 — Pre-planning exists:** the final answer is decodable from internal state *before*
  it is emitted.
- **C2 — RL increases pre-commitment / reduces reactivity:** post-trained (instruct / RL-
  reasoning) models commit earlier and revise less under new information than the base model.
- **C3 — Rotating certainty redistributes uncertainty (novel):** fixing one sub-fact of a
  multi-hop question as "given" measurably moves the model's residual uncertainty onto the
  other hop, and RL models are less able to perform this redistribution.

---

## Motivation & Novelty Assessment

### Why This Research Matters
If RL post-training makes models "lock in" the answer they intend to give, then exactly the
models we deploy for high-stakes interaction are the *least* able to incorporate corrections,
hints, or newly revealed facts mid-conversation — a direct reliability and safety problem
(the "lost-in-conversation" failure). Understanding *whether* the deficit is mechanistic
(early internal commitment) and *which direction* it runs (under-react/repeat vs.
over-react/flip) tells us how to intervene.

### Gap in Existing Work (from literature_review.md)
- **C1** is well established (Emergent Response Planning, Reasoning-Strength Planning, Future
  Lens) but those papers do **not isolate RL** — they don't contrast matched base vs RL models.
- **C2** is studied *behaviorally* (UFO "Try Again", SCoRe, FlipFlop) but **not tied to internal
  pre-commitment probes**, and the deficit *direction is contested* (RL→under-react vs.
  widespread over-reaction).
- **C3** has **no direct precedent**: no work rotates what is held certain across a question to
  redistribute uncertainty.

### Our Novel Contribution
1. A **clean matched base→instruct→RL-reasoning contrast** (Qwen2.5-1.5B family) on the *same*
   items, measuring both (a) internal answer-commitment earliness (probe) and (b) behavioral
   reactivity to new information — tying C1 to C2 in one study.
2. The **first test of C3**: a "rotating certainty" protocol on decomposable StrategyQA/HotpotQA
   items, measuring whether injecting certainty about one hop redistributes uncertainty onto the
   other, and whether RL models redistribute less.

### Experiment Justification
- **Exp 1 (Behavioral reactivity, C2):** directly measures "less ability to react to new
  information" — the core claim — and captures *both* over- and under-reaction (the contested
  direction the literature warns about).
- **Exp 2 (Commitment-earliness probe, C1+C2):** supplies the missing mechanistic link — *is*
  the answer encoded earlier in RL models? Reuses the Emergent-Response-Planning probing protocol.
- **Exp 3 (Rotating certainty, C3):** tests the novel, untested claim with real models and a
  logit-entropy measure of where uncertainty lives.

---

## Hypothesis Decomposition

| Claim | Operationalization | Prediction if hypothesis true |
|-------|--------------------|-------------------------------|
| C1 | Linear probe on prompt-end hidden state predicts the model's own final answer | probe acc ≫ chance even at prompt position |
| C2a (internal) | probe accuracy at the *prompt* position, base vs RL | RL > base (commits earlier) |
| C2b (behavioral) | answer change-rate under generic feedback / challenge / hint | RL revises less to generic feedback; base more reactive |
| C3 | answer-token entropy under {baseline, fix-hop-A, fix-hop-B} | fixing a hop lowers entropy; reduction is *asymmetric* by hop; RL redistributes less |

**Independent variables:** model post-training stage (base / instruct / RL-reasoning); feedback
type; which sub-fact is fixed; generation position.
**Dependent variables:** probe accuracy, answer change/flip rates (i→c, c→i), answer entropy.
**Controls:** identical prompts/items across models; same decoding; greedy for commitment,
fixed seed for sampling; same probe architecture and train/test split.

---

## Proposed Methodology

### Models (matched post-training gradient, same family)
- `Qwen/Qwen2.5-1.5B` — **base** (no RLHF)
- `Qwen/Qwen2.5-1.5B-Instruct` — **instruct** (SFT + RLHF/DPO)
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` — **RL-reasoning** (distilled from R1's RL traces)
- (scale check if time permits: the 7B trio)

Rationale: same tokenizer/architecture lineage → the only systematic difference is post-training
intensity, isolating the variable the hypothesis is about. Local HF weights are **required**
because true *base* models are not served by any chat API.

### Datasets
- **ARC-Challenge** (1,172 MC) — clean single-letter answer extraction → Exp 1 & 2.
- **GSM8K** (1,319) — numeric answers → Exp 1 robustness.
- **StrategyQA** (687, yes/no, has decomposition `facts`) — decomposable → Exp 3.
- **HotpotQA** (2,000, 2-hop, comparison/bridge) → Exp 3 robustness.

### Experimental Steps
1. **Exp 1 — Reactivity.** Transcript-style prompt (no chat template, fair across base/RL).
   Turn 1 answer → inject feedback ∈ {generic-wrong (UFO), challenge "are you sure?" (FlipFlop),
   correct-hint (real new info)} → Turn 2 answer. Compute per-model change rate, flip rates
   (i→c, c→i), unique-answer ratio. N≈300 ARC + 200 GSM8K.
2. **Exp 2 — Commitment probe.** Greedy-generate answer; capture last-token hidden states at
   prompt-end + first K generated tokens; train logistic-regression probe to predict the model's
   own final answer letter; report earliness curve and prompt-position accuracy. N≈600 items,
   80/20 split, mid-layer + last-layer.
3. **Exp 3 — Rotating certainty.** For decomposable items with two facts (A,B): conditions
   {baseline, +A given, +B given}. Measure answer-token entropy (binary for yes/no, over choices
   for MC). Test entropy reduction and its asymmetry; compare across models. N≈300 StrategyQA.

### Baselines
- Chance / majority for probes; the model's own verbalized confidence where applicable.
- Within-study baseline = the **base model** for every RL comparison.
- Exp 1 grounded in UFO (unique-answer ratio) & FlipFlop (flip rate) metrics; Exp 2 in
  Emergent-Response-Planning probe; Exp 3 vs. the no-fact baseline condition.

### Evaluation Metrics
- **Reactivity:** answer-change rate, flip rates Δ^{i→c}/Δ^{c→i}, unique-answer ratio.
- **Pre-commitment:** probe balanced accuracy vs. position (earliness), AUC of earliness curve.
- **Uncertainty:** answer-token entropy; entropy reduction Δ when a hop is fixed; asymmetry index.

### Statistical Analysis Plan
- Paired comparisons across items: **Wilcoxon signed-rank** (non-normal rates/entropies) and
  McNemar for paired binary change. Bootstrap 95% CIs (10k resamples). Effect sizes (Cohen's d /
  rank-biserial). Significance α=0.05; Holm correction across the model×condition family.
- Multiple seeds for sampling-based entropy; report mean ± std.

## Expected Outcomes
- **Supports hypothesis:** RL models show higher prompt-position probe accuracy (earlier
  commitment), lower change-rate to generic feedback, and smaller / more rigid uncertainty
  redistribution in Exp 3.
- **Refutes / complicates:** if base models over-react (FlipFlop pattern) while RL models revise
  *appropriately*, the deficit is direction-dependent — we report it honestly via the i→c vs c→i
  split.

## Timeline and Milestones
1. Env + smoke-test model load (15 min) 2. Shared utils + Exp 1 (45 min) 3. Exp 2 probe (45 min)
4. Exp 3 (40 min) 5. Analysis + figures (40 min) 6. REPORT.md + README (30 min). Buffer 25%.

## Potential Challenges
- **Base-model formatting:** mitigated by a uniform transcript prompt + few-shot exemplars and
  robust answer parsing (regex + last-mention heuristics).
- **Answer extraction failures:** log parse-failure rate; exclude only with documentation.
- **GPU memory / busy GPU:** GPU 2 is busy; pin to GPUs 0/1/3; use bf16; 1.5B fits easily.
- **Contested C2 direction:** designed-for — both flip directions measured.

## Success Criteria
A complete, reproducible run of all three experiments on real local models with statistical
tests, an honest base-vs-RL characterization (including null/contradictory results), and
REPORT.md documenting actual findings.
